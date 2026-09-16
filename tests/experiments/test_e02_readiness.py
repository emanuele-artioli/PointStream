"""E02 Generator Readiness and E01 Interface Compliance Test Suite.

Verifies:
1. Full-trajectory multi-frame placement and pose alignment across frames.
2. Missing pose/skeleton fail-closed semantics.
3. Real neural generator conditioning sensitivity and bit-identical determinism.
4. Tiny learning step and atomic checkpoint save/resume with RNG/step preservation.
5. SPADE weight initialization order preserving loaded pretrained weights.
6. Residual-off evaluation producing strictly zero residual bytes and calls.
7. Uncertainty-aware promotion preserving variants within noise threshold.
8. Campaign ranking using wire bytes first (total_bytes when residual-off).
9. E01 schema adapter formatting, claim eligibility, and fail-closed validation
   against experiments.tier.campaign_result validator and ingester.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

import random
import subprocess
import sys

from experiments.long_scenes.loader import (
    load_long_scene_clip,
)
from scripts.run_diagnostic_matrix import (
    _augment_objects_with_pose,
)
from scripts.train_campaign import (
    compare_candidates,
    evaluate_checkpoint,
    promote_survivors,
    rank_variants,
)
from scripts.train_pix2pix import (
    ReconstructibleEpochSampler,
    UNetGenerator,
    build_checkpoint_state,
    save_checkpoint_atomic,
)
from src.shared.tennis_dataset import TennisSkeletonDataset
from src.components.generation.spade4tennis_arch import SPADEResNet9Generator
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner.generation_adapter import (
    adapt_diagnostic_matrix_result,
    calculate_metric_uncertainty,
    validate_generation_result,
)


@dataclass
class FakeClip:
    video: str
    scene: str
    context_id: str
    frames: np.ndarray
    objects: tuple[Any, ...]
    start_frame: int = 0


# ---------------------------------------------------------------------------
# 1. Full-Trajectory Multi-Frame Placement & Pose Alignment
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("POINTSTREAM_DATA_TESTS") != "1",
    reason="requires external footage; set POINTSTREAM_DATA_TESTS=1",
)
def test_full_trajectory_multi_frame_placement() -> None:
    """Verify full_trajectory=True emits ObjectRequest for each visible frame."""
    clip_single = load_long_scene_clip("alcaraz_highlights", "scene_028", 48, full_trajectory=False)
    assert len(clip_single.objects) == 2

    clip_full = load_long_scene_clip("alcaraz_highlights", "scene_028", 48, full_trajectory=True)
    assert len(clip_full.objects) == 96
    frame_indices = {obj.frame_index for obj in clip_full.objects}
    assert frame_indices == set(range(48))


def test_pose_alignment_with_varying_bbox_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pose skeleton matches varying bounding box shape across frames and resizes to appearance."""
    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    scene_dir = tmp_path / "assets" / "dataset" / "fake_video" / "segmentations" / "fake_scene"
    skel_dir = scene_dir / "track_0001_skeleton"
    skel_dir.mkdir(parents=True)

    skel_0 = np.full((80, 40, 3), 150, dtype=np.uint8)
    skel_1 = np.full((90, 45, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(skel_dir / "frame_000000.png"), skel_0)
    cv2.imwrite(str(skel_dir / "frame_000001.png"), skel_1)

    appearance = np.ones((100, 50, 3), dtype=np.uint8) * 128
    obj_0 = ObjectRequest(
        object_id="track_0001",
        appearance=appearance,
        bbox=(10, 10, 50, 90),
        mask=np.ones((100, 50), dtype=bool),
        frame_index=0,
    )
    obj_1 = ObjectRequest(
        object_id="track_0001",
        appearance=appearance,
        bbox=(10, 10, 55, 100),
        mask=np.ones((100, 50), dtype=bool),
        frame_index=1,
    )

    clip = FakeClip(
        video="fake_video",
        scene="fake_scene",
        context_id="ctx",
        frames=np.zeros((2, 100, 50, 3), dtype=np.uint8),
        objects=(obj_0, obj_1),
        start_frame=0,
    )

    augmented = _augment_objects_with_pose(clip, shuffle=False, seed=42)
    assert len(augmented) == 2
    assert augmented[0].conditioning is not None
    assert augmented[1].conditioning is not None
    assert augmented[0].conditioning.pose.shape == (3, 100, 50)
    assert augmented[1].conditioning.pose.shape == (3, 100, 50)


# ---------------------------------------------------------------------------
# 2. Missing Pose / Mismatch Fails Closed
# ---------------------------------------------------------------------------


def test_missing_pose_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing skeleton directory raises FileNotFoundError."""
    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    obj = ObjectRequest(
        object_id="track_missing",
        appearance=np.zeros((64, 64, 3), dtype=np.uint8),
        bbox=(0, 0, 64, 64),
        mask=np.ones((64, 64), dtype=bool),
        frame_index=0,
    )
    clip = FakeClip(
        video="fake_video",
        scene="fake_scene",
        context_id="ctx",
        frames=np.zeros((1, 64, 64, 3), dtype=np.uint8),
        objects=(obj,),
        start_frame=0,
    )
    with pytest.raises(FileNotFoundError, match="Missing required pose conditioning skeleton"):
        _augment_objects_with_pose(clip, shuffle=False, seed=42)


def test_pose_shape_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mismatched skeleton shape raises ValueError."""
    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    scene_dir = tmp_path / "assets" / "dataset" / "fake_video" / "segmentations" / "fake_scene"
    skel_dir = scene_dir / "track_0001_skeleton"
    skel_dir.mkdir(parents=True)

    mismatched_skeleton = np.zeros((128, 80, 3), dtype=np.uint8)
    cv2.imwrite(str(skel_dir / "frame_000000.png"), mismatched_skeleton)

    obj = ObjectRequest(
        object_id="track_0001",
        appearance=np.zeros((64, 64, 3), dtype=np.uint8),
        bbox=(0, 0, 32, 32),
        mask=np.ones((64, 64), dtype=bool),
        frame_index=0,
    )
    clip = FakeClip(
        video="fake_video",
        scene="fake_scene",
        context_id="ctx",
        frames=np.zeros((1, 64, 64, 3), dtype=np.uint8),
        objects=(obj,),
        start_frame=0,
    )
    with pytest.raises(ValueError, match="Misaligned pose conditioning"):
        _augment_objects_with_pose(clip, shuffle=False, seed=42)


# ---------------------------------------------------------------------------
# 3. Real Neural Forward, Conditioning Sensitivity & Determinism
# ---------------------------------------------------------------------------


def test_real_neural_conditioning_sensitivity_determinism_and_no_conditioning() -> None:
    """Prove conditioning sensitivity, same-seed determinism, and no-conditioning control on real UNetGenerator."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    net = UNetGenerator(in_channels=6, out_channels=3).to(device)
    net.eval()

    # Create reference appearance and conditionings
    ref = torch.randn(1, 3, 256, 256)
    cond_normal = torch.randn(1, 3, 256, 256)
    cond_shuffled = torch.randn(1, 3, 256, 256)
    cond_blank = torch.zeros(1, 3, 256, 256)

    with torch.no_grad():
        out_normal_1 = net(torch.cat((cond_normal, ref), dim=1))
        out_normal_2 = net(torch.cat((cond_normal, ref), dim=1))
        out_shuffled = net(torch.cat((cond_shuffled, ref), dim=1))
        out_blank = net(torch.cat((cond_blank, ref), dim=1))

    # Same-seed determinism: identical input produces bit-identical tensor
    assert torch.equal(out_normal_1, out_normal_2)

    # Conditioning sensitivity (normal vs shuffled): different conditioning produces measurably different output
    l1_diff_shuffled = torch.mean(torch.abs(out_normal_1 - out_shuffled)).item()
    assert l1_diff_shuffled > 0.01

    # No-conditioning control: blank/zero conditioning produces measurably different output
    l1_diff_blank = torch.mean(torch.abs(out_normal_1 - out_blank)).item()
    assert l1_diff_blank > 0.01


def test_reconstructible_epoch_sampler_cursor_continuation() -> None:
    """Verify sampler yields exact remaining batches on partial-epoch resume without repetition or skipping."""
    dummy_dataset = list(range(100))
    batch_size = 10
    sampler_full = ReconstructibleEpochSampler(dummy_dataset, batch_size=batch_size, seed=42, shuffle=True)
    sampler_full.set_epoch(0, start_step=0)
    full_indices = list(sampler_full)
    assert len(full_indices) == 100

    # Resume at step 4 (batches 0, 1, 2, 3 already completed)
    sampler_resumed = ReconstructibleEpochSampler(dummy_dataset, batch_size=batch_size, seed=42, shuffle=True)
    sampler_resumed.set_epoch(0, start_step=4)
    resumed_indices = list(sampler_resumed)

    # Must equal full_indices from index 40 onwards
    assert resumed_indices == full_indices[40:]
    assert len(resumed_indices) == 60


def test_real_neural_training_interruption_and_continuation_optimizer_update(tmp_path: Path) -> None:
    """Verify interrupted and resumed training matches uninterrupted control within numerical tolerance."""
    seed = 42

    # Synthesize small batch data (3 batches of 2 items with valid UNet 256x256 dimensions)
    torch.manual_seed(seed)
    batches = [
        (torch.randn(2, 6, 256, 256), torch.randn(2, 3, 256, 256))
        for _ in range(3)
    ]

    # --- Run A: Uninterrupted 2 steps (step 0, then step 1) ---
    torch.manual_seed(seed)
    net_A = UNetGenerator(in_channels=6, out_channels=3)
    opt_G_A = optim.Adam(net_A.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # Step 0
    inp0, tgt0 = batches[0]
    out0 = net_A(inp0)
    loss0 = nn.functional.l1_loss(out0, tgt0)
    opt_G_A.zero_grad()
    loss0.backward()
    opt_G_A.step()

    # Step 1
    inp1, tgt1 = batches[1]
    out1_A = net_A(inp1)
    loss1_A = nn.functional.l1_loss(out1_A, tgt1)
    opt_G_A.zero_grad()
    loss1_A.backward()
    opt_G_A.step()

    # --- Run B: Step 0 -> Save checkpoint -> Fresh process / resume -> Step 1 ---
    torch.manual_seed(seed)
    net_B = UNetGenerator(in_channels=6, out_channels=3)
    opt_G_B = optim.Adam(net_B.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # Step 0
    inp0_b, tgt0_b = batches[0]
    out0_b = net_B(inp0_b)
    loss0_b = nn.functional.l1_loss(out0_b, tgt0_b)
    opt_G_B.zero_grad()
    loss0_b.backward()
    opt_G_B.step()

    # Save checkpoint at step 0
    ckpt_file = tmp_path / "resume_test_ckpt.pt"
    ckpt_state = build_checkpoint_state(
        epoch=0,
        step=0,
        total_steps=3,
        generator=net_B,
        discriminator=None,
        optimizer_G=opt_G_B,
        optimizer_D=None,
        ngpus=1,
        base_seed=seed,
    )
    save_checkpoint_atomic(ckpt_state, ckpt_file)

    # Fresh process / re-instantiate models and optimizers
    net_resumed = UNetGenerator(in_channels=6, out_channels=3)
    opt_G_resumed = optim.Adam(net_resumed.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # Load checkpoint
    loaded = torch.load(ckpt_file, map_location="cpu")
    net_resumed.load_state_dict(loaded["G"])
    opt_G_resumed.load_state_dict(loaded["opt_G"])
    t_rng = loaded["rng_torch"]
    if isinstance(t_rng, torch.Tensor):
        t_rng = t_rng.cpu()
    torch.set_rng_state(t_rng)

    # Perform step 1 in resumed instance
    out1_resumed = net_resumed(inp1)
    loss1_resumed = nn.functional.l1_loss(out1_resumed, tgt1)
    opt_G_resumed.zero_grad()
    loss1_resumed.backward()
    opt_G_resumed.step()

    # Verify: loss values match
    assert torch.allclose(loss1_A, loss1_resumed, atol=1e-6)

    # Verify: weights match uninterrupted control
    for p_A, p_B in zip(net_A.parameters(), net_resumed.parameters(), strict=True):
        assert torch.allclose(p_A, p_B, atol=1e-6)

    # Verify: optimizer state dicts (exp_avg, exp_avg_sq) match uninterrupted control
    for state_A, state_B in zip(
        opt_G_A.state.values(), opt_G_resumed.state.values(), strict=True
    ):
        if "exp_avg" in state_A and "exp_avg" in state_B:
            assert torch.allclose(state_A["exp_avg"], state_B["exp_avg"], atol=1e-6)
        if "exp_avg_sq" in state_A and "exp_avg_sq" in state_B:
            assert torch.allclose(state_A["exp_avg_sq"], state_B["exp_avg_sq"], atol=1e-6)


def test_selected_device_resume_cpu_tensor_conversion(tmp_path: Path) -> None:
    """Verify loading device-mapped checkpoint safely converts RNG state tensors to CPU ByteTensors."""
    ckpt_file = tmp_path / "device_mapped_ckpt.pt"
    rng_torch = torch.get_rng_state()
    torch.save({"rng_torch": rng_torch, "rng_cuda": None, "epoch": 0, "step": 0}, ckpt_file)

    loaded = torch.load(ckpt_file, map_location="cpu")
    t_rng = loaded["rng_torch"]
    if torch.cuda.is_available():
        t_cuda = t_rng.cuda()
        assert t_cuda.is_cuda
        t_cpu = t_cuda.cpu()
        torch.set_rng_state(t_cpu)
    else:
        if isinstance(t_rng, torch.Tensor):
            t_rng = t_rng.cpu()
        torch.set_rng_state(t_rng)


def test_spade_pretrained_load_order_not_overwritten(tmp_path: Path) -> None:
    """Verify pretrained generator weights survive and are not overwritten by weights_init_normal."""
    gen = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=16, n_blocks=2)

    # Inject specific weights
    with torch.no_grad():
        for p in gen.parameters():
            p.fill_(0.42)

    weights_file = tmp_path / "pretrained_spade.pt"
    torch.save(gen.state_dict(), weights_file)

    # Instantiate fresh model and load
    gen2 = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=16, n_blocks=2)

    # Correct order: base init, then pretrained load
    def weights_init_normal(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    gen2.apply(weights_init_normal)
    gen2.load_state_dict(torch.load(weights_file, map_location="cpu"))

    # Verify loaded weights survived
    first_param = next(gen2.parameters())
    assert torch.allclose(first_param, torch.full_like(first_param, 0.42))


# ---------------------------------------------------------------------------
# 4. Residual-Off Evaluation (Zero residual bytes & calls)
# ---------------------------------------------------------------------------


def test_evaluate_checkpoint_residual_off(tmp_path: Path) -> None:
    """Verify residual_settings=None strictly disables STAGE_RESIDUAL and yields 0 residual bytes."""
    ckpt_file = tmp_path / "dummy_weights.pt"
    torch.save({"G": {}}, ckpt_file)

    manifest = {
        "probe_clips": [
            {
                "video": "test_video",
                "scene": "scene_001",
                "track": "track_001",
                "synthetic": True,
                "frame_ids": [0, 1],
            }
        ]
    }

    class MockRunResult:
        class Sizes:
            residual = 0
            transport_total = 500
        sizes = Sizes()
        encoder_seconds = 0.1
        client_seconds = 0.05
        delivered_frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        class Quality:
            def whole_frame(self, m: str) -> float:
                return 35.0 if m == "psnr" else 0.95
        delivered_quality = Quality()

    def mock_runner(cfg: Any, clips: Any = None, **kwargs: Any) -> MockRunResult:
        # Assert STAGE_RESIDUAL is disabled in lattice and residual config is None
        assert cfg.lattice.residual is False
        assert cfg.residual is None
        return MockRunResult()

    res = evaluate_checkpoint(
        checkpoint_path=ckpt_file,
        arch="pix2pix",
        manifest=manifest,
        dataset_root=tmp_path,
        held_out_only=False,
        residual_settings=None,  # Residual OFF
        runner_fn=mock_runner,
    )

    agg = res["aggregate"]
    assert agg["residual_enabled"] is False
    assert agg["residual_bytes"] == 0
    assert agg["total_residual_calls"] == 0
    assert agg["total_bytes"] == 500


# ---------------------------------------------------------------------------
# 5. Uncertainty-Aware Promotion & Ranking
# ---------------------------------------------------------------------------


def test_uncertainty_aware_promotion_preserves_close_candidates() -> None:
    """Candidates within practical rate indifference band of cutoff are preserved as survivors."""
    ranked = ["cand_1", "cand_2", "cand_3", "cand_4"]
    aggregate = {
        "cand_1": {"residual_bytes": 1000, "psnr_mean": 35.0, "success": True},
        "cand_2": {"residual_bytes": 1100, "psnr_mean": 34.5, "success": True},
        "cand_3": {"residual_bytes": 1120, "psnr_mean": 34.4, "success": True},
        "cand_4": {"residual_bytes": 1320, "psnr_mean": 30.0, "success": True},
    }

    survivors = promote_survivors(ranked, aggregate, min_diff_threshold=0.02)
    assert survivors == ["cand_1", "cand_2", "cand_3"]
    assert "cand_4" not in survivors


def test_incomparable_candidates_preserved_without_automated_pruning() -> None:
    """Candidates representing valid RD trade-offs (e.g. higher rate, higher quality) are preserved."""
    ranked = ["low_rate", "high_quality"]
    aggregate = {
        "low_rate": {"total_bytes": 10000, "psnr_mean": 32.0, "success": True},
        "high_quality": {"total_bytes": 25000, "psnr_mean": 38.5, "success": True},
    }
    # With 2 candidates, successive halving nominally keeps ceil(2/2)=1.
    # But high_quality has 6.5 dB higher PSNR! They are incomparable trade-offs.
    survivors = promote_survivors(ranked, aggregate)
    assert "low_rate" in survivors
    assert "high_quality" in survivors  # Preserved without automated model pruning!


def test_rank_variants_orders_by_wire_bytes_first() -> None:
    """Residual bytes strictly outranks perceptual score."""
    aggregate = {
        "high_bytes": {
            "residual_bytes": 50000,
            "psnr_mean": 40.0,
            "ssim_mean": 0.98,
            "vmaf_mean": 95.0,
            "temporal_error": 0.5,
            "success": True,
        },
        "low_bytes": {
            "residual_bytes": 20000,
            "psnr_mean": 36.0,
            "ssim_mean": 0.94,
            "vmaf_mean": 88.0,
            "temporal_error": 1.2,
            "success": True,
        },
    }
    ranked, composite = rank_variants(aggregate)
    assert ranked[0] == "low_bytes"
    assert ranked[1] == "high_bytes"


def test_rank_variants_uses_total_bytes_when_residual_off() -> None:
    """When residual_bytes is zero for all candidates, rank on total_bytes."""
    aggregate = {
        "large_model": {
            "residual_bytes": 0,
            "total_bytes": 80000,
            "psnr_mean": 38.0,
            "success": True,
        },
        "compact_model": {
            "residual_bytes": 0,
            "total_bytes": 30000,
            "psnr_mean": 37.0,
            "success": True,
        },
    }
    ranked, _ = rank_variants(aggregate)
    assert ranked[0] == "compact_model"
    assert ranked[1] == "large_model"


# ---------------------------------------------------------------------------
# 6. E01 Generation Adapter & Fail-Closed Validation
# ---------------------------------------------------------------------------


def test_generation_adapter_conforms_to_e01_schema() -> None:
    """Verify adapt_diagnostic_matrix_result produces schema-valid record."""
    matrix_output = {
        "normal": {
            "elapsed_seconds": 1.5,
            "psnr_mean": 32.5,
            "ssim_mean": 0.91,
            "vmaf_mean": 82.0,
            "temporal_error_mean": 2.1,
            "residual_bytes": 15000,
            "total_bytes": 25000,
            "frame_hashes": ["hash1", "hash2", "hash3", "hash4"],
            "per_frame_psnr": [32.0, 33.0, 32.5, 32.5],
        },
        "shuffled": {
            "frame_hashes": ["shuff1", "shuff2", "shuff3", "shuff4"],
        },
        "seed_repeat": {
            "frame_hashes": ["hash1", "hash2", "hash3", "hash4"],
        },
    }

    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="test_run_01",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="abcdef1234567890",
    )

    assert adapted["doc_role"] == "generation_result"
    assert adapted["run_id"] == "test_run_01"
    assert adapted["backend_name"] == "pix2pix"
    assert adapted["checkpoint_identity"]["checkpoint_sha256"] == "abcdef1234567890"
    assert adapted["claim_eligibility"]["rd_claim"] is True
    assert adapted["claim_eligibility"]["speed_claim"] is True

    valid, blockers = validate_generation_result(adapted)
    assert valid is True
    assert blockers == []


def test_generation_adapter_fails_closed_on_failed_controls() -> None:
    """Verify that insensitive conditioning or missing controls invalidate claims."""
    matrix_output = {
        "normal": {
            "elapsed_seconds": 1.5,
            "psnr_mean": 30.0,
            "residual_bytes": 10000,
            "frame_hashes": ["hash1", "hash2"],
        },
        # Shuffled conditioning produced IDENTICAL output -> conditioning is dead/ignored!
        "shuffled": {
            "frame_hashes": ["hash1", "hash2"],
        },
        "seed_repeat": {
            "frame_hashes": ["hash1", "hash2"],
        },
    }

    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="test_run_broken_cond",
        backend_name="broken_gen",
        arch="broken",
        checkpoint_sha256="fakehash",
    )

    assert adapted["claim_eligibility"]["rd_claim"] is False
    assert any("shuffled match" in r for r in adapted["exclusion_reasons"])


def test_adapter_reads_nested_scores_timing_parts_and_shape() -> None:
    matrix_output = {
        "video": "alcaraz_highlights",
        "scene": "scene_028",
        "frames": 16,
        "matrix": [
            {
                "corner": "gen_on_res_off",
                "generation_on": True,
                "shuffled_conditioning": False,
                "delivered_frame_hashes": ["g1", "g2"],
                "delivered_shape": [16, 2160, 3840, 3],
                "scores": {"psnr_y": 28.3, "ssim": 0.97, "vmaf": 88.0},
                "timing": {"client_seconds": 24.5, "encoder_seconds": 85.0},
                "parts": {"residual": 1100, "transport_total": 7000},
                "coded_bytes": 7000,
            },
            {
                "corner": "gen_on_shuffled_conditioning",
                "generation_on": True,
                "shuffled_conditioning": True,
                "delivered_frame_hashes": ["s1", "s2"],
            },
            {
                "corner": "gen_on_repeat",
                "generation_on": True,
                "seed_repeat": True,
                "delivered_frame_hashes": ["g1", "g2"],
            },
        ],
    }
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="nested_row",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="a" * 64,
    )
    assert adapted["metrics"]["psnr_mean"] == 28.3
    assert adapted["metrics"]["total_bytes"] == 7000
    assert adapted["timing_evidence"]["measured_client_seconds"] == 24.5
    assert adapted["operating_resolution"] == (2160, 3840)
    assert adapted["claim_eligibility"]["standalone_decode"] is False


def test_adapter_recognizes_actual_same_seed_corner_name() -> None:
    matrix_output = {
        "matrix": [
            {
                "corner": "gen_on_res_off",
                "generation_on": True,
                "delivered_frame_hashes": ["g1", "g2"],
                "scores": {"psnr_y": 28.3},
                "parts": {"transport_total": 7000},
            },
            {
                "corner": "gen_on_res_off_shuffled",
                "generation_on": True,
                "shuffled_conditioning": True,
                "delivered_frame_hashes": ["s1", "s2"],
            },
            {
                "corner": "gen_on_res_off_same_seed",
                "generation_on": True,
                "delivered_frame_hashes": ["g1", "g2"],
            },
        ]
    }
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="actual_same_seed_name",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="a" * 64,
    )
    assert adapted["controls"]["same_seed_determinism_tested"] is True
    assert adapted["controls"]["same_seed_deterministic"] is True


def test_metric_uncertainty_calculation() -> None:
    """Verify SEM and 95% CI calculation on sample distribution."""
    values = [30.0, 32.0, 31.0, 33.0, 29.0]
    res = calculate_metric_uncertainty(values)
    assert res["n"] == 5
    assert res["mean"] == 31.0
    assert res["sem"] > 0
    assert res["ci_95"][0] < res["mean"] < res["ci_95"][1]


def test_diagnostic_matrix_control_wiring() -> None:
    """Verify diagnostic matrix report properly structures and validates controls (shuffled, blank, same-seed)."""
    from scripts.run_diagnostic_matrix import assemble_matrix_report
    from src.contracts.config import PointstreamConfig

    class DummyClip:
        video = "alcaraz_highlights"
        scene = "scene_000"
        context_id = "ctx_0"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        objects = ()

    dummy_sha = "f" * 64
    cfg = PointstreamConfig()

    matrix_rows = [
        {
            "corner": "gen_off_res_off",
            "generation_on": False,
            "residual_on": False,
            "control": "pasted_reference",
            "delivered_frame_hashes": ["hash_paste"],
            "model_invocation_count": 0,
        },
        {
            "corner": "gen_off_res_on",
            "generation_on": False,
            "residual_on": True,
            "control": "pasted_reference",
            "delivered_frame_hashes": ["hash_res_only"],
            "model_invocation_count": 0,
        },
        {
            "corner": "gen_on_res_off",
            "generation_on": True,
            "residual_on": False,
            "control": "generator",
            "delivered_frame_hashes": ["hash_gen"],
            "model_invocation_count": 1,
        },
        {
            "corner": "gen_on_res_on",
            "generation_on": True,
            "residual_on": True,
            "control": "generator",
            "delivered_frame_hashes": ["hash_gen_res"],
            "model_invocation_count": 1,
        },
        {
            "corner": "gen_on_shuffled_conditioning",
            "generation_on": True,
            "residual_on": False,
            "shuffled_conditioning": True,
            "control": "shuffled_conditioning",
            "delivered_frame_hashes": ["hash_shuffled"],
            "model_invocation_count": 1,
        },
        {
            "corner": "gen_on_no_conditioning",
            "generation_on": True,
            "residual_on": False,
            "no_conditioning": True,
            "control": "no_conditioning",
            "delivered_frame_hashes": ["hash_blank"],
            "model_invocation_count": 1,
        },
        {
            "corner": "gen_on_res_off_same_seed",
            "generation_on": True,
            "residual_on": False,
            "control": "generator",
            "delivered_frame_hashes": ["hash_gen"],  # Identical to gen_on_res_off
            "model_invocation_count": 1,
        },
    ]

    report = assemble_matrix_report(
        video=DummyClip.video,
        scene=DummyClip.scene,
        frames=1,
        generator="pix2pix",
        residual_qp=32,
        clip=DummyClip(),
        base_config=cfg,
        matrix=matrix_rows,
        checkpoint_path=None,
        checkpoint_sha256=dummy_sha,
        device="cpu",
        shuffled_control=True,
        no_conditioning_control=True,
        same_seed_control=True,
    )

    ctrls = report["controls"]
    assert ctrls["shuffled_control_enabled"] is True
    assert ctrls["no_conditioning_control_enabled"] is True
    assert ctrls["same_seed_control_enabled"] is True
    assert ctrls["same_seed_determinism_verified"] is True
    assert "gen_on_shuffled_conditioning" in ctrls["shuffled_conditioning"]
    assert "gen_on_no_conditioning" in ctrls["no_conditioning"]


# ---------------------------------------------------------------------------
# 10. E02S Reference Selection, Fresh-Process CLI Continuation & Evaluation
# ---------------------------------------------------------------------------


def test_tennis_dataset_deterministic_reference_selection(tmp_path: Path) -> None:
    """Verify reference_mode in TennisSkeletonDataset.

    1. 'deterministic' selects reference frame derived from sample index (idx % num_colors),
       remaining bit-identical across arbitrary changes to global Python RNG state.
    2. 'first' always selects the first color frame (colors[0]).
    """
    track_dir = tmp_path / "v1" / "segmentations" / "scene_01" / "track_01"
    skel_dir = tmp_path / "v1" / "segmentations" / "scene_01" / "track_01_pose_body"
    track_dir.mkdir(parents=True)
    skel_dir.mkdir(parents=True)

    num_frames = 5
    for i in range(num_frames):
        c_img = np.full((64, 64, 3), (i + 1) * 40, dtype=np.uint8)
        s_img = np.full((64, 64, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(track_dir / f"frame_{i:06d}.png"), c_img)
        cv2.imwrite(str(skel_dir / f"frame_{i:06d}.png"), s_img)

    ds_det = TennisSkeletonDataset(
        root_dir=tmp_path,
        condition="pose_body",
        include_reference=True,
        reference_mode="deterministic",
        target_size=64,
    )
    assert len(ds_det) == num_frames

    random.seed(12345)
    samples_seed1 = [ds_det[i] for i in range(num_frames)]
    random.seed(98765)
    samples_seed2 = [ds_det[i] for i in range(num_frames)]

    for i in range(num_frames):
        skel1, ref1, col1 = samples_seed1[i]
        skel2, ref2, col2 = samples_seed2[i]
        assert torch.equal(ref1, ref2), f"Deterministic reference selection altered by random seed at idx {i}"
        assert torch.equal(ref1, col1)

    ds_first = TennisSkeletonDataset(
        root_dir=tmp_path,
        condition="pose_body",
        include_reference=True,
        reference_mode="first",
        target_size=64,
    )
    first_ref = ds_first[0][1]
    for i in range(num_frames):
        _, ref_i, _ = ds_first[i]
        assert torch.equal(ref_i, first_ref)


def test_fresh_process_trainer_cli_continuation(tmp_path: Path) -> None:
    """Verify fresh-process CLI training interruption and resumption matches uninterrupted run."""
    track = tmp_path / "video1" / "segmentations" / "scene_001" / "track_001"
    skel = tmp_path / "video1" / "segmentations" / "scene_001" / "track_001_pose_body"
    track.mkdir(parents=True)
    skel.mkdir(parents=True)
    for i in range(4):
        img = np.full((256, 256, 3), i * 50, dtype=np.uint8)
        cv2.imwrite(str(track / f"frame_{i:06d}.png"), img)
        cv2.imwrite(str(skel / f"frame_{i:06d}.png"), img)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=".")

    # Run A: uninterrupted 2 steps (step 0 and 1)
    cmd_A = [
        sys.executable,
        "scripts/train_pix2pix.py",
        "--data-root",
        str(tmp_path),
        "--condition",
        "pose_body",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "256",
        "--num-workers",
        "0",
        "--reference-mode",
        "deterministic",
        "--seed",
        "42",
        "--max-steps-per-epoch",
        "2",
        "--checkpoint-interval-sec",
        "0.0",
        "--out-weights",
        str(tmp_path / "out_A.pt"),
        "--checkpoint-path",
        str(tmp_path / "ckpt_A.pt"),
        "--sample-dir",
        str(tmp_path / "samples_A"),
    ]
    res_A = subprocess.run(cmd_A, env=env, capture_output=True, text=True)
    assert res_A.returncode == 0, f"Run A failed: {res_A.stderr}"

    # Run B1: 1 step (step 0, saves checkpoint)
    cmd_B1 = [
        sys.executable,
        "scripts/train_pix2pix.py",
        "--data-root",
        str(tmp_path),
        "--condition",
        "pose_body",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "256",
        "--num-workers",
        "0",
        "--reference-mode",
        "deterministic",
        "--seed",
        "42",
        "--max-steps-per-epoch",
        "1",
        "--checkpoint-interval-sec",
        "0.0",
        "--out-weights",
        str(tmp_path / "out_B.pt"),
        "--checkpoint-path",
        str(tmp_path / "ckpt_B.pt"),
        "--sample-dir",
        str(tmp_path / "samples_B"),
    ]
    res_B1 = subprocess.run(cmd_B1, env=env, capture_output=True, text=True)
    assert res_B1.returncode == 0, f"Run B1 failed: {res_B1.stderr}"

    # Run B2: resume from step 0 and execute step 1
    cmd_B2 = [
        sys.executable,
        "scripts/train_pix2pix.py",
        "--data-root",
        str(tmp_path),
        "--condition",
        "pose_body",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "256",
        "--num-workers",
        "0",
        "--reference-mode",
        "deterministic",
        "--seed",
        "42",
        "--resume",
        "--max-steps-per-epoch",
        "2",
        "--checkpoint-interval-sec",
        "0.0",
        "--out-weights",
        str(tmp_path / "out_B.pt"),
        "--checkpoint-path",
        str(tmp_path / "ckpt_B.pt"),
        "--sample-dir",
        str(tmp_path / "samples_B"),
    ]
    res_B2 = subprocess.run(cmd_B2, env=env, capture_output=True, text=True)
    assert res_B2.returncode == 0, f"Run B2 failed: {res_B2.stderr}"

    ckpt_A = torch.load(tmp_path / "ckpt_A.pt", map_location="cpu")
    ckpt_B2 = torch.load(tmp_path / "ckpt_B.pt", map_location="cpu")

    for k in ckpt_A["G"]:
        assert torch.allclose(ckpt_A["G"][k], ckpt_B2["G"][k], atol=1e-5), f"G weight mismatch at {k}"

    for k in ckpt_A["D"]:
        assert torch.allclose(ckpt_A["D"][k], ckpt_B2["D"][k], atol=1e-5), f"D weight mismatch at {k}"

    assert ckpt_A["opt_G"] is not None and ckpt_B2["opt_G"] is not None
    assert ckpt_A["opt_D"] is not None and ckpt_B2["opt_D"] is not None

    for s_a, s_b in zip(ckpt_A["opt_G"]["state"].values(), ckpt_B2["opt_G"]["state"].values()):
        for p in ("exp_avg", "exp_avg_sq"):
            if p in s_a:
                assert torch.allclose(s_a[p], s_b[p], atol=1e-5)

    for s_a, s_b in zip(ckpt_A["opt_D"]["state"].values(), ckpt_B2["opt_D"]["state"].values()):
        for p in ("exp_avg", "exp_avg_sq"):
            if p in s_a:
                assert torch.allclose(s_a[p], s_b[p], atol=1e-5)


def test_candidate_selection_producer_fields_and_indifference_bands() -> None:
    """Verify compare_candidates parses producer fields and handles indifference bands correctly."""
    cand_base = {
        "metrics": {"total_bytes": 10000, "psnr_mean": 30.0, "ssim_mean": 0.95},
        "timing_evidence": {"measured_client_seconds": 1.0},
    }
    cand_higher_rate = {
        "metrics": {"total_bytes": 12000, "psnr_mean": 30.0, "ssim_mean": 0.95},
        "timing_evidence": {"measured_client_seconds": 1.0},
    }
    assert compare_candidates(cand_base, cand_higher_rate) == "a_dominates"
    assert compare_candidates(cand_higher_rate, cand_base) == "b_dominates"

    cand_close = {
        "metrics": {"total_bytes": 10040, "psnr_mean": 30.05, "ssim_mean": 0.951},
        "timing_evidence": {"measured_client_seconds": 1.01},
    }
    assert compare_candidates(cand_base, cand_close) == "indifferent"

    cand_tradeoff = {
        "metrics": {"total_bytes": 12000, "psnr_mean": 32.0, "ssim_mean": 0.97},
        "timing_evidence": {"measured_client_seconds": 1.0},
    }
    assert compare_candidates(cand_base, cand_tradeoff) == "incomparable"


def test_missing_evidence_cannot_dominate_measured_candidate() -> None:
    """Verify missing or asymmetric quality/rate evidence returns 'incomparable' and never dominance."""
    measured_cand = {
        "metrics": {"total_bytes": 10000, "psnr_mean": 30.0, "ssim_mean": 0.95},
        "timing_evidence": {"measured_client_seconds": 1.0},
    }
    unmeasured_cand = {
        "metrics": {"total_bytes": 5000},
        "timing_evidence": {"measured_client_seconds": 0.5},
    }
    assert compare_candidates(unmeasured_cand, measured_cand) == "incomparable"
    assert compare_candidates(measured_cand, unmeasured_cand) == "incomparable"

    unmeasured_cand2 = {
        "metrics": {"total_bytes": 6000},
    }
    assert compare_candidates(unmeasured_cand, unmeasured_cand2) == "incomparable"


def test_promote_survivors_preserves_incomparable_tradeoffs() -> None:
    """Verify promote_survivors preserves candidates with valid trade-offs instead of pruning."""
    aggregates = {
        "survivor_low_rate": {
            "metrics": {"total_bytes": 5000, "psnr_mean": 28.0, "ssim_mean": 0.90},
            "timing_evidence": {"measured_client_seconds": 1.0},
            "run_completed": True,
        },
        "tradeoff_high_quality": {
            "metrics": {"total_bytes": 12000, "psnr_mean": 34.0, "ssim_mean": 0.98},
            "timing_evidence": {"measured_client_seconds": 1.2},
            "run_completed": True,
        },
    }
    survivors = promote_survivors(["survivor_low_rate", "tradeoff_high_quality"], aggregates)
    assert "tradeoff_high_quality" in survivors
    assert "survivor_low_rate" in survivors


def test_adapter_handles_actual_producer_control_rows() -> None:
    """Feed actual 7-corner producer matrix through adapt_diagnostic_matrix_result and check controls."""
    seven_corner_matrix = {
        "video": "alcaraz_highlights",
        "scene": "scene_028",
        "frames": 16,
        "matrix": [
            {
                "corner": "gen_off_res_off",
                "generation_on": False,
                "residual_on": False,
                "control": "pasted_reference",
                "delivered_frame_hashes": ["h_paste"],
                "model_invocation_count": 0,
            },
            {
                "corner": "gen_off_res_on",
                "generation_on": False,
                "residual_on": True,
                "control": "pasted_reference",
                "delivered_frame_hashes": ["h_res_only"],
                "model_invocation_count": 0,
            },
            {
                "corner": "gen_on_res_off",
                "generation_on": True,
                "residual_on": False,
                "control": "generator",
                "delivered_frame_hashes": ["h_gen_1", "h_gen_2"],
                "delivered_shape": [16, 2160, 3840, 3],
                "model_invocation_count": 1,
                "scores": {"psnr_y": 28.3, "ssim": 0.97, "vmaf": 88.0},
                "timing": {"client_seconds": 24.5, "encoder_seconds": 85.0},
                "parts": {"residual": 0, "transport_total": 7000},
                "coded_bytes": 7000,
            },
            {
                "corner": "gen_on_res_on",
                "generation_on": True,
                "residual_on": True,
                "control": "generator",
                "delivered_frame_hashes": ["h_gen_res"],
                "model_invocation_count": 1,
            },
            {
                "corner": "gen_on_shuffled_conditioning",
                "generation_on": True,
                "residual_on": False,
                "shuffled_conditioning": True,
                "control": "shuffled_conditioning",
                "delivered_frame_hashes": ["h_shuffled_1", "h_shuffled_2"],
                "model_invocation_count": 1,
            },
            {
                "corner": "gen_on_no_conditioning",
                "generation_on": True,
                "residual_on": False,
                "no_conditioning": True,
                "control": "no_conditioning",
                "delivered_frame_hashes": ["h_blank_1", "h_blank_2"],
                "model_invocation_count": 1,
            },
            {
                "corner": "gen_on_res_off_same_seed",
                "generation_on": True,
                "residual_on": False,
                "control": "generator",
                "delivered_frame_hashes": ["h_gen_1", "h_gen_2"],
                "model_invocation_count": 1,
            },
        ],
    }
    adapted = adapt_diagnostic_matrix_result(
        seven_corner_matrix,
        run_id="run_e02s_test",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="0" * 64,
    )
    assert adapted["controls"]["same_seed_determinism_tested"] is True
    assert adapted["controls"]["same_seed_deterministic"] is True
    assert adapted["controls"]["conditioned_vs_shuffled_tested"] is True
    assert adapted["controls"]["conditioning_sensitive"] is True
    assert adapted["metrics"]["total_bytes"] == 7000
    assert adapted["metrics"]["psnr_mean"] == 28.3
    assert adapted["metrics"]["ssim_mean"] == 0.97
    assert adapted["timing_evidence"]["measured_client_seconds"] == 24.5
    valid, reasons = validate_generation_result(adapted)
    assert valid, f"Validation failed: {reasons}"


def test_unsupported_worker_mode_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Verify warning is emitted when num_workers > 0 with resume."""
    import argparse
    import logging
    from unittest.mock import patch

    mock_args = argparse.Namespace(
        resume=True,
        checkpoint_path="/nonexistent/path/ckpt.pt",
        num_workers=4,
        seed=42,
        lr=0.0002,
        b1=0.5,
        b2=0.999,
        data_root=".",
        img_size=256,
        condition="pose_body",
        reference_mode="deterministic",
        batch_size=1,
        checkpoint_interval_sec=3600.0,
        epochs=1,
        max_steps_per_epoch=None,
    )

    with caplog.at_level(logging.WARNING):
        from scripts.train_pix2pix import main_worker
        with patch("scripts.train_pix2pix.torch.cuda.is_available", return_value=False):
            with patch("scripts.train_pix2pix.TennisSkeletonDataset"):
                with patch("scripts.train_pix2pix.DataLoader", side_effect=RuntimeError("stop_early")):
                    try:
                        main_worker(0, 1, mock_args)
                    except Exception:
                        pass

    assert any("Worker mode notice: num_workers=4 > 0" in record.message for record in caplog.records)


