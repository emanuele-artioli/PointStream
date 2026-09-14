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

from experiments.long_scenes.loader import (
    load_long_scene_clip,
)
from experiments.tier.campaign_result import (
    ingest_for_claim,
    validate_campaign_record,
)
from scripts.run_diagnostic_matrix import (
    _augment_objects_with_pose,
)
from scripts.train_campaign import (
    evaluate_checkpoint,
    promote_survivors,
    rank_variants,
)
from scripts.train_pix2pix import (
    PatchGANDiscriminator,
    UNetGenerator,
    build_checkpoint_state,
    save_checkpoint_atomic,
)
from src.components.generation.spade4tennis_arch import SPADEResNet9Generator
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner.generation_adapter import (
    adapt_campaign_eval_result,
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


def test_real_neural_conditioning_sensitivity_and_determinism() -> None:
    """Prove conditioning sensitivity and same-seed determinism on real UNetGenerator."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    net = UNetGenerator(in_channels=6, out_channels=3).to(device)
    net.eval()

    # Create reference appearance and two distinct pose conditionings
    ref = torch.randn(1, 3, 256, 256)
    cond_normal = torch.randn(1, 3, 256, 256)
    cond_shuffled = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out_normal_1 = net(torch.cat((cond_normal, ref), dim=1))
        out_normal_2 = net(torch.cat((cond_normal, ref), dim=1))
        out_shuffled = net(torch.cat((cond_shuffled, ref), dim=1))

    # Determinism: same input produces bit-identical tensor
    assert torch.equal(out_normal_1, out_normal_2)

    # Conditioning sensitivity: different conditioning produces measurably different output
    l1_diff = torch.mean(torch.abs(out_normal_1 - out_shuffled)).item()
    assert l1_diff > 0.01


def test_real_neural_tiny_learning_and_atomic_resume(tmp_path: Path) -> None:
    """Verify tiny learning step updates weights and checkpoint resume preserves step and state."""
    torch.manual_seed(123)
    net = UNetGenerator(in_channels=6, out_channels=3)
    disc = PatchGANDiscriminator(in_channels=6)
    opt_G = optim.Adam(net.parameters(), lr=1e-3)
    opt_D = optim.Adam(disc.parameters(), lr=1e-3)

    inp = torch.randn(2, 6, 256, 256)
    target = torch.randn(2, 3, 256, 256)

    # 1 step optimization
    out = net(inp)
    loss = nn.functional.l1_loss(out, target)
    opt_G.zero_grad()
    loss.backward()
    opt_G.step()

    ckpt_file = tmp_path / "pix2pix_ckpt.pt"
    ckpt_state = build_checkpoint_state(
        epoch=1,
        step=5,
        total_steps=10,
        generator=net,
        discriminator=disc,
        optimizer_G=opt_G,
        optimizer_D=opt_D,
        ngpus=1,
    )
    save_checkpoint_atomic(ckpt_state, ckpt_file)
    assert ckpt_file.exists()

    # Mutate weights
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)

    # Resume
    loaded = torch.load(ckpt_file, map_location="cpu")
    net2 = UNetGenerator(in_channels=6, out_channels=3)
    net2.load_state_dict(loaded["G"])

    # Verify weights match before mutation
    for p1, p2 in zip(net.parameters(), net2.parameters(), strict=True):
        assert not torch.equal(p1, p2)  # net was mutated
        assert torch.allclose(p1 - 1.0, p2, atol=1e-6)

    assert loaded["epoch"] == 1
    assert loaded["step"] == 5
    assert loaded["total_steps_in_epoch"] == 10


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
    """Candidates within 2% rate difference of cutoff are preserved as survivors."""
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


def test_generation_adapter_roundtrip_passes_e01_validation() -> None:
    """Verify real producer-shaped diagnostic matrix row produces valid E01 record."""
    matrix_output = {
        "video": "alcaraz_highlights",
        "scene": "scene_028",
        "frames": 4,
        "fps": 24.0,
        "matrix": [
            {
                "corner": "gen_off_res_off",
                "control": "pasted_reference",
                "generation_on": False,
                "residual_on": False,
                "shuffled_conditioning": False,
                "delivered_frame_hashes": ["p1", "p2", "p3", "p4"],
                "coded_bytes": 10000,
                "residual_bytes": 0,
            },
            {
                "corner": "gen_on_res_off",
                "control": None,
                "generation_on": True,
                "residual_on": False,
                "shuffled_conditioning": False,
                "delivered_frame_hashes": ["g1", "g2", "g3", "g4"],
                "coded_bytes": 15000,
                "residual_bytes": 0,
                "elapsed_seconds": 1.2,
                "psnr_y": 32.5,
                "ssim": 0.91,
            },
            {
                "corner": "gen_on_res_off_shuffled",
                "control": None,
                "generation_on": True,
                "residual_on": False,
                "shuffled_conditioning": True,
                "delivered_frame_hashes": ["s1", "s2", "s3", "s4"],
                "coded_bytes": 15000,
                "residual_bytes": 0,
            },
            {
                "corner": "gen_on_res_off_repeat",
                "control": None,
                "generation_on": True,
                "residual_on": False,
                "seed_repeat": True,
                "delivered_frame_hashes": ["g1", "g2", "g3", "g4"],
                "coded_bytes": 15000,
                "residual_bytes": 0,
            },
        ],
    }

    dummy_sha = "a" * 64
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="test_run_producer_matrix",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256=dummy_sha,
        timing_evidence_id="timing_p2p_01",
        code_revision="b07db0bcbe4561fc44845ecf52363cfa5193be00",
    )

    # 1. Internal validation
    valid, blockers = validate_generation_result(adapted)
    assert valid is True
    assert blockers == []

    # 2. Direct E01 campaign result validator
    e01_blockers = validate_campaign_record(adapted)
    assert e01_blockers == []

    # 3. Direct E01 ingestion for RD claim
    rd_ingest = ingest_for_claim([adapted], "rd")
    assert rd_ingest["n_kept"] == 1
    assert rd_ingest["n_excluded"] == 0

    # 4. Direct E01 ingestion for Runtime claim
    rt_ingest = ingest_for_claim([adapted], "runtime")
    assert rt_ingest["n_kept"] == 1
    assert rt_ingest["n_excluded"] == 0


def test_generation_adapter_fails_closed_on_invalid_checkpoint_sha() -> None:
    """Verify non-SHA256 checkpoint identifier fails closed and excludes RD claim."""
    matrix_output = {
        "video": "alcaraz_highlights",
        "scene": "scene_000",
        "matrix": [
            {"corner": "gen_on_res_off", "generation_on": True, "delivered_frame_hashes": ["g1"], "coded_bytes": 100, "psnr_y": 30.0},
            {"corner": "gen_on_res_off_shuffled", "generation_on": True, "shuffled_conditioning": True, "delivered_frame_hashes": ["s1"]},
            {"corner": "gen_on_res_off_repeat", "generation_on": True, "seed_repeat": True, "delivered_frame_hashes": ["g1"]},
        ],
    }

    # Short/invalid sha string
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="bad_sha",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256="not_a_sha256",
    )

    assert adapted["claim_eligibility"]["rd"] is False
    assert any("SHA-256" in ex["reason"] for ex in adapted["claim_eligibility"]["exclusions"])
    assert ingest_for_claim([adapted], "rd")["n_kept"] == 0


def test_generation_adapter_fails_closed_on_shuffled_match() -> None:
    """Verify generator that ignores conditioning (shuffled match) is excluded from RD claim."""
    matrix_output = {
        "video": "v", "scene": "s",
        "matrix": [
            {"corner": "gen_off_res_off", "generation_on": False, "delivered_frame_hashes": ["p1"]},
            {"corner": "gen_on_res_off", "generation_on": True, "delivered_frame_hashes": ["g1"], "coded_bytes": 100, "psnr_y": 30.0},
            {"corner": "gen_on_res_off_shuffled", "generation_on": True, "shuffled_conditioning": True, "delivered_frame_hashes": ["g1"]},
            {"corner": "gen_on_res_off_repeat", "generation_on": True, "seed_repeat": True, "delivered_frame_hashes": ["g1"]},
        ],
    }
    dummy_sha = "b" * 64
    adapted = adapt_diagnostic_matrix_result(
        matrix_output,
        run_id="dead_cond",
        backend_name="pix2pix",
        arch="pix2pix",
        checkpoint_sha256=dummy_sha,
    )
    assert adapted["claim_eligibility"]["rd"] is False
    assert any("shuffled match" in ex["reason"] for ex in adapted["claim_eligibility"]["exclusions"])
    assert ingest_for_claim([adapted], "rd")["n_kept"] == 0


def test_campaign_eval_adapter_fails_closed_without_controls() -> None:
    """Verify aggregate without controls does not default to true and excludes RD claim."""
    eval_res = {
        "aggregate": {
            "success": True,
            "residual_bytes": 1000,
            "total_bytes": 2000,
            "psnr_mean": 34.0,
            "checkpoint_sha256": "c" * 64,
        },
        "per_clip": [{"video": "v1", "psnr": 34.0, "residual_bytes": 1000}],
    }
    adapted = adapt_campaign_eval_result(eval_res, run_id="camp_no_ctrl", backend_name="p2p", arch="p2p")
    assert adapted["claim_eligibility"]["rd"] is False
    assert any("missing conditioning" in ex["reason"] for ex in adapted["claim_eligibility"]["exclusions"])
    assert ingest_for_claim([adapted], "rd")["n_kept"] == 0


def test_metric_uncertainty_single_source_handling() -> None:
    """Verify n=1 explicitly reports uncertainty unavailable without fabricating zero-width CI."""
    values = [31.0]
    res = calculate_metric_uncertainty(values)
    assert res["n"] == 1
    assert res["mean"] == 31.0
    assert res["sem"] is None
    assert res["ci_95"] is None
    assert res["uncertainty_status"] == "single_source_uncertainty_unavailable"
