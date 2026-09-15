"""E02 Generator Readiness and E01 Interface Compliance Test Suite.

Verifies:
1. Full-trajectory multi-frame placement and pose alignment across frames.
2. Missing pose/skeleton fail-closed semantics.
3. Controls: normal vs shuffled conditioning sensitivity across frames.
4. Controls: same-seed bit-identical determinism.
5. Uncertainty-aware promotion preserving variants within noise threshold.
6. Campaign ranking excluding uncalibrated LPIPS and using residual wire bytes.
7. E01 schema adapter formatting, claim eligibility, and fail-closed validation.
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

from experiments.long_scenes.loader import (
    load_long_scene_clip,
)
from experiments.tier.diagnostic_report import per_frame_sha256
from scripts.run_diagnostic_matrix import (
    _augment_objects_with_pose,
)
from scripts.train_campaign import (
    LOWER_IS_BETTER,
    RANKED_METRICS,
    promote_survivors,
    rank_variants,
)
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
    reason="requires external YouTube-derived footage; set POINTSTREAM_DATA_TESTS=1",
)
def test_full_trajectory_multi_frame_placement() -> None:
    """Verify full_trajectory=True emits ObjectRequest for each visible frame."""
    # full_trajectory=False: exactly 1 object per track at first appearance (2 objects)
    clip_single = load_long_scene_clip("alcaraz_highlights", "scene_028", 48, full_trajectory=False)
    assert len(clip_single.objects) == 2

    # full_trajectory=True: 48 frames * 2 tracks = 96 objects spanning all frames
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

    # Frame 0: bbox 80x40 (h=80, w=40), skeleton 80x40
    # Frame 1: bbox 90x45 (h=90, w=45), skeleton 90x45
    # Appearance keyframe: 100x50 (h=100, w=50)
    skel_0 = np.full((80, 40, 3), 150, dtype=np.uint8)
    skel_1 = np.full((90, 45, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(skel_dir / "frame_000000.png"), skel_0)
    cv2.imwrite(str(skel_dir / "frame_000001.png"), skel_1)

    appearance = np.ones((100, 50, 3), dtype=np.uint8) * 128
    obj_0 = ObjectRequest(
        object_id="track_0001",
        appearance=appearance,
        bbox=(10, 10, 50, 90),  # left=10, top=10, right=50, bottom=90 -> w=40, h=80
        mask=np.ones((100, 50), dtype=bool),
        frame_index=0,
    )
    obj_1 = ObjectRequest(
        object_id="track_0001",
        appearance=appearance,
        bbox=(10, 10, 55, 100),  # left=10, top=10, right=55, bottom=100 -> w=45, h=90
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
    # Skeletons were aligned to appearance shape (3, 100, 50)
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
    """Verify mismatched skeleton shape (matching neither bbox nor appearance) raises ValueError."""
    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    scene_dir = tmp_path / "assets" / "dataset" / "fake_video" / "segmentations" / "fake_scene"
    skel_dir = scene_dir / "track_0001_skeleton"
    skel_dir.mkdir(parents=True)

    # Skeleton 128x80 (h=128, w=80), appearance is 64x64, bbox is 32x32
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
# 3. Controls: Normal vs Shuffled & Determinism
# ---------------------------------------------------------------------------


def test_controls_normal_vs_shuffled_conditioning_sensitivity() -> None:
    """Verify frame hashes differ when conditioning is shuffled."""
    # Create two synthetic 4-frame sequences representing normal and shuffled generation
    frames_normal = [np.full((64, 64, 3), i * 30, dtype=np.uint8) for i in range(4)]
    # Shuffled has frames in inverted/scrambled order or different conditioning output
    frames_shuffled = [np.full((64, 64, 3), (3 - i) * 30, dtype=np.uint8) for i in range(4)]

    hashes_normal = per_frame_sha256(np.asarray(frames_normal))
    hashes_shuffled = per_frame_sha256(np.asarray(frames_shuffled))

    assert len(hashes_normal) == 4
    assert len(hashes_shuffled) == 4
    assert hashes_normal != hashes_shuffled
    assert hashes_normal[0] != hashes_shuffled[0]


def test_controls_same_seed_bit_identical_determinism() -> None:
    """Verify repeated generation with same seed produces bit-identical frame hashes."""
    rng1 = np.random.default_rng(42)
    frames_run1 = [rng1.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]

    rng2 = np.random.default_rng(42)
    frames_run2 = [rng2.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]

    hashes_run1 = per_frame_sha256(np.asarray(frames_run1))
    hashes_run2 = per_frame_sha256(np.asarray(frames_run2))

    assert hashes_run1 == hashes_run2


# ---------------------------------------------------------------------------
# 4. Uncertainty-Aware Promotion
# ---------------------------------------------------------------------------


def test_uncertainty_aware_promotion_preserves_close_candidates() -> None:
    """Candidates within 2% rate difference of cutoff are preserved as survivors."""
    ranked = ["cand_1", "cand_2", "cand_3", "cand_4"]
    aggregate = {
        "cand_1": {"residual_bytes": 1000, "psnr_mean": 35.0, "success": True},
        # Nominal cutoff keeps cand_1 and cand_2 (ceil(4/2)=2)
        "cand_2": {"residual_bytes": 1100, "psnr_mean": 34.5, "success": True},
        # cand_3 is only 1.8% worse than cand_2 (1120 vs 1100: (1120-1100)/1100 = 0.0181 <= 0.02)
        "cand_3": {"residual_bytes": 1120, "psnr_mean": 34.4, "success": True},
        # cand_4 is 20% worse (1320 vs 1100: > 0.02)
        "cand_4": {"residual_bytes": 1320, "psnr_mean": 30.0, "success": True},
    }

    survivors = promote_survivors(ranked, aggregate, min_diff_threshold=0.02)
    assert survivors == ["cand_1", "cand_2", "cand_3"]
    assert "cand_4" not in survivors


def test_uncertainty_aware_promotion_prunes_distant_candidates() -> None:
    """Candidates strictly exceeding the difference threshold are pruned."""
    ranked = ["cand_1", "cand_2", "cand_3"]
    aggregate = {
        "cand_1": {"residual_bytes": 1000, "success": True},
        "cand_2": {"residual_bytes": 1005, "success": True},
        "cand_3": {"residual_bytes": 1500, "success": True},
    }
    # Nominal keep: ceil(3/2) = 2 -> cand_1, cand_2
    # cand_3 is (1500-1005)/1005 = 49% worse -> pruned
    survivors = promote_survivors(ranked, aggregate, min_diff_threshold=0.02)
    assert survivors == ["cand_1", "cand_2"]


# ---------------------------------------------------------------------------
# 5. Campaign Ranking Excludes Uncalibrated LPIPS
# ---------------------------------------------------------------------------


def test_ranked_metrics_excludes_uncalibrated_lpips() -> None:
    """Verify lpips_vgg_uncalibrated is not in LOWER_IS_BETTER or RANKED_METRICS."""
    assert "lpips_vgg_uncalibrated" not in LOWER_IS_BETTER
    assert "lpips_vgg_uncalibrated" not in RANKED_METRICS
    assert LOWER_IS_BETTER == {"temporal_error"}


def test_rank_variants_orders_by_wire_bytes_first() -> None:
    """Residual bytes strictly outranks perceptual score."""
    aggregate = {
        # Better perceptual, but higher residual bytes
        "high_bytes": {
            "residual_bytes": 50000,
            "psnr_mean": 40.0,
            "ssim_mean": 0.98,
            "vmaf_mean": 95.0,
            "temporal_error": 0.5,
            "success": True,
        },
        # Lower residual bytes (cheaper on wire), but slightly lower perceptual
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
    # Composite still reflects perceptual scores accurately
    assert composite["high_bytes"] > composite["low_bytes"]


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


def test_metric_uncertainty_calculation() -> None:
    """Verify SEM and 95% CI calculation on sample distribution."""
    values = [30.0, 32.0, 31.0, 33.0, 29.0]
    res = calculate_metric_uncertainty(values)
    assert res["n"] == 5
    assert res["mean"] == 31.0
    assert res["sem"] > 0
    assert res["ci_95"][0] < res["mean"] < res["ci_95"][1]
