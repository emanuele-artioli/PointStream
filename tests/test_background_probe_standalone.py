"""Focused unit tests for standalone background decode and empty-mask scorer behavior."""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
from pathlib import Path
import struct

import numpy as np
import pytest

from scripts.background_probe import (
    decode_standalone_representation,
    masked_luma_psnr,
    safe_masked_ssim,
    unpack_panorama_side_data,
    unpack_still_or_video_side_data,
)
from src.components.metrics.ssim import masked_ssim


def test_safe_masked_ssim_empty_mask() -> None:
    """Empty mask should return float('nan') cleanly without throwing RuntimeWarning."""
    ref = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    pred = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    empty_mask = np.zeros((2, 32, 32), dtype=bool)

    score = safe_masked_ssim(ref, pred, empty_mask)
    assert np.isnan(score)

    psnr_score = masked_luma_psnr(ref, pred, empty_mask)
    assert np.isnan(psnr_score)


def test_safe_masked_ssim_parity_with_masked_ssim() -> None:
    """When mask is non-empty, safe_masked_ssim must match masked_ssim bit-for-bit."""
    rng = np.random.default_rng(42)
    ref = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    pred = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    mask = rng.choice([True, False], size=(2, 32, 32), p=[0.3, 0.7])

    score_safe = safe_masked_ssim(ref, pred, mask)
    score_orig = masked_ssim(ref, pred, mask)

    assert not np.isnan(score_safe)
    np.testing.assert_allclose(score_safe, score_orig, rtol=1e-6, atol=1e-6)


def test_unpack_still_or_video_side_data_corrupt() -> None:
    """Side data of invalid length must raise ValueError."""
    with pytest.raises(ValueError, match="Payload length mismatch"):
        unpack_still_or_video_side_data(b"too_short")


def test_unpack_panorama_side_data_corrupt() -> None:
    """Panorama side data with incorrect length must raise ValueError."""
    with pytest.raises(ValueError, match="Payload too short for panorama header"):
        unpack_panorama_side_data(b"short")

    # Header ok, but truncated homography payload
    header = struct.pack(">HHHHHf", 360, 640, 360, 640, 2, 12.0)
    with pytest.raises(ValueError, match="Payload length mismatch"):
        unpack_panorama_side_data(header + b"\x00" * 10)


def test_decode_standalone_missing_files(tmp_path: Path) -> None:
    """Missing bitstream or side data must raise FileNotFoundError."""
    bs = tmp_path / "missing.vvc"
    sd = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        decode_standalone_representation(bs, sd)


def test_decode_standalone_saved_e04a_artifacts() -> None:
    """Verify standalone decode on actual saved E04A bitstreams and side data."""
    bs_dir = Path(
        "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e04a/run-20260916-federer007/bitstreams"
    )
    if not bs_dir.is_dir():
        pytest.skip("Saved E04A bitstream directory not present on host")

    for bs_file in sorted(bs_dir.glob("*.vvc")):
        sd_file = bs_file.with_name(bs_file.stem + "_side.bin")
        assert sd_file.is_file(), f"Missing side data for {bs_file.name}"
        rendered, meta = decode_standalone_representation(bs_file, sd_file)
        assert rendered.shape == (48, 360, 640, 3)
        assert rendered.dtype == np.uint8
        assert meta["rejected"] is False
        assert meta["total_client_seconds"] > 0


def test_scorer_calibration_whole_frame_ordering_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When whole-frame ordering fails, run_scorer_calibration must log alarms and mark valid=False."""
    from scripts.e04a_evidence_completion import run_scorer_calibration
    from src.components.metrics.ssim import SsimMetric

    rng = np.random.default_rng(101)
    frames = rng.integers(30, 220, size=(2, 360, 640, 3), dtype=np.uint8)
    masks = np.zeros((2, 360, 640), dtype=bool)
    masks[:, 50:150, 50:150] = True
    boundary = np.zeros_like(masks)

    # Invert score behavior so severe blur gets higher score than mild blur
    orig_score = SsimMetric.score

    def inverted_score(self: SsimMetric, ref: np.ndarray, pred: np.ndarray) -> float:
        val = orig_score(self, ref, pred)
        # Flip severe blur to 0.99 and mild blur to 0.10 to force ordering failure
        return 0.10 if val > 0.50 else 0.99

    monkeypatch.setattr(SsimMetric, "score", inverted_score)
    res = run_scorer_calibration(frames, masks, boundary)
    assert res["valid"] is False
    assert any("Whole-frame windowed ordering violated" in a for a in res["alarms"])


def test_scorer_calibration_identity_and_null_controls() -> None:
    """Calibration must verify identity unit SSIM and empty-mask safe NaN behavior."""
    from scripts.e04a_evidence_completion import run_scorer_calibration

    rng = np.random.default_rng(202)
    frames = rng.integers(20, 240, size=(2, 360, 640, 3), dtype=np.uint8)
    masks = np.zeros((2, 360, 640), dtype=bool)
    masks[:, 100:200, 100:200] = True
    boundary = np.zeros_like(masks)

    res = run_scorer_calibration(frames, masks, boundary)
    assert res["identity_checks"]["visible"]["passed"] is True
    assert res["identity_checks"]["full_frame"]["passed"] is True
    assert res["empty_mask_behavior"]["status"] == "verified_safe_nan_return"
    assert res["null_controls"]["empty_mask_nan_held"] is True


def test_runner_refuses_existing_nonempty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner must refuse to write to an existing non-empty directory."""
    from scripts.e04a_evidence_completion import main

    nonempty_dir = tmp_path / "existing_run"
    nonempty_dir.mkdir(parents=True, exist_ok=True)
    (nonempty_dir / "old_artifact.txt").write_text("prior output")

    monkeypatch.setattr("sys.argv", ["runner", "--output-dir", str(nonempty_dir)])
    with pytest.raises(FileExistsError, match="Refusing to write to existing non-empty directory"):
        main()


def test_runner_aborts_on_calibration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner must abort immediately with RuntimeError if scorer calibration fails."""
    from scripts import e04a_evidence_completion

    run_dir = tmp_path / "fresh_run"

    # Mock inputs so test does not require full dataset
    dummy_frames = np.zeros((2, 360, 640, 3), dtype=np.uint8)
    dummy_masks = np.zeros((2, 360, 640), dtype=bool)
    dummy_meta = {"common_preparation_seconds": 0.1}

    monkeypatch.setattr(
        e04a_evidence_completion,
        "load_360p_input_data",
        lambda: (dummy_frames, dummy_masks, dummy_masks, [], dummy_meta),
    )
    monkeypatch.setattr(
        e04a_evidence_completion,
        "run_scorer_calibration",
        lambda f, m, b: {"valid": False, "alarms": ["Forced synthetic calibration failure"]},
    )
    monkeypatch.setattr("sys.argv", ["runner", "--output-dir", str(run_dir)])

    with pytest.raises(RuntimeError, match="Scorer calibration failed with 1 alarms"):
        e04a_evidence_completion.main()
