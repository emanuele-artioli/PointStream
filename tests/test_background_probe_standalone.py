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
    bs_dir = Path("/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e04a/run-20260916-federer007/bitstreams")
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

