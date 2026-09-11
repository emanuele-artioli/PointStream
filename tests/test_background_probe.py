"""Focused unit tests for background probe (CODEC-ACT-06).

Verifies:
1. Frame warping geometry and singular homography handling.
2. Common foreground-removed frame stack construction invariants:
   - Visible unmasked pixels (~mask) are strictly preserved.
   - Masked pixels are replaced with temporal composite.
   - Hole filling logic works and leaves no NaNs.
3. Accurate side data accounting across all 3 representations.
4. Uncompressed sanity check behavior and anchors.
5. Pre-registered bounds checking and alarm reporting.
6. End-to-end evaluation pipeline smoke on small inputs.
"""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.background_probe import (
    build_common_cleaned_stack,
    charge_side_data,
    check_bounds,
    evaluate_cleaned_video,
    evaluate_registered_panorama,
    evaluate_still_frame0,
    masked_luma_psnr,
    uncompressed_sanity_check,
    warp_plate_to_frame,
)


def _synthetic_clip(
    t: int = 4,
    h: int = 64,
    w: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic test clip with panning background and a moving mask."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    masks = np.zeros((t, h, w), dtype=bool)

    # Base background texture
    bg_texture = rng.integers(50, 200, size=(h + 20, w + 20, 3), dtype=np.uint8)

    for i in range(t):
        # Background pans slightly by 2 pixels per frame
        shift_x = i * 2
        shift_y = i * 1
        frame_bg = bg_texture[shift_y : shift_y + h, shift_x : shift_x + w].copy()

        # Foreground player object (e.g. bright block)
        px = 20 + i * 3
        py = 20
        pw, ph = 10, 15
        frame_bg[py : py + ph, px : px + pw] = 255  # player color
        masks[i, py : py + ph, px : px + pw] = True
        frames[i] = frame_bg

    return frames, masks


def test_warp_plate_to_frame_identity_and_singular() -> None:
    plate = np.ones((64, 64, 3), dtype=np.uint8) * 120
    identity = np.eye(3, dtype=np.float64)
    warped = warp_plate_to_frame(plate, identity, height=64, width=64)
    np.testing.assert_array_equal(plate, warped)

    singular = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Homography is singular"):
        warp_plate_to_frame(plate, singular, height=64, width=64)


def test_build_common_cleaned_stack_invariants(tmp_path: Path) -> None:
    frames, masks = _synthetic_clip(t=4, h=64, w=64)
    cache_file = tmp_path / "cache_cleaned.npz"

    cleaned, plate, homographies, stats = build_common_cleaned_stack(
        frames, masks, register=True, cache_path=cache_file
    )

    assert cleaned.shape == frames.shape
    assert len(homographies) == len(frames)
    assert not np.isnan(cleaned).any()

    # Invariant: Unmasked background pixels MUST be bit-identical to source frames
    for t in range(len(frames)):
        m = masks[t]
        np.testing.assert_array_equal(
            cleaned[t][~m],
            frames[t][~m],
            err_msg=f"Frame {t}: unmasked visible pixels were altered!",
        )
        # Masked pixels should NOT equal the original player color (255)
        # because they were replaced with the warped composite background
        assert not np.all(cleaned[t][m] == 255)

    # Test loading from cache
    assert cache_file.is_file()
    c_cached, p_cached, h_cached, s_cached = build_common_cleaned_stack(
        frames, masks, register=True, cache_path=cache_file
    )
    assert s_cached["from_cache"] is True
    np.testing.assert_array_equal(cleaned, c_cached)
    np.testing.assert_array_equal(plate, p_cached)


def test_charge_side_data_accounting() -> None:
    # Rep 1: Still frame 0
    s1 = charge_side_data("still_frame0", n_frames=48)
    assert s1["total_side_data_bytes"] == 10
    assert s1["homography_bytes"] == 0

    # Rep 2: Registered panorama
    s2 = charge_side_data("registered_panorama", n_frames=48, plate_shape=(2170, 3880))
    # 14 base bytes + 48 frames * 9 * 4 bytes per float32 = 14 + 1,728 = 1,742
    assert s2["total_side_data_bytes"] == 1742
    assert s2["homography_bytes"] == 1728

    # Rep 3: Video
    s3 = charge_side_data("cleaned_video", n_frames=48)
    assert s3["total_side_data_bytes"] == 10

    with pytest.raises(ValueError, match="Unknown representation"):
        charge_side_data("invalid_rep", n_frames=48)


def test_uncompressed_sanity_check() -> None:
    frames, masks = _synthetic_clip(t=4, h=64, w=64)
    cleaned, plate, homographies, _ = build_common_cleaned_stack(frames, masks, register=True)

    sanity = uncompressed_sanity_check(frames, cleaned, plate, homographies, masks)

    # Rep 3 uncompressed matches source exactly on visible background (~mask)
    assert math.isinf(sanity["cleaned_video"]["psnr_y_visible_dB"])
    assert sanity["cleaned_video"]["ssim_visible"] == 1.0

    # Rep 1 uncompressed has camera motion drift, so PSNR should be finite
    assert math.isfinite(sanity["still_frame0"]["psnr_y_visible_dB"])


def test_masked_luma_psnr_exact() -> None:
    f1 = np.full((2, 32, 32, 3), 100, dtype=np.uint8)
    f2 = f1.copy()
    mask = np.ones((2, 32, 32), dtype=bool)

    # Identical images score inf
    assert math.isinf(masked_luma_psnr(f1, f2, mask))

    # Known offset
    f2[0, 0, 0] = [200, 100, 100]  # single pixel error
    psnr = masked_luma_psnr(f1, f2, mask)
    assert math.isfinite(psnr)
    assert psnr > 40.0


def test_check_bounds_logic() -> None:
    sample_points = [
        {
            "representation": "still_frame0",
            "qp": 47,
            "total_package_bytes": 5000,
            "metrics": {
                "psnr_y_visible_dB": 30.0,
                "ssim_visible": 0.85,
            },
            "timing": {
                "encode_seconds": 2.0,
                "decode_render_seconds": 1.0,
            },
        }
    ]

    passed, alarms = check_bounds(sample_points)
    assert passed is True
    assert len(alarms) == 0

    # Trigger alarm: PSNR too low
    bad_points = [
        {
            "representation": "still_frame0",
            "qp": 47,
            "total_package_bytes": 5000,
            "metrics": {
                "psnr_y_visible_dB": 10.0,  # below 15.0
                "ssim_visible": 0.85,
            },
            "timing": {
                "encode_seconds": 2.0,
                "decode_render_seconds": 1.0,
            },
        }
    ]
    passed_bad, alarms_bad = check_bounds(bad_points)
    assert passed_bad is False
    assert len(alarms_bad) == 1
    assert "psnr_y_visible" in alarms_bad[0]


def test_evaluate_representations_smoke(tmp_path: Path) -> None:
    """Smoke test running small encode/decode for the 3 representations."""
    frames, masks = _synthetic_clip(t=4, h=64, w=64)
    cleaned, plate, homographies, _ = build_common_cleaned_stack(frames, masks, register=True)

    # Use VVC at QP 47
    qp = 47

    # Rep 1: Still
    r1 = evaluate_still_frame0(cleaned, frames, masks, qp=qp, codec="vvc", preset="faster")
    assert r1["representation"] == "still_frame0"
    assert r1["encoded_payload_bytes"] > 0
    assert r1["total_package_bytes"] == r1["encoded_payload_bytes"] + 10
    assert math.isfinite(r1["metrics"]["psnr_y_visible_dB"])

    # Rep 2: Panorama
    r2 = evaluate_registered_panorama(
        plate, homographies, frames, masks, qp=qp, codec="vvc", preset="faster"
    )
    assert r2["representation"] == "registered_panorama"
    assert r2["encoded_payload_bytes"] > 0
    assert r2["side_data_bytes"] == charge_side_data("registered_panorama", 4)["total_side_data_bytes"]
    assert math.isfinite(r2["metrics"]["psnr_y_visible_dB"])

    # Rep 3: Video
    r3 = evaluate_cleaned_video(
        cleaned, frames, masks, qp=qp, codec="vvc", preset="faster", work_dir=tmp_path
    )
    assert r3["representation"] == "cleaned_video"
    assert r3["encoded_payload_bytes"] > 0
    assert math.isfinite(r3["metrics"]["psnr_y_visible_dB"])
