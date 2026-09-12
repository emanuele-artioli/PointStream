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
    build_probe_identity,
    charge_side_data,
    check_bounds,
    compute_array_sha256,
    evaluate_cleaned_video,
    evaluate_registered_panorama,
    evaluate_still_frame0,
    masked_luma_psnr,
    pack_panorama_side_data,
    pack_side_data,
    pack_still_or_video_side_data,
    uncompressed_sanity_check,
    unpack_panorama_side_data,
    unpack_side_data,
    unpack_still_or_video_side_data,
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


@pytest.mark.integration
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
    assert (
        r2["side_data_bytes"] == charge_side_data("registered_panorama", 4)["total_side_data_bytes"]
    )
    assert math.isfinite(r2["metrics"]["psnr_y_visible_dB"])

    # Rep 3: Video
    r3 = evaluate_cleaned_video(
        cleaned, frames, masks, qp=qp, codec="vvc", preset="faster", work_dir=tmp_path
    )
    assert r3["representation"] == "cleaned_video"
    assert r3["encoded_payload_bytes"] > 0
    assert math.isfinite(r3["metrics"]["psnr_y_visible_dB"])


def test_warp_plate_to_frame_float32_casting() -> None:
    """Verify that homographies are cast to float32 before warping."""
    plate = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)

    # Construct a homography with subtle float64 precision differences
    h64 = np.array(
        [
            [1.00000000001, 0.00000000002, 2.50000000003],
            [0.00000000004, 1.00000000005, 1.25000000006],
            [0.00000000001, 0.00000000002, 1.00000000000],
        ],
        dtype=np.float64,
    )
    h32 = h64.astype(np.float32)

    warped_from_64 = warp_plate_to_frame(plate, h64, height=64, width=64)
    warped_from_32 = warp_plate_to_frame(plate, h32, height=64, width=64)

    # They must produce the exact same rendered pixels because float32 is enforced
    np.testing.assert_array_equal(warped_from_64, warped_from_32)


def test_panorama_side_data_binary_roundtrip() -> None:
    """Verify packed binary homography side data serialization, precision, and error handling."""
    n_frames = 8
    rng = np.random.default_rng(123)
    matrices = rng.standard_normal((n_frames, 3, 3)).astype(np.float32)
    # Ensure non-singular
    for i in range(n_frames):
        matrices[i] += np.eye(3, dtype=np.float32)

    plate_shape = (2160, 4000)
    frame_shape = (2160, 3840)
    fps = 24.0

    payload = pack_panorama_side_data(
        matrices, plate_shape=plate_shape, frame_shape=frame_shape, fps=fps
    )
    expected_len = 14 + n_frames * 9 * 4
    assert len(payload) == expected_len

    unpacked_h, unpacked_plate, unpacked_frame, unpacked_fps = unpack_panorama_side_data(payload)
    assert unpacked_h.dtype == np.float32
    assert unpacked_h.shape == (n_frames, 3, 3)
    np.testing.assert_allclose(unpacked_h, matrices, atol=1e-7)
    assert unpacked_plate == plate_shape
    assert unpacked_frame == frame_shape
    assert unpacked_fps == fps

    # Test error cases
    with pytest.raises(ValueError, match="Payload too short"):
        unpack_panorama_side_data(b"too_short")

    with pytest.raises(ValueError, match="Payload length mismatch"):
        unpack_panorama_side_data(payload[:-4])


def test_still_and_video_side_data_binary_roundtrip() -> None:
    """Verify still/video side data binary roundtrip."""
    frame_shape = (2160, 3840)
    n_frames = 48
    fps = 24.0

    payload = pack_still_or_video_side_data(frame_shape, n_frames, fps=fps)
    assert len(payload) == 10

    unpacked_shape, unpacked_n, unpacked_fps = unpack_still_or_video_side_data(payload)
    assert unpacked_shape == frame_shape
    assert unpacked_n == n_frames
    assert unpacked_fps == fps

    with pytest.raises(ValueError, match="Payload length mismatch"):
        unpack_still_or_video_side_data(payload + b"\x00")


def test_unified_pack_unpack_side_data() -> None:
    """Verify unified pack_side_data and unpack_side_data interfaces."""
    # Rep 1: still
    p1 = pack_side_data("still_frame0", n_frames=48)
    assert len(p1) == 10
    u1 = unpack_side_data("still_frame0", p1)
    assert u1["n_frames"] == 48

    # Rep 2: panorama
    p2 = pack_side_data("registered_panorama", n_frames=48)
    assert len(p2) == 1742
    u2 = unpack_side_data("registered_panorama", p2)
    assert u2["homographies"].shape == (48, 3, 3)

    # Rep 3: video
    p3 = pack_side_data("cleaned_video", n_frames=48)
    assert len(p3) == 10
    u3 = unpack_side_data("cleaned_video", p3)
    assert u3["n_frames"] == 48


def test_identity_and_cache_key_hashing() -> None:
    """Verify that source frames, masks, and git diff affect probe identity and cache key."""
    frames, masks = _synthetic_clip(t=4, h=64, w=64)

    dummy_rev = {"commit": "abc1234", "dirty": False, "diff_sha256": None}
    ident = build_probe_identity("video_a", "scene_1", frames, masks, code_revision=dummy_rev)

    assert ident["frames_sha256"] == compute_array_sha256(frames)
    assert ident["masks_sha256"] == compute_array_sha256(masks)
    assert "frame_hashes" in ident
    assert "mask_hashes" in ident
    assert "code_revision" in ident
    assert "cache_key" in ident
    assert len(ident["frame_hashes"]) == 4
    assert len(ident["mask_hashes"]) == 4

    # Mutating a frame must change frames_sha256 and cache_key
    frames_mut = frames.copy()
    frames_mut[0, 0, 0, 0] = 200
    ident_mut_f = build_probe_identity(
        "video_a", "scene_1", frames_mut, masks, code_revision=dummy_rev
    )
    assert ident_mut_f["frames_sha256"] != ident["frames_sha256"]
    assert ident_mut_f["cache_key"] != ident["cache_key"]

    # Mutating a mask must change masks_sha256 and cache_key
    masks_mut = masks.copy()
    masks_mut[0, 0, 0] = not masks_mut[0, 0, 0]
    ident_mut_m = build_probe_identity(
        "video_a", "scene_1", frames, masks_mut, code_revision=dummy_rev
    )
    assert ident_mut_m["masks_sha256"] != ident["masks_sha256"]
    assert ident_mut_m["cache_key"] != ident["cache_key"]

    # Changing code revision must change cache_key
    dirty_rev = {"commit": "abc1234", "dirty": True, "diff_sha256": "feedbeef"}
    ident_mut_code = build_probe_identity(
        "video_a", "scene_1", frames, masks, code_revision=dirty_rev
    )
    assert ident_mut_code["cache_key"] != ident["cache_key"]


def test_build_common_cleaned_stack_cache_validation(tmp_path: Path) -> None:
    """Verify that build_common_cleaned_stack checks identity on cached stacks and sets validity note."""
    frames, masks = _synthetic_clip(t=4, h=64, w=64)
    cache_file = tmp_path / "cache_with_id.npz"

    dummy_rev = {"commit": "abc1234", "dirty": False, "diff_sha256": None}
    ident = build_probe_identity("video_a", "scene_1", frames, masks, code_revision=dummy_rev)

    cleaned, plate, homographies, stats = build_common_cleaned_stack(
        frames, masks, register=True, cache_path=cache_file, identity=ident
    )
    assert stats["from_cache"] is False
    assert "canvas_validity_note" in stats
    assert (
        "Validity mask covers valid composite canvas coordinates" in stats["canvas_validity_note"]
    )

    # Re-reading with matching identity succeeds from cache
    _, _, _, stats_cached = build_common_cleaned_stack(
        frames, masks, register=True, cache_path=cache_file, identity=ident
    )
    assert stats_cached["from_cache"] is True

    # Re-reading with mismatched identity ignores cache and rebuilds
    ident_mismatch = dict(ident)
    ident_mismatch["frames_sha256"] = "mismatched_sha"
    _, _, _, stats_rebuilt = build_common_cleaned_stack(
        frames, masks, register=True, cache_path=cache_file, identity=ident_mismatch
    )
    assert stats_rebuilt["from_cache"] is False


def test_preprocessing_timing_accounting() -> None:
    """Verify that evaluation points record preprocessing and total end-to-end timing."""
    frames, masks = _synthetic_clip(t=4, h=64, w=64)
    cleaned, plate, homographies, _ = build_common_cleaned_stack(frames, masks, register=True)

    # Provide a non-zero preprocessing time
    prep_time = 1.234
    res_still = evaluate_still_frame0(
        cleaned, frames, masks, qp=47, codec="vvc", preset="faster", preprocessing_seconds=prep_time
    )
    assert res_still["timing"]["preprocessing_seconds"] == 1.234
    assert res_still["timing"]["total_end_to_end_seconds"] == round(
        1.234
        + res_still["timing"]["encode_seconds"]
        + res_still["timing"]["decode_render_seconds"],
        3,
    )

    res_pano = evaluate_registered_panorama(
        plate,
        homographies,
        frames,
        masks,
        qp=47,
        codec="vvc",
        preset="faster",
        preprocessing_seconds=prep_time,
    )
    assert res_pano["timing"]["preprocessing_seconds"] == 1.234
    assert res_pano["timing"]["total_end_to_end_seconds"] == round(
        1.234 + res_pano["timing"]["encode_seconds"] + res_pano["timing"]["decode_render_seconds"],
        3,
    )
