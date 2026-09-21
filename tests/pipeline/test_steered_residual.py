"""Tests for steered residual encoding: cropped actor residual and band-limited background."""

import numpy as np
import pytest

from src.pipeline.residual.steered_residual import (
    ActorMaskProcessor,
    BandLimitedBackgroundResidual,
    CroppedActorResidualDecoder,
    CroppedActorResidualEncoder,
    CroppedResidualPayload,
)


def _make_scene(height: int = 720, width: int = 1280):
    """Create synthetic scene with background and a moving actor patch."""
    bg = np.full((height, width, 3), [180, 105, 30], dtype=np.uint8)
    # Add high-frequency court noise
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 5.0, bg.shape).astype(np.int16)
    target = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Actor in center (red shirt [50, 50, 200])
    mask = np.zeros((height, width), dtype=np.uint8)
    ay1, ay2, ax1, ax2 = 200, 450, 500, 700
    mask[ay1:ay2, ax1:ax2] = 255
    target[ay1:ay2, ax1:ax2] = [50, 50, 200]

    # Reconstructed canvas: background slightly degraded, actor slightly shifted
    canvas = target.copy()
    # Degraded background (smoothed court)
    canvas[mask == 0] = bg[mask == 0]
    # Actor imperfect reconstruction (e.g. from keyframe warp)
    canvas[ay1:ay2, ax1:ax2] = np.clip(
        target[ay1:ay2, ax1:ax2].astype(np.int16) - 30, 0, 255
    ).astype(np.uint8)

    return target, canvas, mask


# Group 1: Behaviour Tests
def test_actor_mask_elliptical_dilation():
    processor = ActorMaskProcessor(dilation_radius=8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255  # single center point

    dilated = processor.dilate_mask(mask)
    assert dilated[50, 50] == 255
    # Radius 8: point (50, 58) and (50, 42) must be covered
    assert dilated[50, 58] == 255
    assert dilated[50, 42] == 255
    # Point beyond radius (50, 60) must be 0
    assert dilated[50, 60] == 0


def test_compute_tight_bbox_returns_even_bounds():
    processor = ActorMaskProcessor(dilation_radius=4)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:121, 60:131] = 255

    bbox = processor.compute_tight_bbox(mask, pad=4, even_align=True)
    assert bbox is not None
    y1, y2, x1, x2 = bbox
    assert (y2 - y1) % 2 == 0
    assert (x2 - x1) % 2 == 0
    assert y1 <= 50 and y2 >= 121
    assert x1 <= 60 and x2 >= 131


def test_cropped_actor_residual_roundtrip_improves_quality():
    target, canvas, mask = _make_scene(720, 1280)
    encoder = CroppedActorResidualEncoder(codec="webp", quality=80)
    decoder = CroppedActorResidualDecoder()

    # Prior actor MSE
    actor_mse_before = np.mean((target[mask > 0].astype(float) - canvas[mask > 0].astype(float)) ** 2)

    payload = encoder.encode_frame(target, canvas, mask)
    assert payload is not None
    assert isinstance(payload, CroppedResidualPayload)

    restored = decoder.apply(canvas, payload)

    # Actor MSE should improve significantly
    actor_mse_after = np.mean((target[mask > 0].astype(float) - restored[mask > 0].astype(float)) ** 2)
    assert actor_mse_after < actor_mse_before * 0.3

    # Passthrough compositing invariant: background outside bbox must be bit-exact
    y1, y2, x1, x2 = payload.bbox
    outside_mask = np.ones((720, 1280), dtype=bool)
    outside_mask[y1:y2, x1:x2] = False
    assert np.array_equal(canvas[outside_mask], restored[outside_mask])


def test_cropped_actor_residual_size_under_four_point_five_kb():
    # 4K resolution scene with 700x600 actor
    target, canvas, mask = _make_scene(2160, 3840)
    encoder = CroppedActorResidualEncoder(codec="webp", quality=75)

    payload = encoder.encode_frame(target, canvas, mask)
    assert payload is not None
    # Invariant: cropped actor residual <= 4.5 kB (4608 bytes)
    assert payload.byte_count <= 4608, f"Payload {payload.byte_count} exceeds 4608 bytes"


def test_band_limited_background_suppresses_noise():
    target, canvas, mask = _make_scene(720, 1280)
    steerer = BandLimitedBackgroundResidual(downscale_factor=0.5)

    raw_diff = target.astype(np.int16) - canvas.astype(np.int16)
    band_limited_diff = steerer.compute_residual(target, canvas, mask)

    # In background region (mask == 0), band-limited residual should have lower variance
    raw_bg_std = np.std(raw_diff[mask == 0])
    bl_bg_std = np.std(band_limited_diff[mask == 0])
    assert bl_bg_std < raw_bg_std


# Group 2: Plausible Misuse Tests
def test_empty_mask_returns_none_cleanly():
    target, canvas, _ = _make_scene(100, 100)
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    encoder = CroppedActorResidualEncoder()

    payload = encoder.encode_frame(target, canvas, empty_mask)
    assert payload is None


def test_mismatched_target_canvas_shapes_raises():
    target = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas = np.zeros((120, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    encoder = CroppedActorResidualEncoder()

    with pytest.raises(ValueError, match="shape"):
        encoder.encode_frame(target, canvas, mask)


def test_apply_crop_with_invalid_bbox_raises():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    bad_payload = CroppedResidualPayload(
        payload=b"dummy",
        bbox=(50, 150, 0, 50),  # y2 > canvas height
        original_shape=(100, 100, 3),
    )
    decoder = CroppedActorResidualDecoder()
    with pytest.raises(ValueError, match="bounds"):
        decoder.apply(canvas, bad_payload)
