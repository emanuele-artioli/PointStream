"""Background warp, delta reconstruction, and compositor placement."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.reconstruction import (
    BackgroundModelView,
    BackgroundResolver,
    apply_plate_delta,
    bit_identical,
    composite_clip,
    warp_plate,
)
from src.pipeline.reconstruction.compositor import Placement, heuristic_mask


def _identity() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_identity_warp_of_a_matching_plate_is_bit_identical() -> None:
    plate = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    frames = warp_plate(plate, (_identity(), _identity()), height=32, width=32, frame_count=2)
    assert bit_identical(np.stack([plate, plate]), frames)


def test_missing_homographies_are_identity_not_an_invented_camera() -> None:
    plate = np.full((16, 16, 3), 77, dtype=np.uint8)
    frames = warp_plate(plate, (), height=16, width=16, frame_count=3)
    assert frames.shape == (3, 16, 16, 3)
    assert np.all(frames == 77)


def test_delta_then_apply_restores_the_plate() -> None:
    previous = np.full((8, 8, 3), 40, dtype=np.uint8)
    current = np.full((8, 8, 3), 50, dtype=np.uint8)
    diff = np.clip(current.astype(np.int16) - previous.astype(np.int16) + 128, 0, 255).astype(
        np.uint8
    )
    restored = apply_plate_delta(previous, diff)
    assert bit_identical(current[None], restored[None])


def test_delta_without_a_prior_plate_is_a_protocol_violation() -> None:
    resolver = BackgroundResolver()
    view = BackgroundModelView(
        plate=np.full((4, 4, 3), 128, dtype=np.uint8),
        mode="delta",
        scene_id="scene-a",
    )
    with pytest.raises(ValueError, match="previously decoded"):
        resolver.resolve(view)


def test_full_then_delta_reconstructs_across_chunks() -> None:
    resolver = BackgroundResolver()
    first = np.full((6, 6, 3), 30, dtype=np.uint8)
    second = np.full((6, 6, 3), 40, dtype=np.uint8)
    resolver.resolve(
        BackgroundModelView(plate=first, mode="full", scene_id="rally")
    )
    diff = np.clip(second.astype(np.int16) - first.astype(np.int16) + 128, 0, 255).astype(
        np.uint8
    )
    restored = resolver.resolve(
        BackgroundModelView(plate=diff, mode="delta", scene_id="rally")
    )
    assert restored is not None
    assert bit_identical(second[None], restored[None])


def test_deferred_background_is_zeros() -> None:
    resolver = BackgroundResolver()
    frames, _ = resolver.frames_for(
        BackgroundModelView(plate=None, deferred_to_residual=True, mode="none"),
        frame_count=2,
        height=8,
        width=8,
    )
    assert np.all(frames == 0)


def test_unknown_background_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        BackgroundModelView(plate=np.zeros((2, 2, 3), dtype=np.uint8), mode="warp")


def test_compositing_an_object_onto_a_background_is_bit_identical_inside_the_box() -> None:
    background = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    crop = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = composite_clip(
        background,
        (Placement(crop=crop, bbox=(2, 2, 10, 10), frame_index=0),),
        use_heuristic_mask=True,
    )
    assert np.all(out[0, 2:10, 2:10] == 200)
    assert np.all(out[0, 0, 0] == 0)


def test_heuristic_mask_is_the_bbox_rectangle() -> None:
    mask = heuristic_mask((4, 4, 12, 10), height=16, width=16)
    assert mask.sum() == 8 * 6
    assert mask[4, 4] and not mask[3, 4]


def test_segmentation_off_ignores_a_provided_mask() -> None:
    """Using a mask while segmentation is off would mean we consumed a disabled
    stage's output. The heuristic bbox must win."""
    background = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    crop = np.full((8, 8, 3), 255, dtype=np.uint8)
    tiny = np.zeros((8, 8), dtype=np.uint8)
    tiny[0, 0] = 255
    out = composite_clip(
        background,
        (Placement(crop=crop, bbox=(0, 0, 8, 8), mask=tiny),),
        use_heuristic_mask=True,
    )
    assert np.all(out[0, 0:8, 0:8] == 255)


def test_inverted_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="inverted"):
        Placement(crop=np.zeros((2, 2, 3), dtype=np.uint8), bbox=(8, 8, 2, 2))


def test_placement_past_the_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="frames"):
        composite_clip(
            np.zeros((1, 8, 8, 3), dtype=np.uint8),
            (Placement(crop=np.zeros((2, 2, 3), dtype=np.uint8), bbox=(0, 0, 2, 2), frame_index=3),),
            use_heuristic_mask=True,
        )


def test_singular_homography_is_rejected() -> None:
    plate = np.zeros((8, 8, 3), dtype=np.uint8)
    zero = (0.0,) * 9
    with pytest.raises(ValueError, match="singular"):
        warp_plate(plate, (zero,), height=8, width=8, frame_count=1)
