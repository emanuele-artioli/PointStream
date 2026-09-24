"""Behavioral checks for separate-object tracking and decoder-only warping."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.modular.object_foreground_campaign import (
    _boxes,
    _render_objects,
    split_object_tracks,
    warp_articulated,
)


def test_tracks_preserve_two_players_and_late_entrant() -> None:
    masks = np.zeros((4, 32, 64), dtype=bool)
    masks[:, 2:8, 2:8] = True
    masks[:, 18:26, 25:33] = True
    masks[2:, 10:16, 50:56] = True
    tracks = split_object_tracks(masks, min_area=10)
    assert len(tracks) == 3
    assert [int(np.flatnonzero(t.any(axis=(1, 2)))[0]) for t in tracks] == [0, 0, 2]
    np.testing.assert_array_equal(np.logical_or.reduce(tracks), masks)
    assert all(not np.any(a & b) for i, a in enumerate(tracks) for b in tracks[i+1:])
    boxes, presence, first = _boxes(tracks[2])
    assert first == 2
    assert presence.tolist() == [False, False, True, True]
    assert boxes[0] == boxes[2]


def test_detached_fragment_stays_with_player() -> None:
    masks = np.zeros((2, 100, 100), dtype=bool)
    masks[:, 15:70, 15:70] = True
    masks[1, 72:80, 60:68] = True
    tracks = split_object_tracks(masks, min_area=10)
    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0], masks)


def test_invalid_or_empty_masks_rejected() -> None:
    with pytest.raises(ValueError):
        split_object_tracks(np.zeros((2, 10), dtype=bool))
    with pytest.raises(ValueError):
        split_object_tracks(np.zeros((2, 10, 10), dtype=bool))


def test_each_object_pastes_its_own_crop_and_late_entry_is_absent_earlier() -> None:
    background = np.zeros((2, 32, 48, 3), dtype=np.uint8)
    first = {
        "first": 0, "presence": np.asarray([True, True]),
        "boxes": [(0, 8, 0, 8), (0, 8, 8, 16)],
        "crop": np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8),
        "alpha": np.ones((8, 8), dtype=bool),
    }
    second = {
        "first": 1, "presence": np.asarray([False, True]),
        "boxes": [(16, 24, 32, 40), (16, 24, 32, 40)],
        "crop": np.full((8, 8, 3), (40, 50, 60), dtype=np.uint8),
        "alpha": np.ones((8, 8), dtype=bool),
    }
    result, _seconds = _render_objects(background, [first, second], "bbox")
    assert tuple(result[0, 3, 3]) == (30, 20, 10)
    assert tuple(result[1, 3, 11]) == (30, 20, 10)
    assert tuple(result[0, 19, 35]) == (0, 0, 0)
    assert tuple(result[1, 19, 35]) == (60, 50, 40)


def test_articulated_pose_changes_geometry_and_uses_same_alpha() -> None:
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[5:35, 5:35] = (20, 100, 220)
    alpha = np.zeros((40, 40), dtype=bool)
    alpha[5:35, 5:35] = True
    src = np.zeros((17, 3), dtype=np.float16)
    dst = np.zeros((17, 3), dtype=np.float16)
    src[:4, :2] = [[10, 10], [30, 10], [10, 30], [30, 30]]
    dst[:4, :2] = [[10, 10], [30, 10], [15, 30], [35, 30]]
    src[:4, 2] = dst[:4, 2] = 1
    pixels, cover = warp_articulated(crop, alpha, src, dst, (0, 40, 0, 40), (50, 50))
    assert pixels.shape == (50, 50, 3)
    assert cover.shape == (50, 50)
    assert pixels[cover].shape[0] > 0
    assert np.any(cover[28:35, 31:37])
    assert np.all(pixels[~cover] == 0)


def test_nonfinite_joints_fall_back_to_bbox() -> None:
    crop = np.full((8, 8, 3), 77, dtype=np.uint8)
    alpha = np.ones((8, 8), dtype=bool)
    src = np.full((17, 3), np.nan, dtype=np.float16)
    dst = np.full((17, 3), np.nan, dtype=np.float16)
    pixels, cover = warp_articulated(
        crop, alpha, src, dst, (0, 8, 0, 8), (24, 24), target_box=(8, 16, 8, 16)
    )
    assert cover[8:16, 8:16].all()
    assert not cover[:8].any()
    assert np.all(pixels[cover] == 77)
