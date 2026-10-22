"""K6b (hands) 2: crop geometry and FoundHand's keypoint conditioning (docs/experiments.md, 2026-10-22)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from experiments.visor import k6bh  # noqa: E402


def test_crop_box_is_a_square_around_points_and_mask() -> None:
    points = np.array([[100.0, 200.0], [140.0, 260.0]])
    mask = np.zeros((480, 640), bool)
    mask[180:300, 90:150] = True
    x0, y0, side = k6bh.crop_box(points, mask)
    assert side == int(np.ceil(1.25 * 120))
    assert x0 <= 90 and x0 + side >= 150 and y0 <= 180 and y0 + side >= 300


def test_crop_box_has_a_minimum_side() -> None:
    assert k6bh.crop_box(np.array([[10.0, 10.0], [11.0, 12.0]]), None)[2] == k6bh.MIN_CROP


def test_to_crop_maps_the_box_onto_the_crop() -> None:
    box = (50, 20, 128)
    corners = np.array([[50.0, 20.0], [178.0, 148.0]])
    assert np.allclose(k6bh.to_crop(corners, box), [[0, 0], [k6bh.SIZE, k6bh.SIZE]])


def test_crop_pixels_pads_outside_the_image() -> None:
    image = np.full((100, 100, 3), 200, np.uint8)
    crop = k6bh.crop_pixels(image, (-50, -50, 100))
    assert crop.shape == (k6bh.SIZE, k6bh.SIZE, 3)
    assert crop[0, 0].max() == 0 and crop[-1, -1].min() == 200


def test_paste_writes_only_the_mask_inside_the_frame() -> None:
    canvas = np.zeros((60, 60, 3), np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[10:20, 10:20] = True
    mask[55:, 55:] = True  # outside the box
    written = k6bh.paste(canvas, np.full((k6bh.SIZE, k6bh.SIZE, 3), 255, np.uint8), (0, 0, 40), mask)
    assert written[10:20, 10:20].all() and not written[55:, 55:].any()
    assert canvas[15, 15].min() == 255 and canvas[5, 5].max() == 0


def test_slots_put_the_right_hand_first() -> None:
    pts = np.arange(42, dtype=float).reshape(21, 2) + 1
    right, left = k6bh.slots(pts, True), k6bh.slots(pts, False)
    assert np.array_equal(right[:21], pts) and not right[21:].any()
    assert np.array_equal(left[21:], pts) and not left[:21].any()


def test_heatmaps_peak_at_the_keypoint_and_drop_invalid_ones() -> None:
    pts = np.zeros((42, 2))
    pts[3] = [64.0, 128.0]  # latent (8, 16)
    hm = k6bh.heatmaps(pts)
    assert hm.shape == (42, k6bh.LATENT, k6bh.LATENT)
    assert np.unravel_index(hm[3].argmax(), hm[3].shape) == (16, 8)
    assert not hm[0].any()  # (0, 0) is invalid, as FoundHand's check_keypoints_validity
