"""Unit checks for the measured single-appearance motion controls."""

from __future__ import annotations

import numpy as np

from experiments.modular.appearance_motion_probe import (
    _bbox_affine,
    _box_payload,
    _keypoint_affine,
)


def test_bbox_motion_wire_is_four_int16_values_per_frame() -> None:
    payload = _box_payload([(10, 30, 20, 40), (12, 32, 24, 44)])
    assert len(payload) == 2 * 8


def test_keypoint_motion_uses_keypoints_when_two_joints_are_visible() -> None:
    source = np.asarray([[10.0, 10.0, 1.0], [20.0, 10.0, 1.0]], dtype=np.float32)
    target = source.copy()
    target[:, 0] += 5.0
    matrix, mode = _keypoint_affine(source, target, (0, 40, 0, 40), (0, 40, 5, 45))
    assert mode == "keypoints"
    np.testing.assert_allclose(matrix[:, 2], [5.0, 0.0], atol=1.0)


def test_bbox_affine_fallback_scales_and_places_the_crop() -> None:
    matrix = _bbox_affine((0, 20, 0, 20), (10, 50, 20, 60))
    np.testing.assert_allclose(matrix, [[2.0, 0.0, 20.0], [0.0, 2.0, 10.0]])
