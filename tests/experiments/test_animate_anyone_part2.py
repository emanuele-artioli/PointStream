"""Focused wire and decoder checks for the Animate Anyone development screen."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.modular import animate_anyone_part2 as probe
from src.contracts.keypoints import COCO_17_JOINTS, OPENPOSE_18_JOINTS


def test_all_named_joints_project_and_neck_uses_both_shoulders() -> None:
    pose = np.zeros((17, 3), dtype=np.float16)
    for i in range(17):
        pose[i] = (i + 1, 100 + i, 0.75)
    pose[COCO_17_JOINTS.index("left_shoulder"), 2] = 0.5
    pose[COCO_17_JOINTS.index("right_shoulder"), 2] = 0.9
    result = probe.coco17_to_openpose18(pose)
    for name in COCO_17_JOINTS:
        np.testing.assert_allclose(
            result[OPENPOSE_18_JOINTS.index(name)],
            pose[COCO_17_JOINTS.index(name)].astype(np.float32),
        )
    neck = result[OPENPOSE_18_JOINTS.index("neck")]
    left = result[OPENPOSE_18_JOINTS.index("left_shoulder")]
    right = result[OPENPOSE_18_JOINTS.index("right_shoulder")]
    np.testing.assert_allclose(neck, [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, 0.5])


def test_missing_and_nonfinite_joints_are_not_fabricated() -> None:
    pose = np.zeros((17, 3), dtype=np.float16)
    pose[COCO_17_JOINTS.index("left_shoulder")] = (np.nan, np.nan, 0)
    pose[COCO_17_JOINTS.index("right_shoulder")] = (20, 10, 1)
    pose[COCO_17_JOINTS.index("left_eye")] = (np.nan, np.nan, np.nan)
    result = probe.coco17_to_openpose18(pose)
    np.testing.assert_array_equal(result[OPENPOSE_18_JOINTS.index("neck")], 0)
    np.testing.assert_array_equal(result[OPENPOSE_18_JOINTS.index("left_eye")], 0)
    pose[COCO_17_JOINTS.index("left_shoulder"), 2] = 1
    with pytest.raises(ValueError, match="left_shoulder"):
        probe.coco17_to_openpose18(pose)
    with pytest.raises(ValueError, match="17,3"):
        probe.coco17_to_openpose18(np.zeros((18, 3)))


def test_two_objects_are_charged_independently() -> None:
    objects = [
        {"crop_wire": b"abc", "bbox_wire": b"1234", "presence_wire": b"1", "pose_wire": b"x" * 6, "alpha_wire": b"pq"},
        {"crop_wire": b"defgh", "bbox_wire": b"5678", "presence_wire": b"2", "pose_wire": b"y" * 6, "alpha_wire": b"rst"},
    ]
    parts = probe._byte_components(objects, 24648)
    assert parts == {"B": 24648, "F": 8, "M": 22, "R": 0, "H": 6, "total_bytes": 24684}
    with pytest.raises(ValueError, match="empty"):
        probe._byte_components([{**objects[0], "crop_wire": b""}], 24648)


def test_decoder_composite_uses_two_crops_in_rgb_order_without_target_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "mask" not in inspect.signature(probe._compose_generated).parameters
    monkeypatch.setattr(probe, "CANVAS", 8)
    background = np.zeros((8, 16, 3), dtype=np.uint8)
    objects = [
        {"crop_bgr": np.zeros((4, 4, 3), dtype=np.uint8), "alpha": np.ones((4, 4), bool),
         "boxes": [(0, 4, 0, 4)], "presence": np.array([True])},
        {"crop_bgr": np.zeros((4, 4, 3), dtype=np.uint8), "alpha": np.ones((4, 4), bool),
         "boxes": [(4, 8, 12, 16)], "presence": np.array([True])},
    ]
    generated = [
        np.full((1, 8, 8, 3), (3, 2, 1), dtype=np.uint8),
        np.full((1, 8, 8, 3), (6, 5, 4), dtype=np.uint8),
    ]
    frame = probe._compose_generated(background, objects, generated, 0)
    assert frame[1, 1].tolist() == [1, 2, 3]
    assert frame[5, 13].tolist() == [4, 5, 6]
    assert frame[1, 13].tolist() == [0, 0, 0]
