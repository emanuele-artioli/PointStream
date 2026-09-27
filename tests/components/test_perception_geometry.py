from __future__ import annotations

import numpy as np
import pytest

from src.components.perception.coordinates import (
    crop_and_mask,
    make_crop_transform,
    render_object_view,
)
from src.components.perception.conditioning import render_conditioning_view
from src.components.pose.wire import Pose
from src.contracts.keypoints import COCO_WHOLEBODY_133
from src.runner.perception import render_runtime_conditioning_view


def test_crop_transform_roundtrips_points_with_padding() -> None:
    transform = make_crop_transform(
        (20.0, 10.0, 60.0, 50.0),
        source_size=(100, 80),
        target_size=(64, 48),
    )
    points = np.asarray([[20.0, 10.0], [39.5, 21.25], [60.0, 50.0]])
    restored = transform.canvas_to_source(transform.source_to_canvas(points))
    np.testing.assert_allclose(restored, points, atol=1e-10)
    assert transform.pad_left > 0
    assert transform.to_record()["kind"] == "crop_resize_pad_v1"


def test_masked_crop_blackens_background_and_restores_source_coordinates() -> None:
    frame = np.full((12, 16, 3), 240, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 5:11] = 1
    transform = make_crop_transform((3, 2, 13, 10), source_size=(16, 12), target_size=(20, 20))
    crop, resized_mask = crop_and_mask(frame, mask, transform)
    assert crop.shape == (20, 20, 3)
    assert resized_mask.shape == (20, 20)
    assert np.all(crop[resized_mask == 0] == 0)
    restored = transform.restore_mask(resized_mask)
    assert restored[3:9, 5:11].mean() > 0.9


def test_crop_transform_rejects_degenerate_or_outside_boxes() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        make_crop_transform((4, 4, 4, 8), source_size=(16, 12), target_size=(20, 20))


def test_shared_object_view_records_appearance_transform() -> None:
    frame = np.full((12, 16, 3), 240, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 5:11] = 1
    crop, condition_mask, transform = render_object_view(
        frame,
        mask,
        (3, 2, 13, 10),
        target_size=(20, 24),
    )
    assert crop.shape == (24, 20, 3)
    assert condition_mask.shape == (24, 20)
    assert transform.to_record()["kind"] == "crop_resize_pad_v1"
    assert np.all(crop[condition_mask == 0] == 0)


def test_offline_and_runtime_conditioning_use_identical_reversible_views() -> None:
    frame = np.full((30, 40, 3), 180, dtype=np.uint8)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[4:26, 12:28] = 1
    values = np.zeros((len(COCO_WHOLEBODY_133), 3), dtype=np.float32)
    for name, point in (("left_shoulder", (18, 9)), ("left_elbow", (16, 14))):
        values[COCO_WHOLEBODY_133.index_of[name]] = (*point, 0.9)
    present = values[:, 2] > 0
    pose = Pose(COCO_WHOLEBODY_133, values, present)
    pose_transform = make_crop_transform(
        (0, 0, 80, 60), source_size=(80, 60), target_size=(64, 64)
    )
    offline = render_conditioning_view(
        frame,
        mask,
        (10, 2, 30, 28),
        target_size=(64, 64),
        pose=pose,
        pose_transform=pose_transform,
    )
    runtime = render_runtime_conditioning_view(
        frame,
        mask,
        (10, 2, 30, 28),
        target_size=(64, 64),
        pose=pose,
        pose_transform=pose_transform,
    )
    np.testing.assert_array_equal(runtime.appearance, offline.appearance)
    np.testing.assert_array_equal(runtime.mask, offline.mask)
    assert runtime.pose is not None and offline.pose is not None
    np.testing.assert_array_equal(runtime.pose, offline.pose)
    assert runtime.appearance_transform.to_record() == offline.appearance_transform.to_record()
    assert runtime.pose_transform.to_record() == pose_transform.to_record()
