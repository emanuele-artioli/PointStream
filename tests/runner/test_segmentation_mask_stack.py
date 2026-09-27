from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from src.contracts.lattice import ART_MASKS
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner.stages import (
    StageContext,
    _frame_mask,
    _subjects_for_reconstruct,
    make_pose,
    make_segmentation,
)


class _FrameSegmenter:
    def segment(self, frame, detection):
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[int(frame[0, 0, 0]), 2:4] = 1
        return mask


def _objects(frame_count: int) -> tuple[ObjectRequest, ...]:
    return tuple(
        ObjectRequest(
            object_id="player",
            appearance=np.zeros((2, 2, 3), dtype=np.uint8),
            bbox=(0, 0, 5, 4),
            frame_index=index,
            object_class="person",
        )
        for index in range(frame_count)
    )


def test_full_frame_mask_is_preserved_without_resizing_into_the_detection_box() -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:3, 5:7] = 1
    restored = _frame_mask(mask, (0, 0, 2, 2), height=6, width=8)
    np.testing.assert_array_equal(restored, mask.astype(bool))


def test_segmenter_keeps_each_frame_mask_in_a_track_aligned_stack() -> None:
    frames = np.zeros((2, 6, 8, 3), dtype=np.uint8)
    frames[1] = 1
    objects = _objects(2)
    result = make_segmentation(cast(StageContext, SimpleNamespace(segmenter=_FrameSegmenter())))(
        {SOURCE: frames, "detection": objects}
    )
    assert result["player"].shape == (2, 6, 8)
    assert result["player"][0, 0, 2]
    assert result["player"][1, 1, 2]
    assert result["player"].sum() == 4

    resolved = _subjects_for_reconstruct(
        {SOURCE: frames, "detection": objects, ART_MASKS: result}
    )
    assert len(resolved) == 2
    assert resolved[0].mask is not None and resolved[1].mask is not None
    np.testing.assert_array_equal(resolved[0].mask, result["player"][0])
    np.testing.assert_array_equal(resolved[1].mask, result["player"][1])


def test_pose_estimator_receives_the_matching_frame_segmentation_mask() -> None:
    class _PoseEstimator:
        def __init__(self):
            self.masks = []

        def estimate(self, frame, detection, *, mask=None):
            self.masks.append(np.asarray(mask).copy())
            return None

    frames = np.zeros((2, 6, 8, 3), dtype=np.uint8)
    objects = _objects(2)
    masks = np.zeros((2, 6, 8), dtype=np.uint8)
    masks[0, 1, 2] = 1
    masks[1, 4, 6] = 1
    estimator = _PoseEstimator()
    make_pose(cast(StageContext, SimpleNamespace(pose_estimator=estimator)))(
        {SOURCE: frames, "detection": objects, ART_MASKS: {"player": masks}}
    )
    np.testing.assert_array_equal(estimator.masks[0], masks[0])
    np.testing.assert_array_equal(estimator.masks[1], masks[1])
