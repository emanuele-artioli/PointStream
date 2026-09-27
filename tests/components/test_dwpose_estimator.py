from __future__ import annotations

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.pose.dwpose import DWPoseEstimator
from src.contracts.keypoints import COCO_WHOLEBODY_133


class _FakeWholebody:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray):
        self.inputs.append(frame.copy())
        points = np.zeros((1, 133, 2), dtype=np.float32)
        scores = np.zeros((1, 133), dtype=np.float32)
        points[0, 10] = (5.0, 6.0)
        scores[0, 10] = 0.8
        points[0, 9] = (2.0, 4.0)
        scores[0, 9] = 0.2
        return points, scores


def test_dwpose_uses_crop_coordinates_and_preserves_joint_visibility() -> None:
    model = _FakeWholebody()
    estimator = DWPoseEstimator(model=model, device="cpu")
    frame = np.full((30, 40, 3), 255, dtype=np.uint8)
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:25, 7:31] = 1
    detection = Detection("person", Box(5, 3, 35, 28), track_id="p0")
    pose = estimator.estimate(frame, detection, mask=mask)
    assert pose is not None
    assert pose.schema == COCO_WHOLEBODY_133
    np.testing.assert_allclose(pose.values[10], [10.0, 9.0, 0.8])
    assert pose.visibility[10] == 2
    assert pose.visibility[9] == 1
    assert pose.present[9]
    assert pose.visibility[0] == 0
    assert not pose.present[0]
    assert pose.values[0].tolist() == [0.0, 0.0, 0.0]
    assert np.all(model.inputs[0][:2] == 0)  # pixels outside the supplied mask
