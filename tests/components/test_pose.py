"""Canonical internally; the wire schema is whatever the generator consumes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.pose import REGISTRY as POSE
from src.components.pose.wire import from_coco17, to_wire, wire_schema
from src.components.pose.yolo import YoloPoseEstimator
from src.contracts.errors import UnknownBackendError
from src.contracts.keypoints import CANONICAL_HUMAN, COCO_17, OPENPOSE_18


def _coco17(conf: float = 0.9) -> np.ndarray:
    values = np.zeros((17, 3), dtype=np.float32)
    for index, name in enumerate(COCO_17.joints):
        values[index] = (10.0 + index, 20.0 + index, conf)
    return values


def test_pose_backend_is_registered_as_yolo() -> None:
    assert POSE.spec("yolo").name == "yolo"
    assert POSE.spec("yolo-pose").name == "yolo"


def test_unknown_pose_name_suggests_yolo() -> None:
    with pytest.raises(UnknownBackendError, match="Did you mean"):
        POSE.spec("yolo-pos")


def test_coco17_lifts_to_canonical_with_the_other_joints_absent() -> None:
    pose = from_coco17(_coco17())
    assert pose.schema is CANONICAL_HUMAN
    assert pose.values.shape == (133, 3)
    assert int(pose.present.sum()) == 17
    assert not pose.present[17:].any()
    assert np.all(pose.values[17:, 2] == 0.0)


def test_wire_schema_for_openpose_is_18_not_133() -> None:
    """Sending 133 joints to a conditioner that reads 18 is wasted payload."""
    assert wire_schema("openpose-18") is OPENPOSE_18
    assert len(wire_schema("openpose-18")) == 18
    canonical = from_coco17(_coco17())
    wire = to_wire(canonical, "openpose-18")
    assert wire.schema is OPENPOSE_18
    assert wire.values.shape == (18, 3)
    assert wire.present[OPENPOSE_18.index_of["nose"]]
    assert wire.present[OPENPOSE_18.index_of["neck"]]
    # Neck is synthesised from the shoulders, not copied from a source joint.
    left = canonical.values[CANONICAL_HUMAN.index_of["left_shoulder"], :2]
    right = canonical.values[CANONICAL_HUMAN.index_of["right_shoulder"], :2]
    np.testing.assert_allclose(wire.values[OPENPOSE_18.index_of["neck"], :2], 0.5 * (left + right))


def test_absent_source_joints_stay_absent_on_the_wire_not_zero_filled_as_present() -> None:
    values = _coco17()
    values[0] = (0.0, 0.0, 0.0)  # nose missing
    wire = to_wire(from_coco17(values), OPENPOSE_18)
    nose = OPENPOSE_18.index_of["nose"]
    assert not wire.present[nose]
    assert wire.values[nose, 2] == 0.0


def test_yolo_pose_estimator_projects_mocked_coco17_onto_the_wire() -> None:
    keypoints = SimpleNamespace(data=_coco17()[None, ...])
    result = SimpleNamespace(keypoints=keypoints)
    model = SimpleNamespace(predict=lambda **_kwargs: [result])
    estimator = YoloPoseEstimator(model=model)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    detection = Detection("player", Box(5, 5, 40, 60))
    pose = estimator.estimate_to_schema(frame, detection, "openpose-18")
    assert pose is not None
    assert pose.values.shape == (18, 3)


def test_pose_is_skipped_for_a_class_without_a_skeleton() -> None:
    def _should_not_run(**_kwargs: object) -> None:
        raise AssertionError("pose estimator must not run on a racket")

    estimator = YoloPoseEstimator(model=SimpleNamespace(predict=_should_not_run))
    pose = estimator.estimate(
        np.zeros((40, 40, 3), dtype=np.uint8),
        Detection("racket", Box(1, 1, 8, 8)),
    )
    assert pose is None
