"""Recovery is a composed policy; a non-YOLO predictor can drive it."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.tracking import REGISTRY as TRACKING
from src.components.tracking.recovery import RecoveryPolicy
from src.components.tracking.tracker import IdentityTracker
from src.contracts.errors import UnknownBackendError


def _det(class_name: str, xyxy: list[float], track_id: str | None = None) -> Detection:
    return Detection(class_name=class_name, bbox=Box.from_xyxy(xyxy), track_id=track_id)


class _RoiStub:
    """Not a YOLO subclass. Recovery must still be able to call it."""

    def __init__(self, box: Box | None) -> None:
        self.box = box
        self.calls: list[tuple[Box, str]] = []

    def predict_roi(self, frame: object, bbox: Box, class_name: str) -> Box | None:
        self.calls.append((bbox, class_name))
        return self.box


def test_tracker_is_registered_under_the_config_default_name() -> None:
    assert TRACKING.spec("tracker").name == "tracker"
    built = TRACKING.build("tracker")
    assert isinstance(built, IdentityTracker)


def test_unknown_tracker_name_lists_the_registered_set() -> None:
    with pytest.raises(UnknownBackendError, match="Registered tracking backends"):
        TRACKING.spec("sort")


def test_recovery_reuses_a_non_yolo_predictor() -> None:
    previous = [_det("player", [10, 10, 30, 50], "player_far")]
    current: list[Detection] = []
    stub = _RoiStub(Box(12, 11, 32, 52))
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    recovered = RecoveryPolicy(quotas={"player": 1}).recover(
        frame=frame,
        detections=current,
        previous=previous,
        predictor=stub,
    )
    assert len(recovered) == 1
    assert recovered[0].bbox.xyxy == (12.0, 11.0, 32.0, 52.0)
    assert recovered[0].track_id == "player_far"
    assert stub.calls and stub.calls[0][1] == "player"


def test_recovery_holds_the_previous_box_when_the_predictor_finds_nothing() -> None:
    previous = [_det("player", [10, 10, 30, 50], "player_far")]
    recovered = RecoveryPolicy(quotas={"player": 1}).recover(
        frame=np.zeros((80, 80, 3), dtype=np.uint8),
        detections=[],
        previous=previous,
        predictor=_RoiStub(None),
    )
    assert recovered[0].bbox.xyxy == (10.0, 10.0, 30.0, 50.0)


def test_recovery_does_not_invent_a_box_when_there_is_no_history() -> None:
    """Synthesising a canned player is a silent wrong detection."""
    recovered = RecoveryPolicy(quotas={"player": 2}).recover(
        frame=np.zeros((80, 80, 3), dtype=np.uint8),
        detections=[],
        previous=None,
        predictor=_RoiStub(Box(1, 1, 2, 2)),
    )
    assert recovered == []


def test_identity_tracker_composes_recovery_and_reuses_ids() -> None:
    stub = _RoiStub(Box(11, 11, 31, 51))
    tracker = IdentityTracker(recovery=RecoveryPolicy(quotas={"player": 1}))
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    first = tracker.update(frame, [_det("player", [10, 10, 30, 50])], predictor=stub)
    assert first[0].track_id == "player_1"
    second = tracker.update(frame, [_det("player", [12, 12, 32, 52])], predictor=stub)
    assert second[0].track_id == "player_1"
    empty = tracker.update(frame, [], predictor=stub)
    assert empty[0].track_id == "player_1"
    assert stub.calls
