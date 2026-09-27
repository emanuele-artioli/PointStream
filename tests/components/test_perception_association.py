from __future__ import annotations

import numpy as np

from src.components.perception.association import RacketPlayerAssociator
from src.components.rigid.types import ObservedObject, PlayerPose


def _player(object_id: str, frame_index: int, wrist: tuple[float, float], confidence: float = 0.9):
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[10] = (*wrist, confidence)
    return PlayerPose(object_id, frame_index, joints, "coco-17")


def _racket(object_id: str, frame_index: int, wrist: tuple[float, float]) -> ObservedObject:
    mask = np.zeros((60, 100), dtype=np.uint8)
    x, y = map(int, wrist)
    mask[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = 1
    return ObservedObject(object_id, "racket", frame_index, mask=mask)


def test_association_uses_visible_wrist_distance_and_temporal_continuity() -> None:
    assoc = RacketPlayerAssociator()
    first = assoc.associate(
        [_racket("r0", 0, (30, 30))],
        [_player("p0", 0, (30, 30)), _player("p1", 0, (80, 30))],
    )
    assert first[0].associated_player_id == "p0"
    second = assoc.associate(
        [_racket("r0", 1, (47, 30))],
        [_player("p0", 1, (57, 30)), _player("p1", 1, (47, 30))],
    )
    # The 24 px continuity term keeps a plausible persistent pair through a crossing.
    assert second[0].associated_player_id == "p0"
    assert second[0].associated_wrist == "right_wrist"


def test_missing_or_distant_wrists_remain_unassociated() -> None:
    assoc = RacketPlayerAssociator(maximum_distance_px=10)
    racket = _racket("r0", 0, (20, 20))
    absent = assoc.associate([racket], [_player("p0", 0, (90, 50), confidence=0.0)])
    assert absent[0].associated_player_id is None
