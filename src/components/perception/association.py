"""Spatially grounded class identity and racket/player association helpers."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

import numpy as np

from src.components.rigid.types import ObservedObject, PlayerPose
from src.contracts.keypoints import CANONICAL_HUMAN, schema as resolve_schema


class RacketPlayerAssociator:
    """Associate each racket with a nearby visible wrist and preserve continuity."""

    def __init__(self, *, minimum_confidence: float = 0.2, maximum_distance_px: float = 96.0) -> None:
        if not 0.0 <= minimum_confidence <= 1.0 or maximum_distance_px <= 0:
            raise ValueError("association confidence and distance thresholds are invalid")
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_distance_px = float(maximum_distance_px)
        self.previous_player_by_racket: dict[str, str] = {}

    def associate(
        self,
        rackets: Sequence[ObservedObject],
        player_poses: Sequence[PlayerPose],
    ) -> tuple[ObservedObject, ...]:
        """Return copied racket objects carrying explicit player/wrist links.

        Candidate links are ranked by wrist-to-mask distance with a continuity
        cost for changing player identity. Greedy one-to-one assignment avoids
        assigning the same visible wrist to two racket tracks in one frame.
        No bbox-size heuristic or contour-area ranking is used.
        """
        candidates: list[tuple[float, str, str, str, tuple[float, float]]] = []
        racket_by_id = {item.object_id: item for item in rackets}
        players_by_frame: dict[int, list[PlayerPose]] = {}
        for pose in player_poses:
            players_by_frame.setdefault(pose.frame_index, []).append(pose)
        for racket in rackets:
            for pose in players_by_frame.get(racket.frame_index, []):
                for wrist_name, point, confidence in _visible_wrists(
                    pose, self.minimum_confidence
                ):
                    distance = _distance_to_object(racket, point)
                    if distance > self.maximum_distance_px:
                        continue
                    if self.previous_player_by_racket.get(racket.object_id) == pose.object_id:
                        distance -= min(24.0, self.maximum_distance_px * 0.25)
                    else:
                        distance += min(12.0, self.maximum_distance_px * 0.125)
                    distance += (1.0 - confidence) * 2.0
                    candidates.append(
                        (distance, racket.object_id, pose.object_id, wrist_name, point)
                    )
        assigned_rackets: set[str] = set()
        assigned_wrists: set[tuple[str, str, int]] = set()
        assignment: dict[str, tuple[str, str]] = {}
        for _, racket_id, player_id, wrist_name, _point in sorted(candidates):
            racket = racket_by_id[racket_id]
            wrist_key = (player_id, wrist_name, racket.frame_index)
            if racket_id in assigned_rackets or wrist_key in assigned_wrists:
                continue
            assignment[racket_id] = (player_id, wrist_name)
            assigned_rackets.add(racket_id)
            assigned_wrists.add(wrist_key)
            self.previous_player_by_racket[racket_id] = player_id
        result: list[ObservedObject] = []
        for racket in rackets:
            associated = assignment.get(racket.object_id)
            result.append(
                replace(
                    racket,
                    associated_player_id=associated[0] if associated else None,
                    associated_wrist=associated[1] if associated else None,
                )
            )
        return tuple(result)


def _visible_wrists(
    pose: PlayerPose,
    minimum_confidence: float,
) -> list[tuple[str, tuple[float, float], float]]:
    joints = np.asarray(pose.keypoints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] < 2:
        return []
    try:
        indexes = resolve_schema(pose.schema_name).index_of
    except ValueError:
        indexes = CANONICAL_HUMAN.index_of
    result: list[tuple[str, tuple[float, float], float]] = []
    for name in ("left_wrist", "right_wrist"):
        index = indexes.get(name)
        if index is None or index >= len(joints):
            continue
        x, y = float(joints[index, 0]), float(joints[index, 1])
        confidence = float(joints[index, 2]) if joints.shape[1] > 2 else 1.0
        if confidence < minimum_confidence or not math.isfinite(x + y + confidence):
            continue
        result.append((name, (x, y), confidence))
    return result


def _distance_to_object(obj: ObservedObject, point: tuple[float, float]) -> float:
    if obj.mask is not None:
        mask = np.asarray(obj.mask)
        if mask.ndim == 2 and mask.size:
            ys, xs = np.nonzero(mask)
            if xs.size:
                return float(np.sqrt(np.min((xs - point[0]) ** 2 + (ys - point[1]) ** 2)))
    if obj.bbox is not None:
        x0, y0, x1, y1 = obj.bbox
        dx = max(x0 - point[0], 0.0, point[0] - x1)
        dy = max(y0 - point[1], 0.0, point[1] - y1)
        return float(math.hypot(dx, dy))
    return math.inf


__all__ = ["RacketPlayerAssociator"]
