"""Ad-hoc tennis heuristic: two players, not ball kids or crowd."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import Detection, is_person, is_racket

_FAR_SLOT = "player_far"
_NEAR_SLOT = "player_near"


class HeuristicSelector:
    """Pick the two on-court players and the rackets attached to them.

    Scoring prefers large, central people who overlap a racket and who continue
    last frame's tracks. A missing player is held from history rather than
    replaced by the next-largest person — that next person is usually a ball
    kid or a spectator.
    """

    def __init__(self) -> None:
        self._last: dict[str, Detection] = {}

    def select(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        height, width = frame_shape[0], frame_shape[1]
        people = [item for item in detections if is_person(item.class_name)]
        rackets = [item for item in detections if is_racket(item.class_name)]

        far, near = self._pick_players(people, rackets, width=width, height=height)
        far = self._stabilize(_FAR_SLOT, far, height=height)
        near = self._stabilize(_NEAR_SLOT, near, height=height)

        selected_people = [item for item in (far, near) if item is not None]
        selected_rackets = self._pick_rackets(rackets, selected_people)
        return [*selected_people, *selected_rackets]

    def _pick_players(
        self,
        people: Sequence[Detection],
        rackets: Sequence[Detection],
        *,
        width: int,
        height: int,
    ) -> tuple[Detection | None, Detection | None]:
        if not people:
            return None, None

        midline = height * 0.5
        scored = sorted(
            people,
            key=lambda person: self._score(person, rackets, width=width, height=height),
            reverse=True,
        )
        above = [item for item in scored if item.bbox.center[1] < midline]
        below = [item for item in scored if item.bbox.center[1] >= midline]

        far = self._slot_pick(above, _FAR_SLOT, width=width, height=height, anchor_y=height * 0.30)
        near = self._slot_pick(below, _NEAR_SLOT, width=width, height=height, anchor_y=height * 0.72)

        used = {id(item) for item in (far, near) if item is not None}
        remaining = [item for item in scored if id(item) not in used]
        if far is None:
            far = self._first_plausible(remaining, _FAR_SLOT, width=width, height=height)
            if far is not None:
                remaining = [item for item in remaining if item is not far]
        if near is None:
            near = self._first_plausible(remaining, _NEAR_SLOT, width=width, height=height)
        return far, near

    def _slot_pick(
        self,
        candidates: Sequence[Detection],
        slot: str,
        *,
        width: int,
        height: int,
        anchor_y: float,
    ) -> Detection | None:
        plausible = [
            item
            for item in candidates
            if self._is_plausible(slot, item, width=width, height=height)
        ]
        if not plausible:
            return None
        previous = self._last.get(slot)
        if previous is not None:
            return min(plausible, key=lambda item: _center_distance(item.bbox, previous.bbox))
        anchor = Box(width * 0.5, anchor_y, width * 0.5, anchor_y)
        return min(plausible, key=lambda item: _center_distance(item.bbox, anchor))

    def _first_plausible(
        self,
        candidates: Sequence[Detection],
        slot: str,
        *,
        width: int,
        height: int,
    ) -> Detection | None:
        for item in candidates:
            if self._is_plausible(slot, item, width=width, height=height):
                return item
        return None

    def _is_plausible(
        self, slot: str, candidate: Detection, *, width: int, height: int
    ) -> bool:
        previous = self._last.get(slot)
        if previous is None:
            return True
        prev_area = max(1.0, previous.bbox.area)
        cand_area = max(1.0, candidate.bbox.area)
        ratio = max(prev_area / cand_area, cand_area / prev_area)
        if ratio > 4.0:
            return False
        distance = _center_distance(candidate.bbox, previous.bbox)
        frame_diag = max(1.0, float(np.hypot(width, height)))
        max_distance = max(float(height) * 0.35, previous.bbox.height * 2.0, frame_diag * 0.12)
        return distance <= max_distance

    def _score(
        self,
        person: Detection,
        rackets: Sequence[Detection],
        *,
        width: int,
        height: int,
    ) -> float:
        overlap = sum(person.bbox.iou(racket.bbox) for racket in rackets)
        cx, _cy = person.bbox.center
        center_score = 1.0 - abs(cx - width * 0.5) / max(1.0, width * 0.5)
        temporal = 0.0
        frame_diag = max(1.0, float(np.hypot(width, height)))
        for previous in self._last.values():
            distance = _center_distance(person.bbox, previous.bbox)
            temporal = max(temporal, max(0.0, 1.0 - min(distance / frame_diag, 1.0)) * 220.0)
        return overlap + person.bbox.area * 0.0015 + center_score * 100.0 + temporal

    def _stabilize(
        self, slot: str, candidate: Detection | None, *, height: int
    ) -> Detection | None:
        previous = self._last.get(slot)
        if candidate is None:
            return previous
        named = candidate.with_class_name("player").with_track_id(slot)
        if previous is None:
            self._last[slot] = named
            return named
        movement = _center_distance(named.bbox, previous.bbox)
        max_step = max(float(height) * 0.18, previous.bbox.height * 2.2)
        if movement > max_step:
            held = previous.with_track_id(slot)
            self._last[slot] = held
            return held
        self._last[slot] = named
        return named

    def _pick_rackets(
        self, rackets: Sequence[Detection], players: Sequence[Detection]
    ) -> list[Detection]:
        if not rackets:
            return []
        if len(rackets) <= 2 and not players:
            return [
                item.with_class_name("racket").with_track_id(f"racket_{index}")
                for index, item in enumerate(rackets[:2])
            ]
        chosen: list[Detection] = []
        used: set[int] = set()
        slot_for = {_FAR_SLOT: "racket_far", _NEAR_SLOT: "racket_near"}
        for player in players:
            racket_slot = slot_for.get(player.track_id or "", f"racket_{len(chosen)}")
            best_index = None
            best_distance = float("inf")
            for index, racket in enumerate(rackets):
                if index in used:
                    continue
                distance = _center_distance(racket.bbox, player.bbox)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                continue
            used.add(best_index)
            chosen.append(
                rackets[best_index].with_class_name("racket").with_track_id(racket_slot)
            )
        return chosen[:2]


def _center_distance(left: Box, right: Box) -> float:
    lx, ly = left.center
    rx, ry = right.center
    return float(np.hypot(lx - rx, ly - ry))
