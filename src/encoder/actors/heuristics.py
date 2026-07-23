"""Deciding which detections are the players.

A tennis frame contains ball kids, line judges and crowd; the heuristic
is what keeps the payload to the two actors that matter."""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import numpy as np
from src.shared.schemas import FrameState, SceneActor
from src.shared.geometry import get_iou
from src.encoder.actors.weights import _bbox_area, _bbox_center
_COCO_PERSON_CLASS_ID = 0
_COCO_TENNIS_RACKET_CLASS_ID = 38
_LOGGER = logging.getLogger(__name__)


class BaseHeuristic(ABC):
    @abstractmethod
    def select(self, frame_state: FrameState, frame_shape: tuple[int, int]) -> FrameState:
        raise NotImplementedError
class StandardTennisHeuristic(BaseHeuristic):
    def __init__(self) -> None:
        self._last_player_bboxes: dict[str, list[float]] = {}
        self._last_player_source_track_ids: dict[str, str] = {}

    def select(self, frame_state: FrameState, frame_shape: tuple[int, int]) -> FrameState:
        h, w = frame_shape
        players = [a for a in frame_state.actors if a.class_name == "player"]
        rackets = [a for a in frame_state.actors if a.class_name == "racket"]

        selected_players = self._select_players(players=players, rackets=rackets, frame_width=w, frame_height=h)
        stabilized_players = self._stabilize_players(selected_players=selected_players, frame_height=h)
        selected_rackets = self._select_rackets(rackets=rackets, selected_players=stabilized_players)
        stabilized_rackets = self._stabilize_rackets(
            selected_rackets=selected_rackets,
            stabilized_players=stabilized_players,
        )

        return FrameState(frame_id=frame_state.frame_id, actors=stabilized_players + stabilized_rackets)

    def _select_players(
        self,
        players: list[SceneActor],
        rackets: list[SceneActor],
        frame_width: int,
        frame_height: int,
    ) -> list[SceneActor]:
        if not players:
            return []

        selected = self._select_players_by_previous_track_id(players)
        selected_ids = {actor.track_id for actor in selected}

        if len(players) <= 2 and not selected:
            return players[:2]

        scored: list[tuple[float, SceneActor]] = []
        frame_diag = max(1.0, float(np.hypot(frame_width, frame_height)))
        for actor in players:
            overlap_score = sum(get_iou(actor.bbox, racket.bbox) for racket in rackets)
            area_score = _bbox_area(actor.bbox)
            cx, cy = _bbox_center(actor.bbox)
            center_score = 1.0 - abs(cx - frame_width * 0.5) / max(1.0, frame_width * 0.5)

            temporal_score = 0.0
            for previous_bbox in self._last_player_bboxes.values():
                px, py = _bbox_center(previous_bbox)
                distance = float(np.hypot(cx - px, cy - py))
                continuity = max(0.0, 1.0 - min(distance / frame_diag, 1.0))
                temporal_score = max(temporal_score, continuity * 220.0)

            score = overlap_score * 1.0 + area_score * 0.0015 + center_score * 100.0 + temporal_score
            scored.append((score, actor))

        remaining_scored = [item for item in scored if item[1].track_id not in selected_ids]

        if len(selected) < 2 and not selected:
            top_half = [item for item in remaining_scored if _bbox_center(item[1].bbox)[1] < frame_height * 0.5]
            bottom_half = [item for item in remaining_scored if _bbox_center(item[1].bbox)[1] >= frame_height * 0.5]
            has_history = bool(self._last_player_bboxes)

            if top_half:
                if has_history:
                    selected_actor = sorted(top_half, key=lambda p: p[0], reverse=True)[0][1]
                else:
                    selected_actor = min(
                        [actor for _, actor in top_half],
                        key=lambda actor: self._distance_to_anchor(
                            actor=actor,
                            anchor_x=frame_width * 0.5,
                            anchor_y=frame_height * 0.30,
                        ),
                    )
                selected.append(selected_actor)
                selected_ids.add(selected_actor.track_id)
            if bottom_half and len(selected) < 2:
                if has_history:
                    selected_actor = sorted(bottom_half, key=lambda p: p[0], reverse=True)[0][1]
                else:
                    selected_actor = min(
                        [actor for _, actor in bottom_half],
                        key=lambda actor: self._distance_to_anchor(
                            actor=actor,
                            anchor_x=frame_width * 0.5,
                            anchor_y=frame_height * 0.72,
                        ),
                    )
                if selected_actor.track_id not in selected_ids:
                    selected.append(selected_actor)
                    selected_ids.add(selected_actor.track_id)

        if len(selected) < 2:
            remaining_sorted = sorted(
                [item for item in remaining_scored if item[1].track_id not in selected_ids],
                key=lambda p: p[0],
                reverse=True,
            )
            for _, actor in remaining_sorted:
                if len(selected) >= 2:
                    break
                if len(selected) == 1:
                    slot = self._slot_for_source_track(selected[0].track_id)
                    if slot is not None:
                        missing_slot = "player_near" if slot == "player_far" else "player_far"
                        if not self._is_candidate_plausible_for_slot(
                            actor=actor,
                            slot=missing_slot,
                            frame_height=frame_height,
                            frame_width=frame_width,
                        ):
                            continue
                selected.append(actor)
                selected_ids.add(actor.track_id)

        return selected[:2]

    def _select_players_by_previous_track_id(self, players: list[SceneActor]) -> list[SceneActor]:
        selected: list[SceneActor] = []
        used: set[str] = set()
        for slot in ("player_far", "player_near"):
            previous_track_id = self._last_player_source_track_ids.get(slot)
            if previous_track_id is None:
                continue
            for actor in players:
                if actor.track_id == previous_track_id and actor.track_id not in used:
                    selected.append(actor)
                    used.add(actor.track_id)
                    break
        return selected

    def _slot_for_source_track(self, source_track_id: str) -> str | None:
        for slot, previous_track_id in self._last_player_source_track_ids.items():
            if source_track_id == previous_track_id:
                return slot
        return None

    def _is_candidate_plausible_for_slot(
        self,
        actor: SceneActor,
        slot: str,
        frame_height: int,
        frame_width: int,
    ) -> bool:
        previous_bbox = self._last_player_bboxes.get(slot)
        if previous_bbox is None:
            return True

        prev_area = max(1.0, _bbox_area(previous_bbox))
        actor_area = max(1.0, _bbox_area(actor.bbox))
        area_ratio = max(prev_area / actor_area, actor_area / prev_area)
        if area_ratio > 4.0:
            return False

        px, py = _bbox_center(previous_bbox)
        cx, cy = _bbox_center(actor.bbox)
        distance = float(np.hypot(cx - px, cy - py))
        previous_h = max(1.0, float(previous_bbox[3] - previous_bbox[1]))
        frame_diag = max(1.0, float(np.hypot(frame_width, frame_height)))
        max_distance = max(float(frame_height) * 0.35, previous_h * 2.0, frame_diag * 0.12)
        return distance <= max_distance

    def _distance_to_anchor(self, actor: SceneActor, anchor_x: float, anchor_y: float) -> float:
        cx, cy = _bbox_center(actor.bbox)
        return float(np.hypot(cx - anchor_x, cy - anchor_y))

    def _select_rackets(self, rackets: list[SceneActor], selected_players: list[SceneActor]) -> list[SceneActor]:
        if len(rackets) <= 2:
            return rackets[:2]

        scored: list[tuple[float, SceneActor]] = []
        for racket in rackets:
            area_score = _bbox_area(racket.bbox)
            proximity = 0.0
            rx, ry = ((racket.bbox[0] + racket.bbox[2]) * 0.5, (racket.bbox[1] + racket.bbox[3]) * 0.5)
            for player in selected_players:
                px, py = ((player.bbox[0] + player.bbox[2]) * 0.5, (player.bbox[1] + player.bbox[3]) * 0.5)
                distance = float(np.hypot(rx - px, ry - py))
                proximity += 1.0 / max(distance, 1.0)
            score = area_score * 0.02 + proximity * 1000.0
            scored.append((score, racket))

        scored_sorted = sorted(scored, key=lambda p: p[0], reverse=True)
        return [actor for _, actor in scored_sorted[:2]]

    def _stabilize_players(self, selected_players: list[SceneActor], frame_height: int) -> list[SceneActor]:
        if not selected_players:
            return []

        if len(selected_players) == 1:
            player = selected_players[0]
            slot = self._select_single_player_slot(player=player, frame_height=frame_height)
            stabilized = player.model_copy(update={"track_id": slot})
            stabilized, accepted = self._apply_temporal_bbox_guard(
                slot=slot,
                candidate=stabilized,
                frame_height=frame_height,
            )
            self._last_player_bboxes[slot] = list(stabilized.bbox)
            if accepted:
                self._last_player_source_track_ids[slot] = player.track_id

            other_slot = "player_near" if slot == "player_far" else "player_far"
            previous_other_bbox = self._last_player_bboxes.get(other_slot)
            if previous_other_bbox is None:
                return [stabilized]

            held_other = SceneActor(
                track_id=other_slot,
                class_name="player",
                bbox=list(previous_other_bbox),
            )
            return [stabilized, held_other]

        candidate_a = selected_players[0]
        candidate_b = selected_players[1]

        assign_a = self._assignment_cost(
            far_player=candidate_a,
            near_player=candidate_b,
            frame_height=frame_height,
        )
        assign_b = self._assignment_cost(
            far_player=candidate_b,
            near_player=candidate_a,
            frame_height=frame_height,
        )

        if assign_a <= assign_b:
            far_player = candidate_a
            near_player = candidate_b
        else:
            far_player = candidate_b
            near_player = candidate_a

        far_stable = far_player.model_copy(update={"track_id": "player_far"})
        near_stable = near_player.model_copy(update={"track_id": "player_near"})

        far_stable, far_accepted = self._apply_temporal_bbox_guard(
            slot="player_far",
            candidate=far_stable,
            frame_height=frame_height,
        )
        near_stable, near_accepted = self._apply_temporal_bbox_guard(
            slot="player_near",
            candidate=near_stable,
            frame_height=frame_height,
        )

        self._last_player_bboxes["player_far"] = list(far_stable.bbox)
        self._last_player_bboxes["player_near"] = list(near_stable.bbox)
        if far_accepted:
            self._last_player_source_track_ids["player_far"] = far_player.track_id
        if near_accepted:
            self._last_player_source_track_ids["player_near"] = near_player.track_id
        return [far_stable, near_stable]

    def _apply_temporal_bbox_guard(
        self,
        slot: str,
        candidate: SceneActor,
        frame_height: int,
    ) -> tuple[SceneActor, bool]:
        previous_bbox = self._last_player_bboxes.get(slot)
        if previous_bbox is None:
            return candidate, True

        prev_cx, prev_cy = _bbox_center(previous_bbox)
        cand_cx, cand_cy = _bbox_center(candidate.bbox)
        movement = float(np.hypot(cand_cx - prev_cx, cand_cy - prev_cy))

        prev_h = max(1.0, float(previous_bbox[3] - previous_bbox[1]))
        max_step = max(float(frame_height) * 0.18, prev_h * 2.2)
        if movement <= max_step:
            return candidate, True

        guarded = candidate.model_copy(update={"bbox": list(previous_bbox)})
        return guarded, False

    def _assignment_cost(self, far_player: SceneActor, near_player: SceneActor, frame_height: int) -> float:
        far_cx, far_cy = _bbox_center(far_player.bbox)
        near_cx, near_cy = _bbox_center(near_player.bbox)

        penalty = 0.0
        if far_cy > near_cy:
            penalty += float(frame_height)

        far_prev = self._last_player_bboxes.get("player_far")
        near_prev = self._last_player_bboxes.get("player_near")

        if far_prev is None:
            far_anchor_y = float(frame_height) * 0.30
            far_cost = abs(far_cy - far_anchor_y)
        else:
            prev_cx, prev_cy = _bbox_center(far_prev)
            far_cost = float(np.hypot(far_cx - prev_cx, far_cy - prev_cy))

        if near_prev is None:
            near_anchor_y = float(frame_height) * 0.70
            near_cost = abs(near_cy - near_anchor_y)
        else:
            prev_cx, prev_cy = _bbox_center(near_prev)
            near_cost = float(np.hypot(near_cx - prev_cx, near_cy - prev_cy))

        return far_cost + near_cost + penalty

    def _select_single_player_slot(self, player: SceneActor, frame_height: int) -> str:
        slot_from_source_track = self._slot_for_source_track(player.track_id)
        if slot_from_source_track is not None:
            return slot_from_source_track

        cx, cy = ((player.bbox[0] + player.bbox[2]) * 0.5, (player.bbox[1] + player.bbox[3]) * 0.5)
        far_prev = self._last_player_bboxes.get("player_far")
        near_prev = self._last_player_bboxes.get("player_near")

        if far_prev is not None and near_prev is not None:
            far_cx, far_cy = _bbox_center(far_prev)
            near_cx, near_cy = _bbox_center(near_prev)
            dist_far = float(np.hypot(cx - far_cx, cy - far_cy))
            dist_near = float(np.hypot(cx - near_cx, cy - near_cy))
            return "player_far" if dist_far <= dist_near else "player_near"

        return "player_far" if cy < (float(frame_height) * 0.5) else "player_near"

    def _stabilize_rackets(self, selected_rackets: list[SceneActor], stabilized_players: list[SceneActor]) -> list[SceneActor]:
        if not selected_rackets:
            return []

        player_by_slot = {player.track_id: player for player in stabilized_players}
        used: set[int] = set()
        stabilized: list[SceneActor] = []

        slot_pairs = (("player_far", "racket_far"), ("player_near", "racket_near"))
        for player_slot, racket_slot in slot_pairs:
            player = player_by_slot.get(player_slot)
            if player is None:
                continue

            px, py = ((player.bbox[0] + player.bbox[2]) * 0.5, (player.bbox[1] + player.bbox[3]) * 0.5)
            best_idx: int | None = None
            best_distance = float("inf")
            for idx, racket in enumerate(selected_rackets):
                if idx in used:
                    continue
                rx, ry = ((racket.bbox[0] + racket.bbox[2]) * 0.5, (racket.bbox[1] + racket.bbox[3]) * 0.5)
                distance = float(np.hypot(rx - px, ry - py))
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx

            if best_idx is None:
                continue

            used.add(best_idx)
            stabilized.append(selected_rackets[best_idx].model_copy(update={"track_id": racket_slot}))

        for idx, racket in enumerate(selected_rackets):
            if idx in used or len(stabilized) >= 2:
                continue
            stabilized.append(racket.model_copy(update={"track_id": f"racket_extra_{idx}"}))

        return stabilized[:2]
