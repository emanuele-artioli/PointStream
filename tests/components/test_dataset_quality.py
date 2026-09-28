from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.shared.dataset_quality import (
    QualityDecision,
    filter_player_candidates,
    filter_racket_candidates,
)


def _pose(*, wrist: tuple[float, float] = (100.0, 100.0), confidence: float = 0.9):
    values = np.zeros((133, 3), dtype=np.float32)
    values[:8, 2] = confidence
    values[9, :2] = wrist
    values[9, 2] = confidence
    values[10, :2] = wrist
    values[10, 2] = confidence
    visibility = np.zeros(133, dtype=np.uint8)
    visibility[:8] = 2
    return SimpleNamespace(
        values=values,
        visibility=visibility,
        schema=SimpleNamespace(name="coco-wholebody-133"),
    )


def _person_mask(x: int, *, height: int = 90, width: int = 28) -> np.ndarray:
    mask = np.zeros((256, 384), dtype=np.uint8)
    mask[40 : 40 + height, x : x + width] = 1
    return mask


def test_player_filter_caps_per_frame_and_ranks_camera_relative_motion() -> None:
    frames = np.zeros((8, 256, 384, 3), dtype=np.uint8)
    candidates: dict[tuple[int, str], SimpleNamespace] = {}
    poses: dict[int, list[tuple[str, object, None]]] = {}
    for frame_index in range(8):
        poses[frame_index] = []
        for object_id, x in (("court-player-a", 30 + frame_index * 8), ("court-player-b", 145 - frame_index * 6), ("static-spectator", 300)):
            candidates[(frame_index, object_id)] = SimpleNamespace(
                mask=_person_mask(x), score=0.92
            )
            poses[frame_index].append((object_id, _pose(), None))

    decisions, tracks = filter_player_candidates(candidates, poses, frames)

    for frame_index in range(8):
        accepted = [
            object_id
            for (candidate_frame, object_id), decision in decisions.items()
            if candidate_frame == frame_index and decision.eligible
        ]
        assert len(accepted) <= 2
        assert "court-player-a" in accepted
        assert "court-player-b" in accepted
        assert "static-spectator" not in accepted
    assert tracks["court-player-a"]["camera_compensated_motion"] > tracks["static-spectator"]["camera_compensated_motion"]


def test_player_filter_quarantines_missing_confidence_and_pose() -> None:
    frames = np.zeros((3, 256, 384, 3), dtype=np.uint8)
    candidates = {
        (0, "unknown-score"): SimpleNamespace(mask=_person_mask(80), score=None),
        (1, "unknown-score"): SimpleNamespace(mask=_person_mask(82), score=None),
        (0, "no-pose"): SimpleNamespace(mask=_person_mask(180), score=0.95),
        (1, "no-pose"): SimpleNamespace(mask=_person_mask(182), score=0.95),
    }
    decisions, _tracks = filter_player_candidates(candidates, {}, frames)
    assert "sam_confidence_missing" in decisions[(0, "unknown-score")].reasons
    assert "body_pose_support_too_low" in decisions[(0, "no-pose")].reasons


def test_racket_filter_keeps_small_wrist_linked_mask_and_rejects_net() -> None:
    player = SimpleNamespace(mask=_person_mask(60, height=150, width=55), score=0.95)
    players = {(frame, "player"): player for frame in (0, 1)}
    player_decisions = {
        key: QualityDecision(True, 0.9, (), {}) for key in players
    }
    poses = {
        frame: [("player", _pose(wrist=(112.0, 135.0)), None)] for frame in (0, 1)
    }
    rackets: dict[tuple[int, str], SimpleNamespace] = {}
    associations: dict[tuple[int, str], SimpleNamespace] = {}
    for frame in (0, 1):
        racket = np.zeros((256, 384), dtype=np.uint8)
        racket[125:145, 108:138] = 1
        arm = np.zeros_like(racket)
        arm[125:145, 75:105] = 1
        net = np.zeros_like(racket)
        net[118:138, 10:375] = 1
        rackets[(frame, "racket")] = SimpleNamespace(mask=racket, score=0.95)
        rackets[(frame, "arm-drift")] = SimpleNamespace(mask=arm, score=0.95)
        rackets[(frame, "net")]=SimpleNamespace(mask=net, score=0.99)
        associations[(frame, "racket")] = SimpleNamespace(associated_player_id="player", associated_wrist="right_wrist")
        associations[(frame, "arm-drift")] = SimpleNamespace(associated_player_id="player", associated_wrist="right_wrist")
        associations[(frame, "net")] = SimpleNamespace(associated_player_id="player", associated_wrist="right_wrist")

    decisions, _tracks = filter_racket_candidates(
        rackets,
        players,
        player_decisions,
        associations,
        poses,
        (256, 384),
    )

    assert decisions[(0, "racket")].eligible
    assert not decisions[(0, "arm-drift")].eligible
    assert "racket_mask_mostly_overlaps_player_mask" in decisions[(0, "arm-drift")].reasons
    assert not decisions[(0, "net")].eligible
    assert "racket_area_not_smaller_than_player" in decisions[(0, "net")].reasons
