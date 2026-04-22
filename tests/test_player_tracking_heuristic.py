from __future__ import annotations

from src.encoder.actor_components import StandardTennisHeuristic
from src.shared.schemas import FrameState, SceneActor


def _player(track_id: str, bbox: list[float]) -> SceneActor:
    return SceneActor(track_id=track_id, class_name="player", bbox=bbox)


def test_heuristic_prefers_temporal_player_tracks_over_distractor() -> None:
    heuristic = StandardTennisHeuristic()

    frame0 = FrameState(
        frame_id=0,
        actors=[
            _player("far_track", [520.0, 120.0, 620.0, 260.0]),
            _player("near_track", [500.0, 410.0, 700.0, 710.0]),
            _player("kid_track", [780.0, 140.0, 940.0, 380.0]),
        ],
    )
    selected0 = heuristic.select(frame0, frame_shape=(720, 1280))

    players0 = [actor for actor in selected0.actors if actor.class_name == "player"]
    assert len(players0) == 2

    frame1 = FrameState(
        frame_id=1,
        actors=[
            _player("kid_track", [700.0, 150.0, 980.0, 430.0]),
            _player("far_track", [525.0, 122.0, 625.0, 262.0]),
            _player("near_track", [505.0, 412.0, 705.0, 712.0]),
        ],
    )
    selected1 = heuristic.select(frame1, frame_shape=(720, 1280))

    players1 = {actor.track_id: actor for actor in selected1.actors if actor.class_name == "player"}
    assert "player_far" in players1
    assert "player_near" in players1

    far_bbox = players1["player_far"].bbox
    near_bbox = players1["player_near"].bbox

    assert far_bbox[0] < 650.0
    assert far_bbox[2] < 700.0
    assert near_bbox[1] > 350.0

    frame2 = FrameState(
        frame_id=2,
        actors=[
            _player("kid_track", [690.0, 155.0, 980.0, 440.0]),
            _player("near_track", [510.0, 416.0, 710.0, 716.0]),
        ],
    )
    selected2 = heuristic.select(frame2, frame_shape=(720, 1280))
    players2 = {actor.track_id: actor for actor in selected2.actors if actor.class_name == "player"}

    assert "player_far" in players2
    assert "player_near" in players2

    # The missing far player should be held from temporal history, not replaced by the kid track.
    held_far_bbox = players2["player_far"].bbox
    assert held_far_bbox[0] < 650.0
    assert held_far_bbox[2] < 700.0
