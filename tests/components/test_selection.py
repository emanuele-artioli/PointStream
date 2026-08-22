"""Both selectors are registry-selectable; the heuristic is the tennis-specific one."""

from __future__ import annotations

import pytest

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.selection import REGISTRY as SELECTION
from src.components.selection.heuristic import HeuristicSelector
from src.components.selection.prompt import PromptSelector
from src.contracts.errors import UnknownBackendError


def _person(track_id: str, xyxy: list[float], score: float = 1.0) -> Detection:
    return Detection(
        class_name="person",
        bbox=Box.from_xyxy(xyxy),
        score=score,
        track_id=track_id,
    )


def test_heuristic_and_open_vocabulary_are_both_selectable() -> None:
    assert SELECTION.spec("heuristic").name == "heuristic"
    assert SELECTION.spec("open-vocabulary").name == "open-vocabulary"
    heuristic = SELECTION.build("heuristic")
    prompt = SELECTION.build("open-vocabulary")
    assert isinstance(heuristic, HeuristicSelector)
    assert isinstance(prompt, PromptSelector)


def test_unknown_selector_name_suggests_the_heuristic() -> None:
    with pytest.raises(UnknownBackendError, match="Did you mean"):
        SELECTION.spec("heuristik")


def test_heuristic_keeps_two_on_court_players_and_drops_a_ball_kid() -> None:
    selector = HeuristicSelector()
    frame_shape = (720, 1280)
    first = [
        _person("far_track", [520.0, 120.0, 620.0, 260.0]),
        _person("near_track", [500.0, 410.0, 700.0, 710.0]),
        _person("kid_track", [780.0, 140.0, 940.0, 380.0]),
    ]
    selected = selector.select(first, frame_shape)
    players = [item for item in selected if item.class_name == "player"]
    assert {item.track_id for item in players} == {"player_far", "player_near"}
    far = next(item for item in players if item.track_id == "player_far")
    assert far.bbox.x2 < 700.0


def test_heuristic_holds_a_missing_far_player_instead_of_promoting_the_kid() -> None:
    """A silent swap of player_far onto a ball kid would look like tracking working."""
    selector = HeuristicSelector()
    frame_shape = (720, 1280)
    selector.select(
        [
            _person("far_track", [520.0, 120.0, 620.0, 260.0]),
            _person("near_track", [500.0, 410.0, 700.0, 710.0]),
            _person("kid_track", [780.0, 140.0, 940.0, 380.0]),
        ],
        frame_shape,
    )
    selector.select(
        [
            _person("kid_track", [700.0, 150.0, 980.0, 430.0]),
            _person("far_track", [525.0, 122.0, 625.0, 262.0]),
            _person("near_track", [505.0, 412.0, 705.0, 712.0]),
        ],
        frame_shape,
    )
    selected = selector.select(
        [
            _person("kid_track", [690.0, 155.0, 980.0, 440.0]),
            _person("near_track", [510.0, 416.0, 710.0, 716.0]),
        ],
        frame_shape,
    )
    players = {item.track_id: item for item in selected if item.class_name == "player"}
    assert "player_far" in players
    assert players["player_far"].bbox.x2 < 700.0
    assert players["player_near"].bbox.y1 > 350.0


def test_prompt_selector_keeps_every_detection_matching_the_class_prompt() -> None:
    """Open-vocabulary selection is the control: no crowd filter, no two-player cap."""
    selector = PromptSelector(prompt="tennis player")
    detections = [
        _person("a", [10, 10, 40, 80]),
        _person("b", [50, 10, 80, 80]),
        _person("c", [90, 10, 120, 80]),
        Detection("tennis racket", Box.from_xyxy([1, 1, 8, 8])),
    ]
    selected = selector.select(detections, (100, 200))
    assert len(selected) == 3
    assert all(item.class_name == "player" for item in selected)


def test_prompt_selector_uses_the_domain_profile_when_no_prompt_is_set() -> None:
    selector = PromptSelector(domain="tennis")
    detections = [
        Detection("person", Box.from_xyxy([10, 10, 30, 50])),
        Detection("tennis racket", Box.from_xyxy([40, 10, 50, 30])),
        Detection("chair", Box.from_xyxy([60, 10, 80, 40])),
    ]
    selected = selector.select(detections)
    names = {item.class_name for item in selected}
    assert names == {"player", "racket"}
