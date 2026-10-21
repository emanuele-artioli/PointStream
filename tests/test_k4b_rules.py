import json

import numpy as np
import pytest

from experiments.tennis.k4b_rules import RULES, choose, correct, features, merge_prompt_tracks, validate_marks
from experiments.visor.b1 import file_sha256


def row(t, x, y=0, height=10, contact=False, keypoints=None):
    return dict(t=t, x=x, y=y, height=height, contact=contact, keypoints=keypoints)


def test_motion_has_no_gap_interpolation_and_tail_is_discounted():
    f = features([row(0, 0), row(1, 3, 4), row(8, 100), row(9, 110)], 10, 2)
    assert f["a"] == 1.5
    assert f["b"] == 3
    assert f["presence"] == .4
    with pytest.raises(ValueError):
        features([row(0, 0), row(0, 1)], 10, 2)


def test_contact_majority_is_strict_and_uses_all_clip_frames():
    f = features([row(t, t, contact=True) for t in range(5)], 10, 2)
    assert f["c"] == .5 and not f["racket_most"]
    assert features([row(t, t, contact=True) for t in range(6)], 10, 2)["racket_most"]


def test_keypoint_motion_requires_joint_confidence_on_both_frames():
    a = [[0, 0, .9], [100, 0, .1]]
    b = [[3, 4, .9], [200, 0, .9]]
    assert features([row(0, 0, keypoints=a), row(1, 0, keypoints=b)], 2, 1)["e"] == .5


def test_candidates_and_combination_do_not_use_annotation_input():
    tracks = {"person:1": dict(presence=1, a=10, b=2, c=.4, e=3, racket_most=False),
              "person:2": dict(presence=1, a=2, b=6, c=.8, e=5, racket_most=True),
              "person:3": dict(presence=1, a=1, b=7, c=.7, e=8, racket_most=True),
              "person:4": dict(presence=.4, a=100, b=100, c=1, e=100, racket_most=True)}
    picks = choose(tracks)
    assert list(picks) == RULES["tie_order"]
    assert picks == {"a": ["person:1", "person:2"], "b": ["person:3", "person:2"],
                     "c": ["person:2", "person:3"], "e": ["person:3", "person:2"], "a+c": ["person:2", "person:3"]}
    assert choose({"person:1": tracks["person:1"]})["c"] == []


def test_duplicate_prompt_aliases_are_one_to_one():
    counts = {"person:1": 10, "player:1": 10, "player:2": 10}
    pairs = {("person:1", "player:1"): dict(co_visible=10, agree=9, iou_sum=9),
             ("person:1", "player:2"): dict(co_visible=10, agree=8, iou_sum=8)}
    assert merge_prompt_tracks(counts, pairs) == {"person:1": "person:1", "player:1": "person:1", "player:2": "player:2"}


def test_matching_is_injective_and_one_player_ignores_other_pick():
    a = np.eye(4, dtype=bool)
    b = np.fliplr(a)
    assert correct([a, b], [b, a], 2, False)
    assert not correct([a], [a, a], 2, False)
    assert correct([a, b], [a], 1, False)
    assert correct([], [], 0, False)
    assert not correct([a], [], 0, False)
    assert not correct([a], [a], 2, True)


def test_mark_identity_and_confirmation_are_required(tmp_path):
    p = tmp_path / "instances.json"
    r = dict(selection_sha256="sel", frames=[dict(id="f", instances=[dict(id=1)])])
    p.write_text(json.dumps(r))
    m = dict(schema=1, selection_sha256="sel", result_sha256=file_sha256(p),
             marks={"f": dict(selected=[1], visible=1, missing_instance=False, confirmed=True)})
    assert validate_marks(m, r, p) == m["marks"]
    m["marks"]["f"]["confirmed"] = False
    with pytest.raises(ValueError):
        validate_marks(m, r, p)
