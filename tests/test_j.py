import math

import pytest

from experiments.visor import j


def test_hull_drops_points_above_the_chord_in_log_rate():
    points = [(1.0, 4.0), (2.0, 3.9), (4.0, 1.9), (8.0, 1.0)]
    assert j.hull(points) == [(1.0, 4.0), (4.0, 1.9), (8.0, 1.0)]
    assert j.hull([(1.0, 4.0), (4.0, 2.0), (8.0, 1.0)]) == [(1.0, 4.0), (8.0, 1.0)]  # collinear: dropped


def test_hull_keeps_the_lower_value_at_equal_rates_and_skips_non_finite():
    assert j.hull([(2.0, 3.0), (2.0, 1.0), (4.0, math.nan), (0.0, 0.5)]) == [(2.0, 1.0)]


def test_read_is_linear_in_log_rate_and_clamped_with_flags():
    points = [(1.0, 4.0), (4.0, 2.0)]
    assert j.read(points, 2.0)["value"] == pytest.approx(3.0)
    assert j.read(points, 2.0)["flag"] is None
    assert j.read(points, 0.5) == {"value": 4.0, "flag": "below the anchor's range"}
    assert j.read(points, 8.0) == {"value": 2.0, "flag": "above the anchor's range"}


def test_a_subset_of_the_anchors_decides_only_failures_inside_its_range():
    inside = {"value": 1.0, "flag": None}
    outside = {"value": 1.0, "flag": "below the anchor's range"}
    assert j.content(2.0, inside, 1.0, complete=False) is False  # above max(1.0, 1.5)
    assert j.content(0.9, inside, 1.0, complete=False) is True  # under the keypoints' floor: no anchor needed
    assert j.content(1.4, {"value": 1.6, "flag": None}, 0.1, complete=False) is None  # a subset pass decides nothing
    assert j.content(2.0, outside, 1.0, complete=False) is None
    assert j.content(2.0, outside, 1.0, complete=True) is False
    assert j.realism(0.02, {"value": 0.01, "flag": None}, complete=False) is False
    assert j.realism(0.005, {"value": 0.01, "flag": None}, complete=True) is True


def test_time_is_the_or_of_flicker_against_the_anchor_and_energy_against_the_source():
    anchor = {"value": 3.0, "flag": None}
    assert j.over_time(5.0, anchor, 10.0, 9.0, complete=False) is True  # energy branch: 10 <= 11.25
    assert j.over_time(5.0, anchor, 20.0, 9.0, complete=False) is False  # both fail, inside the range
    assert j.over_time(2.0, anchor, 20.0, 9.0, complete=False) is None  # subset flicker pass decides nothing
    assert j.over_time(2.0, anchor, 20.0, 9.0, complete=True) is True
    assert j.over_time(5.0, anchor, None, 9.0, complete=True) is None  # energy unscored: cannot fail the OR


def test_decide_stops_once_the_undecided_items_cannot_change_the_result():
    assert j.decide({"a": False, "b": False, "c": None, "d": None})["decision"] == "fail"
    assert j.decide({"a": True, "b": True, "c": True, "d": None})["decision"] == "pass"
    assert j.decide({"a": True, "b": False, "c": None, "d": None})["decision"] == "undecided"
    assert j.decide({str(i): i < 6 for i in range(8)})["decision"] == "pass"  # 6 of 8


def test_h3e_item_fails_when_every_point_fails_content():
    item = {"keypoints_pose": 0.01, "judged_hands": 3,
            "pixel_points": [{"name": "svt@63/8", "kbps": 1.0, "pose": 0.03}, {"name": "svt@63/4", "kbps": 4.0, "pose": 0.01},
                             {"name": "reference@0/4", "kbps": 0.5, "pose": 0.02}],
            "v1": [{"name": "@0/4", "kbps": 2.0, "pose": 0.025}, {"name": "@0/2", "kbps": 3.0, "pose": 0.025}]}
    row = j.h3e_item(item)
    assert row["status"] is False
    assert [p["checks"]["content"] for p in row["points"]] == [False, False]
