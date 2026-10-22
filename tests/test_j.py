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


def test_player_pose_error_is_over_confident_source_keypoints_and_a_miss_counts_as_one_box():
    np = pytest.importorskip("numpy")

    ref = np.zeros((17, 3))
    ref[:, 2] = 0.9
    ref[0, 2] = 0.1  # not confident: ignored
    pred = ref.copy()
    pred[:, 0] += 10.0
    pred[0, 0] += 1000.0
    assert j.pose_error(pred, ref, 100.0) == pytest.approx(0.1)
    assert j.pose_error(None, ref, 100.0) == j.MISSED
    ref[:, 2] = 0.0
    assert j.pose_error(pred, ref, 100.0) is None


def test_residual_energies_of_a_frozen_output_and_of_the_source_itself():
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(0)
    prev = rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)
    cur = rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)  # no motion a flow can explain: a large residual
    region = np.zeros((64, 64), bool)
    region[16:48, 16:48] = True
    f, e_out, e_src = j.residual_energies(cur, prev, cur, prev, region, region)
    assert f == pytest.approx(0.0)  # the source against itself
    assert e_out == pytest.approx(e_src) and e_src > 10
    brighter = cur + np.uint8(20)  # the output jumps by 20 luma levels at t: flicker is that jump
    f, e_out, e_src = j.residual_energies(brighter, prev, cur, prev, region, region)
    assert f == pytest.approx(20.0, abs=1e-3)
    assert j.residual_energies(cur, prev, cur, prev, np.zeros_like(region), region) is None


def test_j2_tracks_split_by_side_and_skipped_frames():
    np = pytest.importorskip("numpy")
    from experiments.visor import j2

    t = np.array([0, 1, 2, 4, 5, 0, 1])
    right = np.array([0, 0, 0, 0, 0, 1, 1], bool)
    assert [list(x) for x in j2.tracks(t, right)] == [[0, 1, 2], [3, 4], [5, 6]]


def test_j2_smooth_keeps_constants_and_lines_and_damps_alternation():
    np = pytest.importorskip("numpy")
    from experiments.visor import j2

    const = np.full((9, 21, 2), 3.0)
    assert np.allclose(j2.smooth(const, 2), const)
    line = np.arange(9, dtype=float)[:, None, None] * np.ones((1, 21, 2))
    assert np.allclose(j2.smooth(line, 1)[4], line[4], atol=1e-9)  # the centre sees symmetric weights: a line is kept
    zigzag = ((-1.0) ** np.arange(9))[:, None, None] * np.ones((1, 21, 2))
    assert np.abs(j2.smooth(zigzag, 2)[4]).max() < 0.1
    assert np.array_equal(j2.smooth(zigzag, 0), zigzag)
