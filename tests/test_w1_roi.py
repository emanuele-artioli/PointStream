"""W1: the ROI map, the rate-only CRF search and the frozen decision rule."""
import math

import numpy as np
import pytest

from experiments.tennis import w1_roi as w
from experiments.tennis.parallel_baselines import roi_text as old_roi_text


def synthetic(crf_to_rate):
    """Run the search against a rate function and return (status, measured rates, encodes)."""
    known: dict[int, float] = {}
    for _ in range(40):
        status, crf = w.next_crf(known, w.TARGET_KBPS, 67)
        if crf is None:
            return status, known
        assert crf not in known
        known[crf] = crf_to_rate(crf)
    raise AssertionError('search did not terminate')


def test_rate_matches_the_existing_anchor_rates():
    assert w.rate_kbps(20557) == pytest.approx(17.130833333333335)
    assert w.rate_kbps(40422) == pytest.approx(33.685)


def test_roi_text_marks_the_same_blocks_as_the_older_comparator():
    region = np.zeros((2, 128, 128), bool)
    region[0, 0, 0] = True
    region[1, 127, 127] = True
    assert w.roi_text(region, 128, 128, -20).splitlines() == ['0 -20 0 0 0', '1 0 0 0 -20']
    # the older map wrote 0 inside and 32 outside; the touched blocks are the same
    old = old_roi_text(region, 128, 128).split('\n')[:-1]
    new = w.roi_text(region, 128, 128, -1).split('\n')[:-1]
    for a, b in zip(old, new):
        assert [x == '0' for x in a.split()[1:]] == [x == '-1' for x in b.split()[1:]]


def test_roi_text_block_grid_follows_the_coded_size():
    region = np.zeros((1, 540, 960), bool)
    region[0, 300, 500] = True
    line = w.roi_text(region, 1920, 1080, -40).split()
    assert len(line) == 1 + 17 * 30  # ceil(1080 / 64) rows, 30 columns
    assert line.count('-40') == 1
    line = w.roi_text(region, 960, 544, -40).split()
    assert len(line) == 1 + 9 * 15
    with pytest.raises(ValueError):
        w.roi_text(region, 960, 544, -64)
    with pytest.raises(ValueError):
        w.roi_text(region, 961, 544, -20)


def test_search_brackets_with_adjacent_crfs_and_one_neighbour():
    status, known = synthetic(lambda c: 33.685 * math.exp(-0.075 * (c - 63)))
    assert status == 'complete'
    chosen = w.select_crfs(known, w.TARGET_KBPS)
    assert chosen is not None and len(chosen) == 3
    a = chosen[0] if known[chosen[0]] >= w.TARGET_KBPS > known[chosen[1]] else chosen[1]
    assert known[a] >= w.TARGET_KBPS > known[a + 1]
    assert len(known) <= 6


def test_search_walks_down_when_the_first_probe_is_far_too_cheap():
    status, known = synthetic(lambda c: 22.74 * math.exp(-0.1 * (c - 40)))
    assert status == 'complete' and min(known) < 40 <= max(known) + 1


def test_search_reports_a_rate_that_stays_above_the_target_at_crf_70():
    status, known = synthetic(lambda c: 80.0 * math.exp(-0.03 * (c - 63)))
    assert status == 'unbracketed_high'
    assert max(known) == w.CRF_MAX and w.select_crfs(known, w.TARGET_KBPS) is None


def test_search_reports_a_rate_below_the_target_at_crf_1():
    status, _ = synthetic(lambda c: 5.0)
    assert status == 'unbracketed_low'


def test_search_survives_a_non_monotone_step():
    rates = {66: 30.0, 67: 24.0, 68: 25.0, 69: 20.0, 70: 18.0}
    status, known = synthetic(lambda c: rates.get(c, 40.0 if c < 66 else 10.0))
    assert status == 'complete'
    assert w.select_crfs(known, w.TARGET_KBPS) is not None


def test_selection_keeps_the_neighbour_inside_the_encoder_range():
    known = {69: 24.0, 70: 21.0, 68: 27.0}
    assert w.next_crf(known, w.TARGET_KBPS, 67) == ('complete', None)
    assert w.select_crfs(known, w.TARGET_KBPS) == [68, 69, 70]


def test_interpolation_is_linear_in_log_rate_and_never_extrapolates():
    low, high = (10.0, 0.5), (40.0, 0.1)
    assert w.interpolate_log_rate(20.0, low, high) == pytest.approx(0.3)
    assert w.interpolate_log_rate(10.0, low, high) == pytest.approx(0.5)
    points = [{'crf': 1, 'rate_kbps': 30.0, 'whole_lpips': 0.2, 'foreground_lpips': 0.5, 'whole_dists': 0.1, 'foreground_dists': 0.3}]
    assert w.policy_at_target(points) is None  # one side only: unresolved
    points.append({**points[0], 'crf': 2, 'rate_kbps': 20.0, 'whole_lpips': 0.3, 'foreground_lpips': 0.6})
    value = w.policy_at_target(points)
    assert value is not None and value['crf'] == [2, 1]
    weight = (math.log(w.TARGET_KBPS) - math.log(20.0)) / (math.log(30.0) - math.log(20.0))
    assert value['whole_lpips'] == pytest.approx(0.3 + weight * (0.2 - 0.3))


def test_bracketing_uses_the_nearest_points_on_each_side():
    base = {'whole_lpips': 0, 'foreground_lpips': 0, 'whole_dists': 0, 'foreground_dists': 0}
    points = [{'crf': c, 'rate_kbps': r, **base} for c, r in ((1, 15.0), (2, 21.0), (3, 24.0), (4, 31.0))]
    assert [p['crf'] for p in w.bracketing_pair(points)] == [2, 3]


def policy(whole_l, fg_l, whole_d=1.0, fg_d=1.0):
    return {'whole_lpips': whole_l, 'foreground_lpips': fg_l, 'whole_dists': whole_d, 'foreground_dists': fg_d}


def test_dominance_uses_the_margin_on_both_axes():
    h = w.HYBRID['lpips']
    assert w.dominates(policy(h['whole'] + 0.001, h['foreground'] + 0.001), 'lpips')
    assert not w.dominates(policy(h['whole'] + 0.0011, h['foreground']), 'lpips')
    assert not w.dominates(policy(h['whole'], h['foreground'] + 0.0011), 'lpips')


def test_outcome_a_when_no_policy_dominates_and_unresolved_never_does():
    # envelope-like: good whole, bad foreground; and the reverse
    policies = {'s2_o0': policy(0.234, 0.61, 0.165, 0.45), 's2_o40': policy(0.31, 0.30, 0.2, 0.2), 's1_o0': None}
    result = w.decide(policies)
    assert result['outcome'] == 'A' and result['unresolved'] == ['s1_o0']
    assert result['dominating'] == {'lpips': [], 'dists': []}


def test_outcome_b_when_one_policy_dominates_on_one_metric():
    policies = {'s2_o20': policy(0.27, 0.33, 0.3, 0.4), 's2_o0': policy(0.234, 0.61)}
    result = w.decide(policies)
    assert result['outcome'] == 'B'
    assert result['dominating'] == {'lpips': ['s2_o20'], 'dists': []}


def test_foreground_at_whole_interpolates_inside_the_range_only():
    curve = [(0.24, 0.60), (0.28, 0.40), (0.34, 0.30)]
    assert w.foreground_at_whole(curve, 0.26) == pytest.approx(0.50)
    assert w.foreground_at_whole(curve, 0.2758) == pytest.approx(0.60 + (0.2758 - 0.24) / 0.04 * (0.40 - 0.60))
    assert w.foreground_at_whole(curve, 0.23) is None
    assert w.foreground_at_whole(curve, 0.35) is None
    assert w.foreground_at_whole([(0.3, 0.3)], 0.3) is None


def test_best_foreground_is_reported_per_resolution_and_metric():
    policies = {'s2_o0': policy(0.24, 0.60, 0.15, 0.50), 's2_o20': policy(0.28, 0.40, 0.17, 0.35),
                's2_o40': policy(0.34, 0.30, 0.20, 0.25), 's1_o0': None}
    report = w.best_foreground(policies)
    assert report['s2_lpips']['foreground_at_hybrid_whole'] == pytest.approx(
        0.60 + (w.HYBRID['lpips']['whole'] - 0.24) / (0.28 - 0.24) * (0.40 - 0.60))
    assert report['s1_lpips']['foreground_at_hybrid_whole'] is None
    assert report['s2_dists']['hybrid_foreground'] == w.HYBRID['dists']['foreground']


def test_policy_names_are_stable():
    assert w.policy_name(2, 0) == 's2_o0' and w.policy_name(1, -40) == 's1_o40'
