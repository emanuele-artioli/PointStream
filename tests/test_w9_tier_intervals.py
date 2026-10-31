import pytest

from experiments.tennis import w9_tier_intervals as w


def _window():
    return {'fixed_kbps': 5.0, 'one_time_bytes': 1000, 'hybrid': {'lpips': .4, 'dists': .4},
            'required': ['svt'], 'anchors': {'svt': [
                {'rate_kbps': 10.0, 'foreground_lpips': .6, 'foreground_dists': .6},
                {'rate_kbps': 40.0, 'foreground_lpips': .2, 'foreground_dists': .2}]}}


def test_interval_pass_fail_and_tie_boundary_stay_distinct():
    result = w.intervals([_window(), _window(), _window()], max_depth=14)
    assert result['coverage'] == pytest.approx([8 / 35, 8 / 5])
    rows = result['intervals']
    assert rows[0]['verdict'] == 'fail' and rows[-1]['verdict'] == 'pass'
    uncertain = [r for r in rows if r['verdict'] == 'unresolved']
    assert uncertain and uncertain[0]['seconds'][0] <= 8 / 15 <= uncertain[-1]['seconds'][1]


def test_missing_anchor_and_unmeasured_rates_are_unresolved():
    a = _window()
    a['required'].append('dcvc')
    assert w.intervals([a, a])['coverage'] is None
    with pytest.raises(ValueError, match='outside'):
        w.curve_range([(10, .6), (40, .2)], 9, 20)


def test_nonmonotone_quality_extrema_and_duplicate_rate():
    assert w.curve_range([(10, .5), (20, .9), (40, .4)], 10, 40) == pytest.approx((.4, .9))
    with pytest.raises(ValueError, match='duplicate'):
        w.curve_range([(10, .5), (10, .3)], 10, 10)


def test_joint_rule_does_not_pool_tiers():
    wins = [_window(), _window(), _window()]
    losses = [_window(), _window(), _window()]
    for window in losses:
        window['hybrid'] = {'lpips': .9, 'dists': .9}
    result = w.joint_intervals([wins, losses], max_depth=8)
    assert result['coverage'] is not None
    assert all(cell['verdict'] == 'fail' for cell in result['intervals'])


@pytest.mark.parametrize('n', [3, 6])
def test_confidence_encloses_original_rule_at_box_corners_and_midpoint(n):
    from itertools import product

    from experiments.tennis.w7_analysis import t_interval

    boxes = [(-.04 + i * .012, .09 + i * .017) for i in range(n)]
    lower, upper = w.confidence_bounds(boxes)
    for values in [*product(*boxes), tuple((a + b) / 2 for a, b in boxes)]:
        original = t_interval(list(values))['lo']
        assert lower - 1e-12 <= original <= upper + 1e-12
    collapsed = [(x, x) for x in (.02 + i * .031 for i in range(n))]
    bounds = w.confidence_bounds(collapsed)
    assert bounds == pytest.approx((t_interval([a for a, _ in collapsed])['lo'],) * 2)


def test_invalid_hybrid_gaps_and_required_family_are_rejected():
    a = _window()
    a['required'] = []
    with pytest.raises(ValueError, match='required families'):
        w.coverage([a])
    a = _window()
    a['hybrid']['lpips'] = float('nan')
    with pytest.raises(ValueError, match='finite hybrid'):
        w.coverage([a])
    with pytest.raises(ValueError, match='invalid gap'):
        w.confidence_bounds([(0, 1), (float('nan'), 1)])


def test_roundoff_at_measured_endpoint_never_extrapolates():
    import math

    assert w.curve_range([(10, .6), (40, .2)], 10, math.nextafter(40, math.inf))[1] == pytest.approx(.6)
    with pytest.raises(ValueError, match='outside'):
        w.curve_range([(10, .6), (40, .2)], 10, 40 + 1e-8)


def test_minimum_anchor_rate_is_not_a_w7_bracket():
    a = _window()
    low_u, high_u = w.coverage([a, a])
    assert a['fixed_kbps'] + a['one_time_bytes'] * 8 / 1000 * low_u > 10
    assert a['fixed_kbps'] + a['one_time_bytes'] * 8 / 1000 * high_u <= 40


def test_repaired_w7_report_refuses_incomplete_audit_set(tmp_path):
    from experiments.tennis.w9_repair_w7 import compute

    with pytest.raises(ValueError, match='all six audited'):
        compute(tmp_path, {'nad000': {}})
