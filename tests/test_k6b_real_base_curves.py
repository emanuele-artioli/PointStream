"""The fixed central point is bracketed or explicitly unresolved."""
from experiments.tennis.k6b_real_base_judge import compare_curves


def test_curve_uses_central_point_and_no_rate_extrapolation():
    row = {'rates': [{'scale': 8, 'base_kbps': 10.0, 'total_kbps': 11.0},
                     {'scale': 4, 'base_kbps': 20.0, 'total_kbps': 21.0},
                     {'scale': 2, 'base_kbps': 30.0, 'total_kbps': 31.0}]}
    row['means'] = {m: [0.5, 0.4, 0.3, 0.1, 0.1, 0.05] for m in
                    ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists')}
    out = compare_curves(row)
    assert all(x['scale4']['won'] is True and x['scale4']['rate'] == 21.0 for x in out.values())
    row['rates'][1]['total_kbps'] = 31.0
    out = compare_curves(row)
    assert all(x['scale4']['won'] is None for x in out.values())
