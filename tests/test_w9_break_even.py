import math

import pytest

from experiments.tennis.w9_break_even import intervals


def test_crossing_and_unmeasured_ranges():
    result = intervals([(10, 0.6), (40, 0.2)], 0.4, 5, 1000)
    assert [r['outcome'] for r in result] == ['win', 'loss']
    assert result[0]['rate_kbps'][1] == pytest.approx(20)
    assert result[0]['length_seconds'] == pytest.approx([8 / 15, 8 / 5])
    assert result[1]['length_seconds'] == pytest.approx([8 / 35, 8 / 15])


def test_nonmonotonic_and_infinite_length():
    result = intervals([(1, .5), (2, .2), (4, .6)], .4, 1, 1000)
    assert [r['outcome'] for r in result] == ['win', 'loss', 'loss', 'win']
    assert result[0]['length_seconds'][1] is None


def test_invalid_inputs():
    for points in ([(1, 2)], [(1, 2), (1, 3)], [(1, 2), (math.nan, 3)]):
        with pytest.raises(ValueError):
            intervals(points, .4, 0, 1000)
