import pytest

from experiments.tennis.parallel_curves import compare


def test_log_rate_integration_recovers_double_rate_and_rejects_no_overlap():
    anchor = [(10., -.3), (20., -.2), (40., -.1)]
    result = compare([(20., -.3), (40., -.2), (80., -.1)], anchor)
    assert result['bd_rate_percent'] == pytest.approx(100.)
    assert compare([(100., -.09), (200., -.08)], anchor)['bd_rate_percent'] is None
