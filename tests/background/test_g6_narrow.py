"""Narrow-model decisions must satisfy quality and latency together."""
from experiments.background.g6_narrow import decision


def test_fast_model_with_insufficient_gain_is_not_selected():
    result = decision(0.1, 0.05, 0.08, {"wall_ms_mean": 10, "wall_ms_p95": 12}, 60)
    assert result["speed_pass"]
    assert not result["quality_pass"]
    assert not result["passed"]


def test_good_mean_cannot_hide_slow_tail():
    result = decision(0.1, 0.05, 0.055, {"wall_ms_mean": 12, "wall_ms_p95": 19}, 60)
    assert result["quality_pass"]
    assert not result["speed_pass"]


def test_no_reference_gain_is_not_a_quality_pass():
    result = decision(0.1, 0.1, 0.09, {"wall_ms_mean": 10, "wall_ms_p95": 12}, 60)
    assert result["retained_gain"] is None
    assert not result["passed"]
