"""Arm comparisons must carry their uncertainty and refuse unsupported calls."""

from __future__ import annotations

import pytest

from src.components.metrics.comparison import compare_paired


def test_the_real_case_that_motivated_this_is_reported_as_suggestive() -> None:
    """+0.98 dB over 12 clips with per-clip sd ~2.0 was reported as a finding.

    It is ~1.7 sigma — suggestive at best. The verdict must say so rather than
    naming a winner outright.
    """
    retrained = [11.45, 12.27, 10.84, 10.57, 12.21, 11.57, 9.08, 11.91, 10.54, 12.67, 11.94, 9.16]
    baseline = [8.20, 9.09, 10.30, 10.66, 11.95, 10.72, 12.10, 7.89, 12.07, 11.10, 10.26, 8.16]
    result = compare_paired("correct", retrained, "wrong", baseline)

    assert result.n == 12
    assert result.mean_difference == pytest.approx(0.976, abs=0.01)
    assert result.verdict == "suggestive"
    assert "SUGGESTIVE" in result.describe()


def test_a_difference_inside_the_noise_names_no_winner() -> None:
    a = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 10.5, 11.5, 9.5]
    b = [10.1, 11.8, 8.3, 10.7, 9.4, 12.6, 7.4, 10.2, 11.2, 9.9]
    result = compare_paired("a", a, "b", b)
    assert result.verdict == "inside-noise"
    assert result.winner is None
    assert "do not report a winner" in result.describe()


def test_a_large_consistent_effect_is_clear() -> None:
    a = [20.0, 21.0, 19.5, 20.5, 22.0, 19.0, 20.2, 21.3, 20.8, 19.7]
    b = [10.0, 11.0, 9.5, 10.5, 12.0, 9.0, 10.2, 11.3, 10.8, 9.7]
    result = compare_paired("a", a, "b", b)
    assert result.verdict == "clear"
    assert result.winner == "a"


def test_lower_is_better_flips_the_winner_not_the_verdict() -> None:
    a = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11, 0.09, 0.10, 0.11, 0.10]
    b = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50, 0.52, 0.49]
    result = compare_paired("a", a, "b", b, higher_is_better=False)
    assert result.verdict == "clear"
    assert result.winner == "a"


def test_too_few_items_is_underpowered_rather_than_a_winner() -> None:
    result = compare_paired("a", [1.0, 2.0, 3.0], "b", [0.0, 1.0, 2.0])
    assert result.verdict == "underpowered"
    assert result.winner is None
    assert "UNDERPOWERED" in result.describe()


def test_a_single_item_is_refused_outright() -> None:
    with pytest.raises(ValueError, match="at least two items"):
        compare_paired("a", [1.0], "b", [0.0])


def test_describe_names_the_quality_axis_when_a_spec_is_passed() -> None:
    from src.contracts.metrics import metric as contract_metric

    a = [20.0, 21.0, 19.5, 20.5, 22.0, 19.0, 20.2, 21.3, 20.8, 19.7]
    b = [10.0, 11.0, 9.5, 10.5, 12.0, 9.0, 10.2, 11.3, 10.8, 9.7]
    result = compare_paired("a", a, "b", b, quality_spec=contract_metric("vmaf"))
    assert result.quality_metric == "vmaf"
    assert "vmaf" in result.describe()
    assert result.winner == "a"


def test_a_spec_and_a_disagreeing_direction_flag_are_refused() -> None:
    from src.contracts.metrics import metric as contract_metric

    with pytest.raises(ValueError, match="disagrees"):
        compare_paired(
            "a",
            [0.1, 0.2],
            "b",
            [0.3, 0.4],
            quality_spec=contract_metric("lpips"),
            higher_is_better=True,
        )
