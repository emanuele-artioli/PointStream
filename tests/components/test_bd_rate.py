"""BD-rate is the comparison currency. Weak overlap is not a result.

Bound before believing, for the hand-computable pair below:

- Anchor rates [2000, 8000] at quality [20, 40]
- Candidate rates [1000, 4000] at the same qualities (half the bitrate everywhere)
- log10(rate_cand) − log10(rate_anchor) = log10(0.5) ≈ −0.30103 at every quality
- BD-rate = 10^(−0.30103) − 1 = −0.5 exactly (−50%)
- Over the overlapping log-rate range the quality gap is 10 dB exactly
  (the curves span 20 quality units over Δlog10(rate)=log10(4), and the
  horizontal shift is log10(2), so BD-quality = 20 * 0.5 = 10)

Plausible range: BD-rate in (−1, +∞) i.e. −100% (free) to unbounded extra
rate. −0.4 or −0.6 on this construction would mean the integral is wrong.
"""

from __future__ import annotations

import pytest

from src.components.metrics.bd_rate import (
    MIN_OVERLAP_FRACTION,
    BDComparison,
    InsufficientOverlapError,
    OperatingPoint,
    RDCurve,
    compare_rd_curves,
    dominates,
)
from src.contracts.metrics import metric as contract_metric


def test_half_rate_curve_is_minus_fifty_percent_bd_rate() -> None:
    anchor = RDCurve(rates=(2000.0, 8000.0), qualities=(20.0, 40.0), label="anchor")
    candidate = RDCurve(rates=(1000.0, 4000.0), qualities=(20.0, 40.0), label="half")
    result = compare_rd_curves(anchor, candidate)
    assert result.bd_rate == pytest.approx(-0.5, rel=1e-12)
    assert result.bd_rate_percent == pytest.approx(-50.0, rel=1e-12)
    assert result.bd_quality == pytest.approx(10.0, rel=1e-12)
    assert result.overlap == (20.0, 40.0)
    assert result.overlap_fraction == pytest.approx(1.0)
    assert result.quality_metric == "psnr"
    assert isinstance(result, BDComparison)


def test_bd_rate_reports_the_quality_overlap_range() -> None:
    anchor = RDCurve(rates=(100.0, 400.0), qualities=(20.0, 40.0))
    candidate = RDCurve(rates=(200.0, 800.0), qualities=(30.0, 50.0))
    result = compare_rd_curves(anchor, candidate)
    assert result.overlap == (30.0, 40.0)
    assert result.overlap_fraction == pytest.approx(0.5)


def test_barely_overlapping_curves_refuse_a_number() -> None:
    anchor = RDCurve(rates=(100.0, 200.0, 400.0), qualities=(10.0, 11.0, 12.0))
    candidate = RDCurve(rates=(150.0, 300.0, 600.0), qualities=(11.8, 30.0, 50.0))
    with pytest.raises(InsufficientOverlapError, match="Weak overlap is not a result") as caught:
        compare_rd_curves(anchor, candidate)
    assert caught.value.overlap[0] == pytest.approx(11.8)
    assert caught.value.overlap[1] == pytest.approx(12.0)
    assert caught.value.overlap_fraction < MIN_OVERLAP_FRACTION
    assert not hasattr(caught.value, "bd_rate") or getattr(caught.value, "bd_rate", None) is None


def test_non_overlapping_curves_refuse_a_number() -> None:
    anchor = RDCurve(rates=(100.0, 200.0), qualities=(10.0, 20.0))
    candidate = RDCurve(rates=(100.0, 200.0), qualities=(30.0, 40.0))
    with pytest.raises(InsufficientOverlapError):
        compare_rd_curves(anchor, candidate)


def test_dominance_is_the_only_valid_point_comparison() -> None:
    psnr = contract_metric("psnr")
    lpips = contract_metric("lpips")
    cheap_and_good = OperatingPoint(rate=100.0, quality=40.0)
    expensive_and_worse = OperatingPoint(rate=200.0, quality=30.0)
    cheap_but_worse = OperatingPoint(rate=100.0, quality=30.0)
    expensive_and_better = OperatingPoint(rate=200.0, quality=40.0)

    assert dominates(cheap_and_good, expensive_and_worse, psnr)
    assert not dominates(cheap_but_worse, expensive_and_better, psnr)
    assert not dominates(expensive_and_better, cheap_but_worse, psnr)

    assert dominates(
        OperatingPoint(rate=100.0, quality=0.10),
        OperatingPoint(rate=200.0, quality=0.40),
        lpips,
    )
    assert not dominates(
        OperatingPoint(rate=100.0, quality=0.40),
        OperatingPoint(rate=200.0, quality=0.10),
        lpips,
    )


def test_a_curve_with_one_point_is_not_an_rd_curve() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        RDCurve(rates=(100.0,), qualities=(30.0,))


def test_non_positive_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        RDCurve(rates=(0.0, 100.0), qualities=(20.0, 30.0))
