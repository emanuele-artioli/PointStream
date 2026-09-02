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
    DegenerateCurveError,
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


# ---------------------------------------------------------------------------
# The absolute span guard (BP24 finding §2)
#
# `MIN_OVERLAP_FRACTION` is a proportion of the shorter curve's span, and a
# proportion is blind to a span that is absolutely tiny. Two flat curves overlap
# almost perfectly, so the relative guard reports a healthy 100% and the
# polynomial fit returns a confident number over a range that resolves nothing.
# Measured once for real: a 0.5 dB span across QP 32→46 returned −0.88.
# ---------------------------------------------------------------------------


def test_two_flat_curves_are_refused_however_well_they_overlap() -> None:
    """The exact failure §2 describes: perfect overlap, nothing resolved."""
    flat_anchor = RDCurve(rates=(100.0, 200.0, 400.0), qualities=(40.00, 40.20, 40.50))
    flat_candidate = RDCurve(rates=(90.0, 190.0, 380.0), qualities=(40.05, 40.25, 40.45))
    with pytest.raises(DegenerateCurveError) as caught:
        compare_rd_curves(flat_anchor, flat_candidate)
    # The trap made visible: the relative guard was perfectly happy.
    assert caught.value.overlap_fraction > 0.5
    assert caught.value.span < 1.0


def test_the_span_guard_is_caught_by_callers_handling_weak_overlap() -> None:
    """Existing callers decline to report a number; they must decline here too.

    `experiments/headroom/measure.py` catches `InsufficientOverlapError` and
    returns `saving: None`. A sibling exception would have arrived there as an
    uncaught crash in code that was already handling the other bad case
    correctly, so the new guard subclasses the old one.
    """
    flat_anchor = RDCurve(rates=(100.0, 400.0), qualities=(40.0, 40.4))
    flat_candidate = RDCurve(rates=(90.0, 380.0), qualities=(40.1, 40.3))
    with pytest.raises(InsufficientOverlapError):
        compare_rd_curves(flat_anchor, flat_candidate)


def test_a_real_span_still_returns_a_number() -> None:
    """The guard must reject the degenerate case without touching a usable one.

    The four real 960x540 frames that established the instrument in the same
    session spanned 14.75 dB. This pair spans 14, well clear of the 3 dB floor.
    """
    anchor = RDCurve(rates=(1.0e5, 4.0e5, 1.6e6), qualities=(28.0, 35.0, 42.0))
    candidate = RDCurve(rates=(0.5e5, 2.0e5, 0.8e6), qualities=(28.0, 35.0, 42.0))
    comparison = compare_rd_curves(anchor, candidate)
    # Half the rate at every quality, so exactly −50%, same construction as the
    # hand-computed pair at the top of this file.
    assert comparison.bd_rate == pytest.approx(-0.5, abs=1e-6)


def test_a_metric_without_a_span_floor_must_still_state_one() -> None:
    """REID is not a BD-rate axis; a dB or LPIPS floor would be a guess."""
    reid = contract_metric("reid")
    assert reid.min_curve_span is None
    anchor = RDCurve(rates=(1.0e5, 1.6e6), qualities=(0.80, 0.50), quality_spec=reid)
    candidate = RDCurve(rates=(0.5e5, 0.8e6), qualities=(0.80, 0.50), quality_spec=reid)
    with pytest.raises(ValueError, match="no default quality span"):
        compare_rd_curves(anchor, candidate)
    comparison = compare_rd_curves(anchor, candidate, min_quality_span=0.10)
    assert comparison.bd_rate == pytest.approx(-0.5, abs=1e-6)
    assert comparison.quality_metric == "reid"


def test_lpips_carries_its_own_span_floor_and_negates_for_the_fit() -> None:
    """A dB floor applied to LPIPS would reject everything; none would be a hole.

    The floor now lives on the spec. Half the rate at every native LPIPS still
    integrates to −50%, and BD-quality is positive because the candidate is
    better (lower LPIPS) at equal rate after the higher-is-better transform.
    """
    lpips = contract_metric("lpips")
    assert lpips.min_curve_span == pytest.approx(0.05)
    assert lpips.curve_quality_transform == "negate"
    anchor = RDCurve(
        rates=(1.0e5, 1.6e6),
        qualities=(0.30, 0.05),
        quality_spec=lpips,
    )
    candidate = RDCurve(
        rates=(0.5e5, 0.8e6),
        qualities=(0.30, 0.05),
        quality_spec=lpips,
    )
    comparison = compare_rd_curves(anchor, candidate)
    assert comparison.bd_rate == pytest.approx(-0.5, abs=1e-6)
    assert comparison.quality_metric == "lpips"
    assert comparison.overlap[0] == pytest.approx(0.05)
    assert comparison.overlap[1] == pytest.approx(0.30)
    assert comparison.bd_quality > 0.0


def test_a_lower_better_curve_and_its_higher_better_negation_agree() -> None:
    """−LPIPS labelled higher-is-better must match LPIPS, not invert the sign.

    Forgetting the spec flip is the silent inversion BP35 named. The matching
    higher-better axis uses the same numeric span, so the rate integral agrees.
    """
    lpips = contract_metric("lpips")
    psnr = contract_metric("psnr")
    anchor_lpips = RDCurve(rates=(1.0e5, 1.6e6), qualities=(0.30, 0.05), quality_spec=lpips)
    candidate_lpips = RDCurve(rates=(0.5e5, 0.8e6), qualities=(0.30, 0.05), quality_spec=lpips)
    correct = compare_rd_curves(anchor_lpips, candidate_lpips)

    anchor_neg = RDCurve(rates=(1.0e5, 1.6e6), qualities=(-0.30, -0.05), quality_spec=psnr)
    candidate_neg = RDCurve(rates=(0.5e5, 0.8e6), qualities=(-0.30, -0.05), quality_spec=psnr)
    as_higher = compare_rd_curves(anchor_neg, candidate_neg, min_quality_span=0.05)
    assert as_higher.bd_rate == pytest.approx(correct.bd_rate, abs=1e-9)
    assert as_higher.bd_quality == pytest.approx(correct.bd_quality, abs=1e-9)

    forgotten = compare_rd_curves(
        RDCurve(rates=(1.0e5, 1.6e6), qualities=(0.30, 0.05), quality_spec=psnr),
        RDCurve(rates=(0.5e5, 0.8e6), qualities=(0.30, 0.05), quality_spec=psnr),
        min_quality_span=0.05,
    )
    assert forgotten.bd_rate == pytest.approx(correct.bd_rate, abs=1e-9)
    assert forgotten.bd_quality == pytest.approx(-correct.bd_quality, abs=1e-9)


def test_vmaf_refuses_a_span_below_ten_points() -> None:
    """A VMAF sliver of 3 points would pass a leftover dB floor and mean nothing."""
    vmaf = contract_metric("vmaf")
    assert vmaf.min_curve_span == pytest.approx(10.0)
    sliver = RDCurve(rates=(100.0, 400.0), qualities=(90.0, 95.0), quality_spec=vmaf)
    other = RDCurve(rates=(80.0, 320.0), qualities=(90.0, 95.0), quality_spec=vmaf)
    with pytest.raises(DegenerateCurveError) as caught:
        compare_rd_curves(sliver, other)
    assert caught.value.span == pytest.approx(5.0)

    usable = RDCurve(rates=(100.0, 400.0), qualities=(70.0, 90.0), quality_spec=vmaf)
    cheap = RDCurve(rates=(50.0, 200.0), qualities=(70.0, 90.0), quality_spec=vmaf)
    comparison = compare_rd_curves(usable, cheap)
    assert comparison.bd_rate == pytest.approx(-0.5, abs=1e-6)
    assert comparison.quality_metric == "vmaf"


def test_swapping_arms_flips_the_bd_rate_sign() -> None:
    lpips = contract_metric("lpips")
    anchor = RDCurve(rates=(1.0e5, 1.6e6), qualities=(0.30, 0.05), quality_spec=lpips)
    candidate = RDCurve(rates=(0.5e5, 0.8e6), qualities=(0.30, 0.05), quality_spec=lpips)
    forward = compare_rd_curves(anchor, candidate)
    backward = compare_rd_curves(candidate, anchor)
    assert backward.bd_rate == pytest.approx(-forward.bd_rate / (1.0 + forward.bd_rate), abs=1e-9)


def test_gate_a_floor_accepts_equal_quality_when_cheaper() -> None:
    psnr = contract_metric("psnr")
    lpips = contract_metric("lpips")
    floor = OperatingPoint(rate=10_000.0, quality=20.0)
    winner = OperatingPoint(rate=8_000.0, quality=20.0)
    loser = OperatingPoint(rate=8_000.0, quality=19.0)
    from src.components.metrics.bd_rate import meets_or_beats_floor

    assert meets_or_beats_floor(winner, floor, psnr)
    assert not meets_or_beats_floor(loser, floor, psnr)
    assert meets_or_beats_floor(
        OperatingPoint(rate=8_000.0, quality=0.40),
        OperatingPoint(rate=10_000.0, quality=0.40),
        lpips,
    )
    assert not meets_or_beats_floor(
        OperatingPoint(rate=8_000.0, quality=0.50),
        OperatingPoint(rate=10_000.0, quality=0.40),
        lpips,
    )
