"""Bjøntegaard delta rate — the currency every comparison is settled in.

Two configurations never land at the same bitrate or the same quality, so a
point-to-point comparison compares nothing unless one arm dominates on both
axes. BD-rate integrates the log-rate gap over the overlapping quality range
and refuses to return a number when that overlap is a sliver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.contracts.metrics import PSNR, MetricSpec

MIN_OVERLAP_FRACTION = 0.5
MIN_POINTS = 2

#: Absolute floor on the overlapping quality range, in dB, for a PSNR curve.
#:
#: `MIN_OVERLAP_FRACTION` is a *proportion* of the shorter curve's span, and a
#: proportion cannot see an absolutely tiny span. Measured during BP24: encoding
#: gradient-plus-noise gave a **0.5 dB** span across QP 32 to 46, because the
#: encoder discards incompressible noise at every QP and PSNR saturates. Both
#: curves were flat, so they overlapped almost completely — `overlap_fraction`
#: came out at 1.0 and this function returned a confident-looking BD-rate of
#: -0.88 over a range where nothing was resolved (`plans/BP24-findings.md` §2).
#:
#: Three dB is a little over one QP step of separation on real content; the real
#: four-frame curves that established the instrument in the same session spanned
#: 14.75 dB, so this rejects the degenerate case without touching a usable one.
MIN_QUALITY_SPAN_DB = 3.0


@dataclass(frozen=True)
class RDCurve:
    """One rate–distortion curve. Rates must be positive; at least two points."""

    rates: tuple[float, ...]
    qualities: tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.rates) != len(self.qualities):
            raise ValueError(
                f"RDCurve {self.label!r} has {len(self.rates)} rates and "
                f"{len(self.qualities)} qualities."
            )
        if len(self.rates) < MIN_POINTS:
            raise ValueError(
                f"RDCurve {self.label!r} needs at least {MIN_POINTS} points; "
                f"got {len(self.rates)}."
            )
        if any(rate <= 0 for rate in self.rates):
            raise ValueError(f"RDCurve {self.label!r} has a non-positive rate.")


@dataclass(frozen=True)
class OperatingPoint:
    """A single (rate, quality) pair. Rate is lower-better."""

    rate: float
    quality: float


@dataclass(frozen=True)
class BDComparison:
    """BD-rate and BD-quality over a reported overlap, or nothing."""

    bd_rate: float
    """Fractional extra rate at equal quality. Negative means the candidate is cheaper."""

    bd_quality: float
    """Mean quality gap at equal rate, in the quality metric's native units."""

    overlap: tuple[float, float]
    """Quality range the BD-rate integral is defined on, ``(low, high)``."""

    overlap_fraction: float
    """Overlap width divided by the shorter curve's quality span."""

    quality_metric: str

    @property
    def bd_rate_percent(self) -> float:
        return self.bd_rate * 100.0


class InsufficientOverlapError(ValueError):
    """Curves barely overlap; a BD-rate number would look like a result and is not."""

    def __init__(
        self,
        overlap: tuple[float, float],
        overlap_fraction: float,
        minimum: float,
    ) -> None:
        self.overlap = overlap
        self.overlap_fraction = overlap_fraction
        self.minimum = minimum
        low, high = overlap
        super().__init__(
            f"RD curves overlap on quality [{low}, {high}] "
            f"({overlap_fraction:.1%} of the shorter span); "
            f"need at least {minimum:.0%} to report BD-rate. "
            "Weak overlap is not a result."
        )


class DegenerateCurveError(InsufficientOverlapError):
    """The curves overlap well but resolve almost nothing, so BD-rate is noise.

    The opposite failure to its parent: two flat curves overlap *perfectly*, so
    the relative guard passes and the number that comes back is a polynomial
    fitted to a fraction of a dB.

    It **subclasses** `InsufficientOverlapError` so every existing caller that
    already declines to report a number on a bad overlap declines here too,
    rather than this guard arriving as an uncaught crash in code that was
    handling the other case correctly. New code can still tell them apart.
    """

    def __init__(
        self,
        overlap: tuple[float, float],
        span: float,
        minimum: float,
        overlap_fraction: float,
    ) -> None:
        self.overlap = overlap
        self.span = span
        self.minimum = minimum
        # The measured fraction, carried through unchanged. A caller that logs
        # it will typically see a number near 1.0, which is the trap itself.
        self.overlap_fraction = overlap_fraction
        low, high = overlap
        ValueError.__init__(
            self,
            f"RD curves overlap on quality [{low:.3f}, {high:.3f}] — a span of "
            f"{span:.3f}, below the {minimum} floor. The curves overlap but "
            "resolve nothing: a BD-rate integrated over this range is fitted "
            "to noise, however confident it looks. Widen the rung range, or "
            "check whether the content is compressible at all "
            "(incompressible noise saturates PSNR at every QP)."
        )


def compare_rd_curves(
    anchor: RDCurve,
    candidate: RDCurve,
    *,
    quality_spec: MetricSpec = PSNR,
    min_overlap_fraction: float = MIN_OVERLAP_FRACTION,
    min_quality_span: float | None = None,
) -> BDComparison:
    """Bjøntegaard delta of ``candidate`` against ``anchor``.

    Fits ``log10(rate) = p(quality)`` with a polynomial of degree at most 3
    (degree 1 when a curve has two points, so a hand-computable linear case
    is exact) and integrates over the overlapping quality range.

    Two guards, and they catch opposite failures. ``min_overlap_fraction`` is
    relative: it rejects curves that barely meet. ``min_quality_span`` is
    absolute: it rejects curves that meet everywhere but span nothing, which
    the relative guard cannot see (`plans/BP24-findings.md` §2).

    ``min_quality_span`` is in the quality metric's own units, so there is no
    default that is right for every metric. `PSNR` gets `MIN_QUALITY_SPAN_DB`;
    any other spec must state one, because silently applying a dB floor to an
    LPIPS curve would reject every real comparison, and silently applying none
    would reinstate the hole this guard exists to close.

    Raises:
        InsufficientOverlapError: The curves barely overlap.
        DegenerateCurveError: They overlap, but over a range that resolves
            nothing.
        ValueError: ``min_quality_span`` was left to the default on a
            non-PSNR metric.
    """
    if min_quality_span is None:
        if quality_spec.name != PSNR.name:
            raise ValueError(
                f"compare_rd_curves has no default quality span for "
                f"{quality_spec.name!r} — the floor is in the metric's own "
                f"units ({quality_spec.unit or 'unitless'}), and MIN_QUALITY_SPAN_DB "
                "is a PSNR figure. Pass min_quality_span explicitly."
            )
        min_quality_span = MIN_QUALITY_SPAN_DB

    low, high, fraction = _quality_overlap(anchor, candidate)
    if high <= low or fraction < min_overlap_fraction:
        raise InsufficientOverlapError((low, high), fraction, min_overlap_fraction)
    if (high - low) < min_quality_span:
        raise DegenerateCurveError((low, high), high - low, min_quality_span, fraction)

    log_rate_gap = _mean_poly_gap(
        _sorted(anchor.qualities, np.log10(np.asarray(anchor.rates, dtype=float))),
        _sorted(candidate.qualities, np.log10(np.asarray(candidate.rates, dtype=float))),
        low,
        high,
    )
    bd_rate = float(10**log_rate_gap - 1.0)

    rate_low = max(min(anchor.rates), min(candidate.rates))
    rate_high = min(max(anchor.rates), max(candidate.rates))
    bd_quality = _mean_poly_gap(
        _sorted(np.log10(np.asarray(anchor.rates, dtype=float)), anchor.qualities),
        _sorted(np.log10(np.asarray(candidate.rates, dtype=float)), candidate.qualities),
        np.log10(rate_low),
        np.log10(rate_high),
    )
    return BDComparison(
        bd_rate=bd_rate,
        bd_quality=float(bd_quality),
        overlap=(low, high),
        overlap_fraction=fraction,
        quality_metric=quality_spec.name,
    )


def dominates(
    challenger: OperatingPoint,
    incumbent: OperatingPoint,
    spec: MetricSpec,
) -> bool:
    """True when ``challenger`` is strictly cheaper and strictly better quality.

    The only valid point comparison. Uses ``spec.is_better`` so LPIPS and PSNR
    share this helper with no per-name branch.
    """
    cheaper = challenger.rate < incumbent.rate
    better = spec.is_better(challenger.quality, incumbent.quality)
    return cheaper and better


def _quality_overlap(anchor: RDCurve, candidate: RDCurve) -> tuple[float, float, float]:
    low = max(min(anchor.qualities), min(candidate.qualities))
    high = min(max(anchor.qualities), max(candidate.qualities))
    span_a = max(anchor.qualities) - min(anchor.qualities)
    span_b = max(candidate.qualities) - min(candidate.qualities)
    shorter = min(span_a, span_b)
    width = high - low
    fraction = (width / shorter) if shorter > 0 else 0.0
    return float(low), float(high), float(fraction)


def _sorted(x_values: Sequence[float] | np.ndarray, y_values: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(np.unique(x)) != len(x):
        raise ValueError("duplicate x-values on an RD curve make the polynomial ill-defined.")
    return x, y


def _mean_poly_gap(
    anchor: tuple[np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray],
    low: float,
    high: float,
) -> float:
    if high <= low:
        raise InsufficientOverlapError((low, high), 0.0, MIN_OVERLAP_FRACTION)
    p_anchor = _fit(anchor[0], anchor[1])
    p_candidate = _fit(candidate[0], candidate[1])
    gap = p_candidate - p_anchor
    integral = gap.integ()
    return float((integral(high) - integral(low)) / (high - low))


def _fit(x: np.ndarray, y: np.ndarray) -> np.poly1d:
    degree = min(3, len(x) - 1)
    return np.poly1d(np.polyfit(x, y, degree))
