"""Prospective all-frame regional objectives and saved-curve sensitivity.

These APIs do not change historical scoring. Regional distances are custom
objectives, not calibrated human preference. Interpolated reference envelopes
are constructions; this module makes no claim that switching realizes them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy.interpolate import PchipInterpolator

Point = tuple[float, float]  # positive bitrate, quality oriented higher-is-better


@dataclass(frozen=True)
class RegionalFrame:
    foreground: float | None
    background: float | None
    foreground_pixels: int
    background_pixels: int


def weighted_region(frame: RegionalFrame, foreground_weight: float = 0.7) -> float:
    """Score a frame, reallocating empty-region weight to the extant region.

    Scores for absent regions must be None. Present regions require finite,
    nonnegative distances. Neither missing scores nor empty frames are dropped.
    """
    if (isinstance(foreground_weight, bool) or not math.isfinite(foreground_weight)
            or not 0 <= foreground_weight <= 1):
        raise ValueError("foreground weight must lie in [0, 1]")
    for count, value in ((frame.foreground_pixels, frame.foreground),
                         (frame.background_pixels, frame.background)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("region pixel counts must be nonnegative integers")
        if count == 0:
            if value is not None:
                raise ValueError("absent region score must be None")
        elif value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("present region needs a finite nonnegative distance")
    if frame.foreground_pixels + frame.background_pixels == 0:
        raise ValueError("frame has no pixels")
    if frame.foreground_pixels == 0:
        assert frame.background is not None
        return frame.background
    if frame.background_pixels == 0:
        assert frame.foreground is not None
        return frame.foreground
    assert frame.foreground is not None and frame.background is not None
    return foreground_weight * frame.foreground + (1 - foreground_weight) * frame.background


def aggregate_regions(frames: Sequence[RegionalFrame], foreground_weight: float = 0.7) -> dict[str, object]:
    """Equal-frame mean with explicit coverage, never finite-value filtering."""
    if not frames:
        raise ValueError("no registered frames")
    values = [weighted_region(frame, foreground_weight) for frame in frames]
    return {"protocol": "regional-all-frame-v1", "foreground_weight": foreground_weight,
            "frames": len(frames), "values": values, "mean": math.fsum(values) / len(values),
            "foreground_only": sum(f.background_pixels == 0 for f in frames),
            "background_only": sum(f.foreground_pixels == 0 for f in frames),
            "both_regions": sum(f.foreground_pixels > 0 and f.background_pixels > 0 for f in frames)}


def _front(points: Sequence[Point]) -> list[Point]:
    if not points:
        raise ValueError("missing curve")
    best: dict[float, float] = {}
    for rate, quality in points:
        if (isinstance(rate, bool) or isinstance(quality, bool)
                or not math.isfinite(rate) or not math.isfinite(quality) or rate <= 0):
            raise ValueError("curve needs finite quality and positive finite rates")
        best[rate] = max(best.get(rate, -math.inf), quality)
    result: list[Point] = []
    for rate, quality in sorted(best.items()):
        if not result or quality > result[-1][1]:
            result.append((rate, quality))
    return result


def _hull(front: Sequence[Point]) -> list[Point]:
    hull: list[Point] = []
    for point in front:
        while len(hull) >= 2:
            (r0, q0), (r1, q1) = hull[-2:]
            if (q1 - q0) * (point[0] - r0) > (point[1] - q0) * (r1 - r0):
                break
            hull.pop()
        hull.append(point)
    return hull


def _pchip_area(front: Sequence[Point], low: float, high: float) -> float:
    interpolator = PchipInterpolator([q for _, q in front], [math.log(r) for r, _ in front],
                                    extrapolate=False)
    return float(interpolator.integrate(low, high))


def _linear_rate_area(front: Sequence[Point], low: float, high: float) -> float:
    # Exact integral of log(rate) where rate is piecewise affine in quality.
    total = 0.0
    for (r0, q0), (r1, q1) in zip(front, front[1:]):
        left, right = max(low, q0), min(high, q1)
        if right <= left:
            continue
        slope = (r1 - r0) / (q1 - q0)
        rl = r0 + slope * (left - q0)
        rr = r0 + slope * (right - q0)
        # Recenter to avoid subtracting two large antiderivatives.
        delta = (rr - rl) / rl
        if abs(delta) < 1e-7:
            average = math.log(rl) + delta / 2 - delta * delta / 6 + delta**3 / 12
        else:
            average = math.log(rl) + (1 + delta) * math.log1p(delta) / delta - 1
        total += (right - left) * average
    return total


def anchor_sensitivity(candidate: Sequence[Point], anchor: Sequence[Point], *,
                       discrete: bool = False, minimum_quality_span: float = 0.0) -> dict[str, object]:
    """Compare three geometries from saved scalars on one common quality interval.

    Input quality must already be oriented higher-is-better (negate distances).
    ``hull_pchip`` reproduces the historical linear-rate hull selection followed
    by PCHIP log-rate integration; ``pareto_pchip`` retains Pareto knots instead;
    ``hull_linear_rate`` uses literal piecewise linear rate-quality chords for
    both curves. Fractions are BD-rate: -0.1 means 10% fewer bits. No discrete
    metric gets a BD-rate and no curve is extrapolated. This diagnostic cannot
    establish native stream validity, physical switching, or independence.
    """
    if (isinstance(minimum_quality_span, bool) or not math.isfinite(minimum_quality_span)
            or minimum_quality_span < 0):
        raise ValueError("minimum quality span must be finite and nonnegative")
    test, reference = _front(candidate), _front(anchor)
    hull = _hull(reference)
    result: dict[str, object] = {
        "protocol": "anchor-geometry-sensitivity-v1",
        "quality_orientation": "higher_is_better", "physically_demonstrated_switching": False,
        "candidate_pareto": test, "anchor_pareto": reference, "anchor_linear_rate_hull": hull,
        "candidate_rate_support": [min(r for r, _ in candidate), max(r for r, _ in candidate)],
        "anchor_rate_support": [min(r for r, _ in anchor), max(r for r, _ in anchor)],
        "bd_rate": {"hull_pchip": None, "pareto_pchip": None, "hull_linear_rate": None},
    }
    if discrete:
        result["status"] = "discrete_unsupported"
        result["common_quality_interval"] = None
        return result
    low, high = max(test[0][1], reference[0][1]), min(test[-1][1], reference[-1][1])
    if min(len(test), len(reference)) < 2 or high <= low or high - low < minimum_quality_span:
        result["status"] = "unsupported_range"
        result["common_quality_interval"] = None
        return result
    result["status"] = "supported"
    result["common_quality_interval"] = [low, high]
    candidate_area = _pchip_area(test, low, high)
    result["bd_rate"] = {
        "hull_pchip": math.expm1((candidate_area - _pchip_area(hull, low, high)) / (high - low)),
        "pareto_pchip": math.expm1((candidate_area - _pchip_area(reference, low, high)) / (high - low)),
        "hull_linear_rate": math.expm1((_linear_rate_area(test, low, high)
                                        - _linear_rate_area(hull, low, high)) / (high - low)),
    }
    return result
