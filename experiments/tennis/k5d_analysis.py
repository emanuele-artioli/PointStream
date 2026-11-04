"""Per-clip ball study summaries; no acceptance rule or inferred independence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from experiments.background.g5 import envelope, pareto
from experiments.tennis.k5 import bd_rate

CONTINUOUS = {"frame_psnr": 1, "crop_lpips": -1, "crop_dists": -1, "full_lpips": -1, "full_dists": -1}
DISCRETE = {"recall5", "recall10", "false_per_negative"}


@dataclass(frozen=True)
class Point:
    codec: str
    variant: str
    setting: str
    byte_count: int
    duration: Fraction
    frames: int
    metric: str
    value: float

    @property
    def rate(self) -> float:
        return self.byte_count * 8 / float(self.duration) / 1000


def validate_points(rows: Sequence[Point], frames: int, duration: Fraction, metric: str) -> None:
    """Reject incomplete or incomparable clip records before constructing curves."""
    if not rows or frames <= 0 or not isinstance(duration, Fraction) or duration <= 0:
        raise ValueError("missing rows or invalid registered clip duration/count")
    if metric not in CONTINUOUS and metric not in DISCRETE:
        raise ValueError("unknown metric")
    seen = set()
    for row in rows:
        key = (row.codec, row.variant, row.setting)
        if key in seen:
            raise ValueError("duplicate point")
        seen.add(key)
        if (not row.codec or not row.variant or not row.setting or isinstance(row.byte_count, bool)
                or not isinstance(row.byte_count, int) or row.byte_count <= 0
                or row.frames != frames or row.duration != duration or row.metric != metric
                or isinstance(row.value, bool) or not math.isfinite(row.value)
                or not math.isfinite(row.rate) or row.rate <= 0):
            raise ValueError("invalid or incomparable point")
        if metric in {"crop_lpips", "crop_dists", "full_lpips", "full_dists"} and row.value < 0:
            raise ValueError("invalid perceptual distance")
        if metric in DISCRETE and not 0 <= row.value <= 1:
            raise ValueError("invalid task rate")


def _unique_rates(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    # At an identical rate, retain the best achievable quality before hull construction.
    best: dict[float, float] = {}
    for rate, quality in points:
        best[rate] = max(best.get(rate, -math.inf), quality)
    return sorted(best.items())


def quality_at(front: Sequence[tuple[float, float]], rate: float,
               support: tuple[float, float] | None = None) -> float | None:
    """Interpolate within measured support; retain best quality on a supported plateau."""
    if not front or not math.isfinite(rate):
        return None
    low, high = support if support is not None else (front[0][0], front[-1][0])
    if rate < low or rate > high or rate < front[0][0]:
        return None
    if rate >= front[-1][0]:
        return front[-1][1]
    if len(front) == 1:
        return front[0][1] if rate == front[0][0] else None
    for (r0, q0), (r1, q1) in zip(front, front[1:]):
        if r0 <= rate <= r1:
            weight = math.log(rate / r0) / math.log(r1 / r0)
            return q0 + weight * (q1 - q0)
    return None


def compare(rows: Sequence[Point], frames: int, duration: Fraction, metric: str,
            variant: str, minimum_quality_span: float) -> dict[str, object]:
    """Compare a variant with the pooled native envelope on one registered clip.

    Quality orientation is normalized internally; discrete measures get only
    observed and supported equal-rate values, never BD-rate.
    """
    validate_points(rows, frames, duration, metric)
    if not math.isfinite(minimum_quality_span) or minimum_quality_span <= 0:
        raise ValueError("minimum quality span must be positive")
    if not variant or variant == "native":
        raise ValueError("candidate variant required")
    sign = CONTINUOUS.get(metric, -1 if metric == "false_per_negative" else 1)
    anchor_rows = [r for r in rows if r.variant == "native"]
    test_rows = [r for r in rows if r.variant == variant]
    if not anchor_rows or not test_rows:
        raise ValueError("missing native or candidate points")
    anchor = envelope(_unique_rates([(r.rate, sign * r.value) for r in anchor_rows]))
    candidate = pareto(_unique_rates([(r.rate, sign * r.value) for r in test_rows]))
    anchor_support = (min(r.rate for r in anchor_rows), max(r.rate for r in anchor_rows))
    candidate_support = (min(r.rate for r in test_rows), max(r.rate for r in test_rows))
    observations = [{"codec": r.codec, "setting": r.setting, "rate_kbps": r.rate,
                     "value": r.value,
                     "anchor_at_rate": None if (q := quality_at(anchor, r.rate, anchor_support)) is None else sign * q}
                    for r in test_rows]
    result: dict[str, object] = {"metric": metric, "native_rate_support": anchor_support,
                                "candidate_rate_support": candidate_support, "native_points": [
        {"codec": r.codec, "setting": r.setting, "rate_kbps": r.rate, "value": r.value}
        for r in anchor_rows], "candidate_points": observations, "bd_rate": None}
    if metric in DISCRETE:
        result["status"] = "descriptive_discrete"
        return result
    lo = max(candidate[0][1], anchor[0][1])
    hi = min(candidate[-1][1], anchor[-1][1])
    result["quality_overlap"] = max(0.0, hi - lo)
    if len(candidate) >= 2 and len(anchor) >= 2 and hi > lo and hi - lo >= minimum_quality_span:
        result["bd_rate"] = bd_rate(candidate, anchor, minimum_quality_span)
        result["status"] = "bd_rate"
        return result
    if hi <= lo:
        rlo, rhi = max(anchor_support[0], candidate_support[0]), min(anchor_support[1], candidate_support[1])
        if rlo < rhi:
            knots = sorted({rlo, rhi, *(r for r, _ in anchor + candidate if rlo < r < rhi)})
            differences = []
            for rate in knots:
                a, c = quality_at(anchor, rate, anchor_support), quality_at(candidate, rate, candidate_support)
                assert a is not None and c is not None
                differences.append(c - a)
            if all(d > 0 for d in differences):
                result["status"] = "better_at_every_supported_rate"
                return result
            if all(d < 0 for d in differences):
                result["status"] = "worse_at_every_supported_rate"
                return result
    result["status"] = "unsupported_range"
    return result


def score_frames(labels: Sequence[tuple[int, float, float]],
                 predictions: Sequence[tuple[bool, float, float]]) -> dict[str, int | float | None]:
    """Count every frame; unknown/occluded labels are excluded, never negatives."""
    if not labels or len(labels) != len(predictions):
        raise ValueError("missing frames or prediction count mismatch")
    visible = absent = unknown = hit5 = hit10 = false = misses = 0
    for (kind, lx, ly), (detected, x, y) in zip(labels, predictions):
        if kind not in (-1, 0, 1) or not isinstance(detected, bool):
            raise ValueError("invalid label kind or prediction")
        if kind == 1 and (not math.isfinite(lx) or not math.isfinite(ly)):
            raise ValueError("invalid visible label")
        if detected and (not math.isfinite(x) or not math.isfinite(y)):
            raise ValueError("invalid detected position")
        if kind == -1:
            unknown += 1
        elif kind == 0:
            absent += 1
            false += int(detected)
        else:
            visible += 1
            d = math.hypot(x - lx, y - ly) if detected else math.inf
            hit5 += int(d <= 5)
            hit10 += int(d <= 10)
            misses += int(d > 10)
    return {"frames": len(labels), "visible": visible, "absent": absent, "unknown": unknown,
            "hit5": hit5, "hit10": hit10, "miss10": misses, "false_on_absent": false,
            "recall5": hit5 / visible if visible else None, "recall10": hit10 / visible if visible else None,
            "false_per_negative": false / absent if absent else None}


def clip_report(clip_id: str, match_id: str, exposure: str,
                source_qualification: dict[str, object], coverage: dict[str, int | float | None],
                comparison: dict[str, object]) -> dict[str, object]:
    """Bind per-clip evidence and provenance; do not aggregate correlated frames."""
    if not clip_id or not match_id or not exposure or not source_qualification:
        raise ValueError("missing clip provenance or source qualification")
    keys = ("frames", "visible", "absent", "unknown")
    counts: list[int] = []
    for key in keys:
        value = coverage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("invalid frame coverage")
        counts.append(value)
    if counts[0] != sum(counts[1:]) or not counts[0]:
        raise ValueError("incomplete frame coverage")
    return {"clip_id": clip_id, "match_id": match_id, "exposure": exposure,
            "source_qualification": source_qualification, "coverage": coverage, "comparison": comparison}
