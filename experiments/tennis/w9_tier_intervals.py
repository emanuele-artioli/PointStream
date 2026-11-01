"""Conservative finite-length intervals for the preregistered W7 tier rule.

Use reciprocal length (u=1/L): every hybrid rate is affine in u. Each anchor
curve is linear in log-rate within observed brackets. Interval bounds never
extrapolate and do not assume monotone quality or a monotone tier decision.
Cells near unresolved confidence boundaries remain explicitly unresolved.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from scipy.stats import t

METRICS = ('lpips', 'dists')


def curve_range(points: list[tuple[float, float]], low: float, high: float) -> tuple[float, float]:
    """Exact quality range on a rate interval inside a piecewise log-rate curve."""
    rows = sorted(points)
    if len(rows) < 2 or any(not math.isfinite(r) or not math.isfinite(q) or r <= 0 for r, q in rows):
        raise ValueError('invalid anchor points')
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('duplicate anchor rates')
    tolerance = 4 * math.ulp(max(1.0, rows[-1][0]))
    if low > high or low < rows[0][0] - tolerance or high > rows[-1][0] + tolerance:
        raise ValueError('rate outside measured bracket')
    low, high = max(low, rows[0][0]), min(high, rows[-1][0])

    def value(rate: float) -> float:
        for (r0, q0), (r1, q1) in zip(rows, rows[1:]):
            if r0 <= rate <= r1:
                return q0 + (q1 - q0) * math.log(rate / r0) / math.log(r1 / r0)
        raise ValueError('no bracket')

    values = [value(low), value(high)] + [q for r, q in rows if low < r < high]
    return min(values), max(values)


def coverage(windows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Common reciprocal-length coverage of every required family, or None."""
    low, high = 0.0, math.inf
    for window in windows:
        fixed, bits = window['fixed_kbps'], window['one_time_bytes'] * 8 / 1000
        if not window['required'] or any(not math.isfinite(window['hybrid'][m]) for m in METRICS):
            raise ValueError('required families and finite hybrid scores are necessary')
        if fixed < 0 or not math.isfinite(fixed) or bits <= 0 or not math.isfinite(bits):
            raise ValueError('invalid rate accounting')
        for family in window['required']:
            if family not in window['anchors']:
                return None
            rows = window['anchors'][family]
            if len(rows) < 2:
                return None
            rates = sorted(float(row['rate_kbps']) for row in rows)
            if any(not math.isfinite(r) or r <= 0 for r in rates) or len(set(rates)) != len(rates):
                raise ValueError('invalid or duplicate anchor rates')
            # W7 requires a measured point strictly below the target. Move
            # the lower coverage edge inward by the minimum representable
            # reciprocal length that produces a rate above the first point.
            boundary = (rates[0] - fixed) / bits
            if boundary >= 0:
                inward = max(math.nextafter(boundary, math.inf),
                             (math.nextafter(rates[0], math.inf) - fixed) / bits)
                while fixed + bits * inward <= rates[0]:
                    inward = math.nextafter(inward, math.inf)
                low = max(low, inward)
            high = min(high, (rates[-1] - fixed) / bits)
    return (low, high) if low < high and math.isfinite(high) else None


@lru_cache(maxsize=None)
def critical(n: int) -> float:
    return float(t.ppf(.975, n - 1))


def confidence_bounds(gaps: list[tuple[float, float]]) -> tuple[float, float]:
    """Safe enclosure of the 95% t lower limit over independent gap intervals."""
    n = len(gaps)
    if n < 2:
        raise ValueError('at least two windows are required')
    if any(not math.isfinite(a) or not math.isfinite(b) or a > b for a, b in gaps):
        raise ValueError('invalid gap bounds')
    mean_low = sum(a for a, _ in gaps) / n
    mean_high = sum(b for _, b in gaps) / n
    min_pairs = max_pairs = 0.0
    for i, (a, b) in enumerate(gaps):
        for c, d in gaps[i + 1:]:
            min_pairs += max(0.0, a - d, c - b) ** 2
            max_pairs += max(abs(a - d), abs(b - c)) ** 2
    factor = critical(n) / math.sqrt(n * n * (n - 1))
    return mean_low - factor * math.sqrt(max_pairs), mean_high - factor * math.sqrt(min_pairs)


def classify(windows: list[dict[str, Any]], u_low: float, u_high: float) -> str:
    """Certify joint t-rule pass/fail on a covered reciprocal-length cell."""
    bounds: dict[str, list[tuple[float, float]]] = {m: [] for m in METRICS}
    for window in windows:
        bits = window['one_time_bytes'] * 8 / 1000
        r0, r1 = window['fixed_kbps'] + bits * u_low, window['fixed_kbps'] + bits * u_high
        for metric in METRICS:
            ranges = [curve_range([(p['rate_kbps'], p[f'foreground_{metric}']) for p in window['anchors'][family]], r0, r1)
                      for family in window['required']]
            bounds[metric].append((min(x[0] for x in ranges) - window['hybrid'][metric],
                                   min(x[1] for x in ranges) - window['hybrid'][metric]))
    limits = [confidence_bounds(bounds[m]) for m in METRICS]
    if all(low > 0 for low, _ in limits):
        return 'pass'
    if any(high <= 0 for _, high in limits):
        return 'fail'
    return 'unresolved'


def intervals(windows: list[dict[str, Any]], *, max_depth: int = 18) -> dict[str, Any]:
    """Return covered, certified intervals in seconds for one tier."""
    return joint_intervals([windows], max_depth=max_depth)


def joint_intervals(tiers: list[list[dict[str, Any]]], *, max_depth: int = 18) -> dict[str, Any]:
    """Certify each tier separately, then intersect their joint-pass intervals."""
    if not tiers or any(len(group) < 2 for group in tiers):
        raise ValueError('each tier needs at least two windows')
    span = coverage([window for group in tiers for window in group])
    if span is None:
        return {'coverage': None, 'intervals': [], 'reason': 'missing family or no common measured bracket'}
    stack = [(span[0], span[1], 0)]
    cells: list[tuple[float, float, str]] = []
    while stack:
        low, high, depth = stack.pop()
        verdicts = [classify(group, low, high) for group in tiers]
        verdict = 'fail' if 'fail' in verdicts else 'pass' if all(v == 'pass' for v in verdicts) else 'unresolved'
        if verdict == 'unresolved' and depth < max_depth:
            mid = (low + high) / 2
            stack.extend([(mid, high, depth + 1), (low, mid, depth + 1)])
        else:
            if cells and cells[-1][2] == verdict and cells[-1][1] == low:
                cells[-1] = (cells[-1][0], high, verdict)
            else:
                cells.append((low, high, verdict))
    def seconds(u: float) -> float | None:
        return None if u == 0 else 1 / u
    return {'coverage': [seconds(span[1]), seconds(span[0])],
            'intervals': [{'seconds': [seconds(high), seconds(low)], 'verdict': verdict}
                          for low, high, verdict in reversed(cells)],
            'note': 'None is infinity; lower-rate bracketing edges are moved inward; quality equality is a tie; unresolved cells are not passes.'}
