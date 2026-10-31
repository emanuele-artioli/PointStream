"""Finite-match player comparison against measured, log-rate interpolated anchors.

All points must come from the same window and metric. Rates include every stream;
`fixed_kbps` excludes only the explicitly supplied one-time bytes. The result
covers measured rate brackets only and makes no monotonic-quality assumption.
"""
from __future__ import annotations

import math
from typing import Any


def intervals(points: list[tuple[float, float]], hybrid: float, fixed_kbps: float,
              one_time_bytes: int) -> list[dict[str, Any]]:
    """Classify each measured segment, splitting at every quality crossing.

    Lower score is better. Length boundaries are in seconds; `None` means
    infinity. Endpoints at an equal-quality crossing are ties, not wins.
    Anything outside the union of returned intervals is unresolved.
    """
    if not math.isfinite(hybrid) or not math.isfinite(fixed_kbps) or fixed_kbps < 0 or one_time_bytes <= 0:
        raise ValueError('invalid hybrid, fixed rate or one-time bytes')
    points = sorted(points)
    if len(points) < 2 or any(not math.isfinite(r) or not math.isfinite(q) or r <= 0 for r, q in points):
        raise ValueError('at least two finite, positive-rate points required')
    if any(a[0] == b[0] for a, b in zip(points, points[1:])):
        raise ValueError('duplicate rates are ambiguous')
    bit_kilobits = one_time_bytes * 8 / 1000
    out = []
    for (r0, q0), (r1, q1) in zip(points, points[1:]):
        lo = max(r0, fixed_kbps)
        if lo >= r1:
            continue
        cuts = [lo, r1]
        if q0 != q1:
            fraction = (hybrid - q0) / (q1 - q0)
            if 0 < fraction < 1:
                crossing = math.exp(math.log(r0) + fraction * (math.log(r1) - math.log(r0)))
                if lo < crossing < r1:
                    cuts.insert(1, crossing)
        for a, b in zip(cuts, cuts[1:]):
            mid = math.sqrt(a * b)
            value = q0 + (q1 - q0) * math.log(mid / r0) / math.log(r1 / r0)
            out.append({'rate_kbps': [a, b], 'length_seconds': [bit_kilobits / (b - fixed_kbps),
                        None if a == fixed_kbps else bit_kilobits / (a - fixed_kbps)],
                        'outcome': 'win' if value > hybrid else 'loss' if value < hybrid else 'tie',
                        'boundary_note': 'quality equality is a tie; outside measured brackets is unresolved'})
    return out


def main() -> None:
    """Analyze a JSON file of historical, adapter-only development cases."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    cases = json.loads(args.input.read_text())['cases']
    results = {}
    for case in cases:
        byte_count = case['adapter_bytes']
        fixed = case['rate_7200_kbps'] - byte_count * 8 / 7200 / 1000
        results[case['name']] = intervals(case['points'], case['hybrid'], fixed, byte_count)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
