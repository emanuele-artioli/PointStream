"""W5: DCVC-UF against the Med hybrid at the hybrid's rates (the rule of W3b, W1's dominance test).

python -m experiments.tennis.w5_dcvc score.json w3b-report.json [points.json]
Uses the W3b Med scale 4 points listed below (docs/experiments/w3-neural-baseline.md); a rate outside their
span is unresolved, never extrapolated.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from experiments.tennis import w1_roi as w

#: Med, DCVC-UF scale 4 (job 20261029T190006Z-d48a344b): QP, kbps, whole LPIPS/DISTS, foreground LPIPS/DISTS.
MED_DCVC = [(15, 12.767, 0.4314, 0.2209, 0.6899, 0.4446), (36, 35.812, 0.4000, 0.1932, 0.5901, 0.3494),
            (63, 130.822, 0.3822, 0.1757, 0.5308, 0.3131)]


def points(table: list[tuple[float, ...]] | None = None) -> list[dict[str, Any]]:
    """The Med W3b points by default; a W7 window passes its own (QP, kbps, whole LPIPS, whole DISTS, fg LPIPS, fg DISTS) rows."""
    return [{'crf': q, 'rate_kbps': r, 'whole_lpips': a, 'whole_dists': b, 'foreground_lpips': c, 'foreground_dists': d}
            for q, r, a, b, c, d in (MED_DCVC if table is None else table)]


def verdicts(hybrid: dict[str, dict[str, float]], rates: dict[str, float], table: list[tuple[float, ...]] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, rate in rates.items():
        value = w.policy_at_target(points(table), rate)
        out[label] = {'rate_kbps': rate, 'value': value, 'verdict': None if value is None else
                      {m: w.dominates(value, m, hybrid) for m in w.METRICS}}
    return out


def main() -> None:
    summary = json.load(open(sys.argv[1]))
    hybrid = summary['hybrid']['hybrid']
    rates = {'kps2': summary['hybrid']['kps2_kbps'], 'kps1': summary['hybrid']['kps1_kbps']}
    table = [tuple(r) for r in json.load(open(sys.argv[3]))] if len(sys.argv) > 3 else None  # W7: this window's DCVC points
    for label, v in verdicts(hybrid, rates, table).items():
        shown = None if v['value'] is None else {k: round(x, 4) for k, x in v['value'].items() if k.startswith(('whole', 'foreground'))}
        print(label, round(v['rate_kbps'], 4), shown, v['verdict'])


if __name__ == '__main__':
    main()
