"""Print the W5 table and the rule per anchor family from the Med scoring job's summary.

python -m experiments.tennis.w5_report score-summary.json
"""
from __future__ import annotations

import json
import sys
from typing import Any

from experiments.tennis import w1_roi as w

KEYS = ('whole_lpips', 'foreground_lpips', 'whole_dists', 'foreground_dists')
FAMILIES = {'half': 's2_', 'quarter': 's4_'}


def family_outcomes(analysis: dict[str, Any]) -> dict[str, Any]:
    """Per anchor family (resolution): outcome B if any of its policies dominates on either metric, else A.
    A family with no resolved policy is 'unresolved'; unresolved policies are listed, never extrapolated."""
    decision = analysis['decision']
    out: dict[str, Any] = {}
    for family, prefix in FAMILIES.items():
        members = [n for n in analysis['at_target'] if n.startswith(prefix)]
        resolved = [n for n in members if analysis['at_target'][n] is not None]
        dominating = {m: [n for n in decision['dominating'][m] if n.startswith(prefix)] for m in w.METRICS}
        outcome = 'unresolved' if not resolved else ('B' if any(dominating.values()) else 'A')
        out[family] = {'outcome': outcome, 'resolved': resolved, 'unresolved': sorted(set(members) - set(resolved)),
                       'dominating': dominating}
    return out


def margins(analysis: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    """(policy, metric, whole - hybrid, foreground - hybrid) for each resolved policy; the rule needs both <= 0.001."""
    hybrid = analysis['hybrid']
    rows = []
    for name, value in sorted(analysis['at_target'].items()):
        if value is None:
            continue
        for metric in w.METRICS:
            rows.append((name, metric, value[f'whole_{metric}'] - hybrid[metric]['whole'],
                         value[f'foreground_{metric}'] - hybrid[metric]['foreground']))
    return rows


def main() -> None:
    summary = json.load(open(sys.argv[1]))
    analysis = summary['analysis']
    by = analysis['by_policy']
    hybrid = summary['hybrid']['hybrid']
    print(summary['gpu'], len(summary['ticks']), 'ticks', round(summary['seconds'], 1), 's; window', summary['window'])
    print('controls', {k: max(v.values()) for k, v in summary['control_errors'].items()})
    print('hybrid', hybrid, 'KPS1', summary['hybrid']['kps1_kbps'], 'KPS2', summary['hybrid']['kps2_kbps'])
    print('| policy | CRF | kbps | whole LPIPS | fg LPIPS | whole DISTS | fg DISTS |')
    for name, rows in sorted(by.items()):
        for r in sorted(rows, key=lambda r: r['crf']):
            print(f"| {name} | {r['crf']} | {r['rate_kbps']:.3f} | " + ' | '.join(f'{r[k]:.4f}' for k in KEYS) + ' |')
    primary = analysis['rates']['primary']
    for label, block, target in ((primary + ' (primary)', analysis, analysis['target_kbps']),
                                 ('other rate', analysis['other_rate'], analysis['other_rate']['target_kbps'])):
        block = {**block, 'hybrid': hybrid}
        print(f"\n{label}: target {target:.4f} kbps, outcome {block['decision']['outcome']}")
        for family, v in family_outcomes(block).items():
            print('  ', family, v['outcome'], 'dominating', v['dominating'], 'unresolved', v['unresolved'])
        for name, metric, dw, df in margins(block):
            print(f'   {name:8s} {metric:5s} whole-hybrid {dw:+.4f}  fg-hybrid {df:+.4f}')
        for key, v in block['best_foreground'].items():
            print('   best foreground at hybrid whole', key, v['foreground_at_hybrid_whole'], 'hybrid', v['hybrid_foreground'])


if __name__ == '__main__':
    main()
