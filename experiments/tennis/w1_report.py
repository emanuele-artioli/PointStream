"""Print the W1 table and the rule at the primary and secondary rates from a score job's summary.

python -m experiments.tennis.w1_report score-summary.json [NEW_KEYPOINT_PACKET_BYTES]
"""
from __future__ import annotations

import json
import sys

from experiments.tennis import w1_roi as w

KEYS = ('whole_lpips', 'foreground_lpips', 'whole_dists', 'foreground_dists')


def main() -> None:
    summary = json.load(open(sys.argv[1]))
    by = summary['analysis']['by_policy']
    print(summary['gpu'], len(summary['ticks']), 'ticks', round(summary['seconds'], 1), 's')
    print('controls', {k: max(v.values()) for k, v in summary['control_errors'].items()})
    print('| policy | CRF | kbps | whole LPIPS | fg LPIPS | whole DISTS | fg DISTS |')
    for name, rows in sorted(by.items()):
        for r in sorted(rows, key=lambda r: r['crf']):
            print(f"| {name} | {r['crf']} | {r['rate_kbps']:.3f} | " + ' | '.join(f'{r[k]:.4f}' for k in KEYS) + ' |')
    scales = tuple(sorted({int(n.split('_')[0][1:]) for n in by}))
    targets = [('primary', w.TARGET_KBPS)] + ([('secondary', w.secondary_kbps(int(sys.argv[2])))] if len(sys.argv) > 2 else [])
    for label, target in targets:
        a = w.analyse(by, scales, target)
        print(f"\n{label} target {target:.6f} kbps: outcome {a['decision']['outcome']}, dominating {a['decision']['dominating']}, unresolved {a['decision']['unresolved']}")
        for name, v in sorted(a['at_target'].items()):
            print('  ', name, None if v is None else ' '.join(f'{v[k]:.4f}' for k in KEYS))
        for key, v in a['best_foreground'].items():
            print('   best foreground at hybrid whole', key, v['foreground_at_hybrid_whole'], 'hybrid', v['hybrid_foreground'])


if __name__ == '__main__':
    main()
