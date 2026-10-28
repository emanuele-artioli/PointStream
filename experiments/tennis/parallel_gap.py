"""CPU-only fixed-pixel gap analysis for the approved parallel development batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.k6b_side_feasibility import hull, member, side_ceiling

CLEAN_SHA = 'eea7147698b036c9182cee3ec635a57361854b049967febc753b03e979fc1ab5'
PRIOR_SHA = 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1'


def analyze(clean: Path, prior: Path) -> dict[str, Any]:
    for path, expected in ((clean, CLEAN_SHA), (prior, PRIOR_SHA)):
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Frozen publication identity changed')
    with tarfile.open(prior) as tar:
        row = json.load(member(tar, 'publish/quality.json'))['rows'][0]
        name = row['id'].replace('/', '__')
        with np.load(member(tar, f'publish/{name}/scores.npz')) as z:
            anchor = {m: z['full_' + m].copy() for m in ('lpips', 'dists')}
    with tarfile.open(clean) as tar:
        summary = json.load(member(tar, 'publish/quality.json'))
        with np.load(member(tar, 'publish/scores.npz')) as z:
            scores = {m: z[m].copy() for m in anchor}
            ticks = z['ticks'].tolist()
    if ticks != list(range(0, 240, 3)) or ticks != row['ticks']:
        raise ValueError('Scoring cohort changed')
    rate = float(summary['rate_kbps'])
    base = float(summary['base_kbps'])
    policies: dict[str, Any] = {}
    for i, policy in enumerate(summary['policies']):
        if not policy.startswith('clean_'):
            continue
        metrics = {}
        for metric in anchor:
            if scores[metric].shape != (4, 80) or not np.isfinite(scores[metric]).all():
                raise ValueError('Invalid scores')
            if not np.array_equal(scores[metric][:2], anchor[metric][[2, 3]]):
                raise ValueError('Historical control mismatch')
            curve = hull([(r['base_kbps'], -float(anchor[metric][2*j].mean()))
                          for j, r in enumerate(row['rates'])])
            if not curve[0][0] <= rate <= curve[-1][0]:
                raise ValueError('Rate outside measured anchor hull')
            target = -float(np.interp(rate, [r for r, _ in curve], [q for _, q in curve]))
            mean = float(scores[metric][i].mean())
            metrics[metric] = {
                'distance': mean, 'anchor_distance': target,
                'distance_reduction_for_strict_robust_win': max(0., mean - target + .001),
                'max_side_kbps_fixed_pixels': side_ceiling(curve, mean, base, .001),
            }
        ceilings = [v['max_side_kbps_fixed_pixels'] for v in metrics.values()]
        policies[policy] = {'metrics': metrics, 'joint_side_ceiling_kbps':
                           min(v for v in ceilings if v is not None)
                           if all(v is not None for v in ceilings) else None}
    return {'scope': 'Saved fixed-pixel accounting only; no new quality or universal bound.',
            'window': row['id'], 'ticks': ticks, 'current_total_kbps': rate,
            'clean_base_kbps': base, 'side_kbps': rate - base,
            'input_sha256': {'clean': CLEAN_SHA, 'prior': PRIOR_SHA}, 'policies': policies,
            'confirmation_manifest_status': 'NOT FROZEN: development selection must finish first; '
            'exclude all fitted, tuned and scored footage, audit training overlap and judge exposure, '
            'then freeze source hashes, timestamps, provenance and split before confirmation.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    record = analyze(args.clean, args.prior)
    record['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['wall_seconds'] = time.monotonic() - started
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'wall_seconds': record['wall_seconds'],
                      'policies': record['policies']}))


if __name__ == '__main__':
    main()
