"""Prospective descriptive decisions for the frozen native ROI/patch pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.pointstream.evaluation import anchor_sensitivity

METRICS = ('weighted_dists', 'weighted_lpips', 'full_dists', 'full_lpips', 'full_rgb_psnr')
ARMS = ('native', 'roi', 'semantic_patch', 'error_patch')


def analyse(score: dict[str, Any]) -> dict[str, Any]:
    rows = score['rows']
    count = score['frames']
    if type(count) is not int or count <= 0:
        raise ValueError('invalid registered frame count')
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row['arm'], row['crf'])
        if key in selected or row['arm'] not in ARMS or type(row['packet_bytes']) is not int or row['packet_bytes'] <= 0:
            raise ValueError('invalid/duplicate rate row')
        if any(type(row['means'].get(m)) not in (int, float) or not math.isfinite(row['means'][m]) for m in METRICS):
            raise ValueError('missing/nonfinite complete-frame metric')
        for metric in METRICS:
            values = row['frames'].get(metric)
            if (not isinstance(values, list) or len(values) != count
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or not math.isclose(math.fsum(values) / count, row['means'][metric], rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError('complete-frame values/mean do not match registered frames')
        selected[key] = row
    rates = {crf for arm, crf in selected if arm == 'native'}
    if len(rates) != 3 or set(selected) != {(arm, rate) for arm in ARMS for rate in rates}:
        raise ValueError('all four arms and three registered rates required')
    comparisons: dict[str, Any] = {}
    for arm in ARMS[1:]:
        by_anchor: dict[str, Any] = {}
        for anchor in ('native', 'roi'):
            if arm == anchor:
                continue
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                sign = 1 if metric == 'full_rgb_psnr' else -1
                candidate = [(selected[arm, r]['packet_bytes'], sign * selected[arm, r]['means'][metric]) for r in sorted(rates)]
                baseline = [(selected[anchor, r]['packet_bytes'], sign * selected[anchor, r]['means'][metric]) for r in sorted(rates)]
                metrics[metric] = anchor_sensitivity(candidate, baseline)
            by_anchor[anchor] = metrics
        comparisons[arm] = by_anchor
    # The price relaxation uses exactly the paid patch arm's measured pixels;
    # no new rendering or metric recomputation is inferred.
    optimistic: dict[str, Any] = {}
    for arm in ('semantic_patch', 'error_patch'):
        metrics = {}
        for metric in METRICS:
            sign = 1 if metric == 'full_rgb_psnr' else -1
            candidate = []
            for r in sorted(rates):
                row = selected[arm, r]
                bound = row['bound']['bytes']
                if type(bound) is not int or not 0 < bound <= row['packet_bytes']:
                    raise ValueError('invalid optimistic finite-family price')
                candidate.append((bound, sign * row['means'][metric]))
            baseline = [(selected['native', r]['packet_bytes'], sign * selected['native', r]['means'][metric]) for r in sorted(rates)]
            metrics[metric] = anchor_sensitivity(candidate, baseline)
        optimistic[arm] = metrics
    decisions: dict[str, Any] = {}
    for arm, anchors in comparisons.items():
        checks: dict[str, Any] = {}
        for anchor, metrics in anchors.items():
            dists = metrics['weighted_dists']['bd_rate']['hull_pchip']
            lpips = metrics['weighted_lpips']['bd_rate']['hull_pchip']
            checks[anchor] = {'weighted_dists_bd_rate': dists, 'weighted_lpips_bd_rate': lpips,
                              'development_target_met': None if dists is None or lpips is None else dists <= -.10 and lpips < 0}
        decisions[arm] = checks
    return {'schema': 1, 'scope': 'reference-guided static-rectangle development mechanism oracle; not automatic sender or confirmation',
            'domain': score['domain'], 'source_group': score['source_group'], 'annotation_provenance': score['annotation_provenance'],
            'frames': score['frames'], 'comparisons': comparisons, 'optimistic_same_pixels': optimistic,
            'decisions': decisions, 'rate_basis': 'complete packet bytes on a single identical-duration window; scale cancels in BD-rate',
            'promotion_limit': '10percent weightedDISTS plus corroboratingLPIPS is a development screen only; simplerROI/error selection and full/temporal outcomes must be reviewed; no population/neural-anchor claim'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = args.score.read_bytes()
    result = analyse(json.loads(data))
    result['score_sha256'] = hashlib.sha256(data).hexdigest()
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'domain': result['domain'], 'decisions': result['decisions'], 'output': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
