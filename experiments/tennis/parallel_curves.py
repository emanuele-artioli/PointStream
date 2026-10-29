"""Post-hoc PCHIP BD-rate from audited curves; never extrapolate quality."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from experiments.tennis.k6b_side_feasibility import hull
from experiments.tennis.parallel_audit import PRIOR_SHA


def compare(points: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    test: list[tuple[float, float]] = []
    for rate, quality in sorted(points):
        if rate <= 0 or not math.isfinite(rate) or not math.isfinite(quality):
            raise ValueError('Finite positive rates and finite quality required')
        if not test or quality > test[-1][1]:
            test.append((rate, quality))
    envelope = hull(anchor)
    low, high = max(test[0][1], envelope[0][1]), min(test[-1][1], envelope[-1][1])
    result: dict[str, Any] = {'test_pareto': test, 'anchor_hull': envelope,
                              'quality_overlap_negative_distance': [low, high], 'bd_rate_percent': None}
    if len(test) < 2 or high <= low:
        result['status'] = 'NO QUALITY OVERLAP; no BD-rate or equal-rate win inferred'
        return result

    def integral(p: list[tuple[float, float]]) -> float:
        return float(PchipInterpolator([q for _, q in p], [math.log(r) for r, _ in p]).integrate(low, high))

    result.update(status='OVERLAP ONLY', bd_rate_percent=100 * math.expm1((integral(test)-integral(envelope))/(high-low)))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.prior.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != PRIOR_SHA:
            raise ValueError('Frozen anchor publication changed')
    audited = json.loads(a.audit.read_text())
    with tarfile.open(a.prior) as t:
        metadata = t.extractfile('publish/quality.json')
        raw = t.extractfile('publish/alcaraz_hurkacz__031@129/scores.npz')
        if metadata is None or raw is None:
            raise ValueError('Missing anchor data')
        rates = json.load(metadata)['rows'][0]['rates']
        with np.load(io.BytesIO(raw.read())) as z:
            anchors = {m: [(r['base_kbps'], -float(z['full_'+m][2*i].mean())) for i, r in enumerate(rates)]
                       for m in ('lpips', 'dists')}
    result: dict[str, Any] = {'scope': 'Post-hoc Al development curves versus three original SVT encodes only; no neural baseline or decision-rule change.',
                              'audit_sha256': hashlib.sha256(a.audit.read_bytes()).hexdigest(), 'prior_sha256': PRIOR_SHA,
                              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ('background', 'motion', 'pixels'):
        rows = audited[name]['rows'][0]['points'] if name == 'pixels' else audited[name]['rows']
        result[name] = {m: compare([(r['rate_kbps'], -r['mean'][m]) for r in rows], anchors[m]) for m in anchors}
        for m in anchors:
            result[name][m]['candidate_distance_sensitivity_bd_percent'] = {
                str(offset): compare([(r['rate_kbps'], -(r['mean'][m]+offset)) for r in rows], anchors[m])['bd_rate_percent']
                for offset in (-.001, 0., .001)}
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({name: {m: result[name][m]['bd_rate_percent'] for m in anchors}
                      for name in ('background', 'motion', 'pixels')}))


if __name__ == '__main__':
    main()
