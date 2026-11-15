"""Recompute the prospectively frozen three-match comparison from saved scores.

No encoder, model or network execution is performed. Authenticate each complete
score file with its independently accepted SHA256 before using the command.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import ball_confirmation as physical
from experiments.tennis import ball_confirmation_score as scorer
from src.pointstream.evaluation import anchor_sensitivity

METRICS = ('crop_dists', 'crop_lpips', 'full_dists', 'full_lpips_standard',
           'full_lpips_spatial_map', 'rgb_psnr')
REFERENCE = {'native_svt', 'native_dcvc', 'roi_raw', 'roi_fitted'}


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def means(row: dict[str, Any], labels: list[tuple[int, float, float]]) -> dict[str, float]:
    frames = row['frames']
    if len(frames) != len(labels):
        raise ValueError('all frozen frames required')
    selected = [i for i, (kind, _, _) in enumerate(labels) if kind == 1]
    if not selected:
        raise ValueError('no supported crop endpoint')
    result = {}
    for name in METRICS[:-1]:
        indices = selected if name.startswith('crop_') else list(range(len(frames)))
        values = [frames[i][name] for i in indices]
        if any(not finite(x) or x < 0 for x in values):
            raise ValueError(f'invalid metric {name}')
        if name.startswith('crop_') and any(frames[i][name] is not None for i in range(len(frames))
                                            if i not in selected):
            raise ValueError('unknown/not-visible crop must be undefined')
        result[name] = math.fsum(values) / len(values)
    sse = sum(f['sse'] for f in frames)
    pixels = sum(f['pixels'] for f in frames)
    if sse <= 0 or pixels <= 0:
        raise ValueError('pooled PSNR requires positive error and pixel count')
    result['rgb_psnr'] = 10 * math.log10(255**2 * pixels / sse)
    return result


def analyse_source(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    source_id = record['source_id']
    labels = scorer.labels_for_source(manifest, source_id)
    checks = scorer.validate_scores(record, labels, 'full', len(labels))
    if not all(checks.values()):
        raise ValueError('incomplete or invalid full score: ' + str([k for k, v in checks.items() if not v]))
    if (record['manifest_sha256'] != scorer.SOURCE_MANIFEST
            or record['source_archive_sha256'] != scorer.SOURCE_ARCHIVE
            or record['frames'] != physical.SOURCES[source_id][0]):
        raise ValueError('source freeze differs')
    curves = []
    for row in record['rows']:
        curves.append({'family': row['family'], 'point': row['point'], 'bytes': row['coded_bytes'],
                       'packet_sha256': row['packet_sha256'], 'means': means(row, labels)})
    candidate = [r for r in curves if r['family'] == 'ptp']
    comparisons: dict[str, Any] = {}
    for name, families in [('reference', REFERENCE), ('native_svt', {'native_svt'}),
                           ('native_dcvc', {'native_dcvc'}), ('roi_raw', {'roi_raw'}),
                           ('roi_fitted', {'roi_fitted'}), ('streak', {'streak'})]:
        anchor = [r for r in curves if r['family'] in families]
        comparisons[name] = {}
        for metric in METRICS:
            sign = 1 if metric == 'rgb_psnr' else -1
            comparisons[name][metric] = anchor_sensitivity(
                [(r['bytes'], sign * r['means'][metric]) for r in candidate],
                [(r['bytes'], sign * r['means'][metric]) for r in anchor])
    counts = {str(kind): sum(v == kind for v, _, _ in labels) for kind in (-1, 0, 1)}
    return {'source_id': source_id, 'frames': len(labels), 'label_counts': counts,
            'physical_curves': curves, 'comparisons': comparisons,
            'sender_visibility': record['sender_visibility'],
            'source_identity': {k: record[k] for k in ('manifest_sha256', 'source_archive_sha256',
                               'base_archive_sha256', 'controls_archive_sha256', 'execution_identity', 'gpu')},
            'scope': '720-view sparse visible point crops; all-frame full metrics; possible encoder TRAIN overlap'}


def source_summary(sources: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Equal-match log ratios; all 27 ordered cluster resamples, never frames."""
    expected = set(physical.SOURCES)
    ids = [s['source_id'] for s in sources]
    if len(ids) != 3 or set(ids) != expected:
        raise ValueError('exactly the three frozen distinct source groups required')
    values = [s['comparisons']['reference'][metric]['bd_rate']['hull_pchip'] for s in sources]
    result: dict[str, Any] = {'source_ids': ids, 'source_bd_rate': values,
                              'replicate_unit': 'physical match', 'source_count': 3}
    if any(x is None for x in values):
        return {**result, 'status': 'inconclusive_unsupported_common_quality',
                'all_sources_improve': None, 'mean_bd_rate': None, 'bootstrap_95_interval': None,
                'direction_supported': False, 'bootstrap_resamples': []}
    if any(not finite(x) or x <= -1 for x in values):
        raise ValueError('invalid supported BD-rate')
    logs = [math.log1p(x) for x in values]
    resamples = [math.expm1(math.fsum(logs[i] for i in draw) / 3)
                 for draw in itertools.product(range(3), repeat=3)]
    interval = [float(x) for x in np.quantile(resamples, [.025, .975], method='linear')]
    all_improve = all(x < 0 for x in values)
    return {**result, 'status': 'supported', 'all_sources_improve': all_improve,
            'mean_bd_rate': math.expm1(math.fsum(logs) / 3),
            'bootstrap_95_interval': interval, 'bootstrap_resamples': resamples,
            'direction_supported': all_improve and interval[1] < 0,
            'interval_limit': 'Descriptive percentile interval from only three matches; no broad population claim.'}


def analyse(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    if len(records) != 3 or {r['source_id'] for r in records} != set(physical.SOURCES):
        raise ValueError('all three fixed sources required; no dropping unsupported groups')
    sources = [analyse_source(r, manifest) for r in sorted(records, key=lambda r: r['source_id'])]
    summary = {metric: source_summary(sources, metric) for metric in METRICS}
    return {'schema': 1, 'sources': sources, 'equal_source_summary': summary,
            'confirmation_primary_criterion_passed': summary['crop_dists']['direction_supported'],
            'interpretation': 'Criterion is narrow ball-priority confirmation, not distinctive novelty, whole foreground, absence, human preference or real time.',
            'corroboration': 'Crop LPIPS, all-frame costs and all interpolation sensitivities must accompany the primary decision.',
            'anchor_accounting': 'Twelve full native streams/ROI points; native DCVC benchmark timing/profile granted free. Candidate and streak pay complete packets.',
            'interpolation': 'Constructed linear-rate reference hull plus PCHIP; Pareto/PCHIP and literal linear-rate sensitivities retained; no physical switching claim.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--score', type=Path, action='append', required=True)
    parser.add_argument('--score-sha256', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.score) != 3 or len(args.score_sha256) != 3:
        parser.error('three independently accepted full scores and exact SHA256 values required')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != scorer.SOURCE_MANIFEST:
        parser.error('frozen source manifest hash differs')
    records = []
    provenance = []
    for path, expected in zip(args.score, args.score_sha256):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            parser.error('score SHA256 differs: ' + str(path))
        records.append(json.loads(raw))
        provenance.append({'path': str(path), 'sha256': expected})
    result = analyse(records, json.loads(args.manifest.read_bytes()))
    result['authenticated_score_inputs'] = provenance
    result['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'output': str(args.output), 'criterion_passed': result['confirmation_primary_criterion_passed']}))


if __name__ == '__main__':
    main()
