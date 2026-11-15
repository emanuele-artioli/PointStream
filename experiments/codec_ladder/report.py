"""Generate paper tables from hash-bound, independently recomputed run records.

python -m experiments.codec_ladder.report --record ANALYSIS.json SCORES.json ... --out NEW_DIR
Only the separately named standard full-frame LPIPS appears in that column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.codec_ladder.analysis import analyse

METRICS = ('weighted_dists', 'weighted_lpips', 'full_lpips_standard')
LABELS = {'roi': 'Conventional ROI', 'semantic_patch': 'Semantic patch', 'error_patch': 'Error patch'}


def verified_record(path: Path, score: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(path.read_text())
    data = score.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['score_sha256']:
        raise ValueError('saved score hash differs from accepted record')
    result = analyse(json.loads(data))
    for key in ('comparisons', 'decisions', 'domain', 'source_group', 'frames', 'rate_basis'):
        if json.dumps(result[key], sort_keys=True, allow_nan=False) != json.dumps(record[key], sort_keys=True, allow_nan=False):
            raise ValueError(f'saved analysis differs from recomputed score: {key}')
    evidence = record['evidence']
    if not evidence.get('score_job') or not evidence.get('score_archive_sha256'):
        raise ValueError('paper table requires a named accepted job and score archive')
    return result, {'record': str(path.resolve()), 'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'score_file': str(score.resolve()), 'score_sha256': record['score_sha256'], 'evidence': evidence}


def table_rows(paths: list[tuple[Path, Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, provenance = [], []
    seen = set()
    for path, score in paths:
        result, identity = verified_record(path, score)
        key = (result['domain'], result['source_group'])
        if key in seen:
            raise ValueError('duplicate source group cannot count as a new paper result')
        seen.add(key)
        provenance.append(identity)
        for arm, label in LABELS.items():
            row: dict[str, Any] = {'domain': result['domain'], 'source_group': result['source_group'],
                                   'frames': result['frames'], 'mode': label}
            metrics = result['comparisons'][arm]['native']
            for metric in METRICS:
                value = metrics.get(metric, {}).get('bd_rate', {}).get('hull_pchip')
                row[metric + '_percent'] = None if value is None else 100 * value
            row['development_target_met'] = result['decisions'][arm]['native']['development_target_met']
            rows.append(row)
    return rows, provenance


def number(value: float | None) -> str:
    return 'unsupported' if value is None else f'{value:+.2f}'


def write_tables(paths: list[tuple[Path, Path]], output: Path) -> dict[str, Any]:
    rows, sources = table_rows(paths)
    output.mkdir(parents=True, exist_ok=False)
    fields = ['domain', 'source_group', 'frames', 'mode', *(m + '_percent' for m in METRICS), 'development_target_met']
    with (output / 'table.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [r'\begin{tabular}{llrrrl}', r'\hline',
             r'Domain & Mode & Weighted DISTS & Weighted LPIPS & Standard full LPIPS & Target \\', r'\hline']
    for row in rows:
        # The visible strings are fixed protocol labels, not source identifiers.
        domain = {'tennis': 'Tennis', 'egocentric': 'Egocentric'}[row['domain']]
        decision = {True: 'met', False: 'not met', None: 'unsupported'}[row['development_target_met']]
        values = [domain, row['mode'], *(number(row[m + '_percent']) for m in METRICS), decision]
        lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}',
                  '% BD-rate percentages versus complete native IVF; candidates pay complete packets.',
                  '% Predeclared hull/PCHIP over common-quality support; unsupported is never extrapolated.',
                  '% Source-mask 70/30 regional objectives; standard scalar full-frame LPIPS separately named.',
                  '% Exposed development windows, not confirmation or population-level claims.'])
    (output / 'table.tex').write_text('\n'.join(lines) + '\n')
    record = {'schema': 1, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'descriptive source-window development tables, no cross-source pooling',
              'sources': sources, 'rows': rows,
              'outputs': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ('table.csv', 'table.tex')}}
    (output / 'provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, nargs=2, metavar=('ANALYSIS', 'SCORES'), action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = write_tables(args.record, args.out)
    print(json.dumps({'rows': len(result['rows']), 'output': str(args.out), 'outputs': result['outputs']}))


if __name__ == '__main__':
    main()
