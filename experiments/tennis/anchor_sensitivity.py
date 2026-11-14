"""Recompute anchor geometry from saved scalars; no codec or metric execution.

Authenticate the accepted primary report with its tracked acceptance index::

    python -m experiments.tennis.anchor_sensitivity --acceptance INDEX.json \
        --accepted-primary-report REPORT.json --output NEW.json

The output embeds the selected oriented scalar curves. Recompute them later::

    python -m experiments.tennis.anchor_sensitivity --input SAVED.json --output NEW.json

The latter verifies arithmetic, not the source authenticity of arbitrary input.
Existing output paths are refused. Missing primary data is reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import scipy

from src.pointstream import evaluation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_curves(acceptance: Path, report: Path) -> dict[str, Any]:
    """Authenticate retained posthoc report using the ROOT acceptance index."""
    index = json.loads(acceptance.read_bytes())
    expected = index['posthoc_curves']['report_sha256']
    provenance = {'acceptance_sha256': digest(acceptance), 'accepted_report_sha256': expected,
                  'job_id': index['job_id'], 'GPU': index['GPU'],
                  'execution_identity': index['execution_identity'],
                  'acceptance_state': index['state']}
    if not report.exists():
        return {'curves': [], 'provenance': provenance, 'status': 'authenticated_scalars_unavailable'}
    if digest(report) != expected:
        raise ValueError('saved primary report SHA256 does not match acceptance index')
    saved = json.loads(report.read_bytes())
    if saved['job_id'] != index['job_id']:
        raise ValueError('primary job identity mismatch')
    curves = []
    for metric, comparisons in saved['metrics'].items():
        row = comparisons['all_test_vs_all_native_anchor']
        curves.append({'metric': metric, 'candidate': row['test_raw_points'],
                       'anchor': row['anchor_raw_points'],
                       'discrete': metric in {'WASB_recall10', 'BallTrack_recall10_diagnostic'},
                       'accepted_hull_pchip_percent': row['BD_rate_percent']})
    return {'curves': curves, 'provenance': provenance,
            'scope': saved['primary_scope'], 'status': 'authenticated_saved_scalars'}


def recompute(data: dict[str, Any]) -> dict[str, Any]:
    """Retain raw points and provenance so a published result is self-contained."""
    results: dict[str, Any] = {}
    for curve in data['curves']:
        metric = curve['metric']
        if metric in results:
            raise ValueError('duplicate metric')
        result = evaluation.anchor_sensitivity(curve['candidate'], curve['anchor'],
                                              discrete=curve['discrete'])
        accepted = curve.get('accepted_hull_pchip_percent')
        actual = result['bd_rate']
        assert isinstance(actual, dict)
        if accepted is not None:
            if actual['hull_pchip'] is None or not math.isclose(100 * actual['hull_pchip'], accepted,
                                                             abs_tol=1e-9, rel_tol=1e-10):
                raise ValueError(f'{metric}: historical PCHIP result did not reproduce')
        results[metric] = result
    return {'schema': 1, 'status': data.get('status', 'unverified_input_scalars'),
            'provenance': data.get('provenance', {}), 'scope': data.get('scope'),
            'curves': data['curves'], 'results': results,
            'software': {'scipy': scipy.__version__, 'evaluation_sha256': digest(Path(evaluation.__file__)),
                         'cli_sha256': digest(Path(__file__))},
            'new_codec_or_scoring_execution': False,
            'interpretation': 'Constructed reference sensitivity only; not physical switching or new independent footage.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--accepted-primary-report', type=Path)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; choose a new result path')
    if args.accepted_primary_report is not None:
        if args.acceptance is None:
            parser.error('--accepted-primary-report requires --acceptance')
        data = primary_curves(args.acceptance, args.accepted_primary_report)
    else:
        data = json.loads(args.input.read_bytes())
    result = recompute(data)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'metrics': len(result['results']),
                      'output': str(args.output), 'output_sha256': digest(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
