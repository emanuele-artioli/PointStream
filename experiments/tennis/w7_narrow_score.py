"""Five fixed W7 scores, gated by actual two-control smoke. No submission."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_cost_pilot as cost
from experiments.tennis import w1_roi as roi, w7_shared_score as shared, w7_specs

PUBLICATION = '55739efe3cbf6937b6251e26c2ed1322663f8d1d2d1ee5f36bfb1ace593cdb59'
PILOT = '588e8ccae59e99ad6667ba30cb277c3fbcd90d47fca060b8a6a5f844b57dffd4'
PILOT_AUDIT = '55fd444a5e56122e9744c100df0062b1408a1dfedff8e103ef16c21bc9aabafa'
HISTORICAL_REVIEW = '581b015e88bbb5d5ce6a5900cf8e3ac8ecc32d649d7f9887ee29893b8d19a18b'
CONTROLS = {('med000', 'control_s2_crf63'), ('med019', 'control_s2_crf63')}
NEW = {('med000', 's2_o20_crf47'), ('med019', 's2_o20_crf47'), ('med019', 's2_o20_crf70')}
PRIOR_SETUP_LAUNCH = 7.787541305180639 + 3.759566524066031
PRIOR_POINT = 88.30073950905353
CONTEXT = 20.0
FIXED = {'windows': '18d184cd39f6a4029a98e6fd9852d1e38f5cfde5c2ee19e5bc67bd69299c5e15',
         'state_med000': '96f8eee7bec42558fc1711a6f0ae03a85bffd0c1fd89235e291325455c5b9afb',
         'state_med019': 'a335f1ccfdec65fc63002533f5c37c21222e3d087f4cdd126bae9f31167dda0d',
         'inventory': shared.INVENTORY_SHA,
         'bindings': '2ee1dd1a03e898c63e64a8080cd8bc31f02328f8e4b1083b6dc2f34f09297b61',
         'byte_publication': PUBLICATION, 'pilot_archive': PILOT,
         'pilot_audit': PILOT_AUDIT, 'historical_review': HISTORICAL_REVIEW}
REQUIRED = {'inventory', 'bindings', 'windows', 'state_med000', 'state_med019', 'lpips', 'dists',
            'med000_source', 'med000_masks', 'med000_hybrid', 'med019_source', 'med019_masks', 'med019_hybrid',
            'med000_historical_spec', 'med000_historical_ready', 'med019_historical_spec', 'med019_historical_ready',
            'byte_publication', 'pilot_archive', 'pilot_audit', 'historical_review'}


def forecast(setup_launch: float, point: float, contexts: list[float], remaining: float, overhead: float = 20.0) -> dict[str, float]:
    values = [setup_launch, point, remaining, overhead, *contexts]
    if len(contexts) != 2 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid observed cost inputs')
    setup = max(PRIOR_SETUP_LAUNCH, setup_launch)
    worst = max(PRIOR_POINT, point)
    sources = sum(max(CONTEXT, c) for c in contexts)
    smoke = 1.5 * (setup + sources + 2 * worst)
    full = 1.5 * (setup + sources + 3 * worst)
    total = 1.5 * (2 * (setup + sources) + 5 * worst)
    staging = 2 * max(20.0, overhead)
    if smoke > 360 or full > 500 or total + staging + 20 + 50 > 1000 or total + staging + 20 + 150 > 1100:
        raise ValueError('five-point stage or inclusive cost exceeds reserved bounds')
    if full + max(20.0, overhead) + 20 > remaining:
        raise ValueError('insufficient actual campaign time remains for full and closure')
    return {'smoke': smoke, 'full': full, 'science': total, 'gpu_inclusive': total + staging + 70,
            'cpu_inclusive': total + staging + 170, 'staging': staging}


def selected(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    wanted = CONTROLS if stage == 'smoke' else NEW if stage == 'full' else set()
    out = [r for r in rows if (r['window'], r['name']) in wanted]
    if {(r['window'], r['name']) for r in out} != wanted or len(out) != len(wanted):
        raise ValueError('fixed five-point cohort differs')
    return out


def prepare(stage: Path, scratch: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    inputs = {r['name']: Path(r['path']) for r in json.loads((stage / 'staging.json').read_text())['inputs']}
    if set(inputs) != REQUIRED:
        raise ValueError('staged input coverage differs')
    spec = json.loads((stage.parent / 'spec.json').read_text())
    if (spec.get('device') != 'gpu' or spec.get('hosts') != ['gpu2', 'gpu3']
            or spec.get('gpu_models') != ['RTX A6000'] or spec.get('cpu_threads') != 1
            or spec.get('single_cpu_campaign') is not True or spec.get('budget_seconds') != 950
            or spec.get('smoke', {}).get('seconds') != 360 or spec.get('full', {}).get('seconds') != 500
            or spec.get('validator_seconds') != 20 or spec.get('inclusive_gpu_seconds') != 1000
            or spec.get('inclusive_cpu_seconds') != 1100 or spec.get('external_gpu_seconds') != 50
            or spec.get('external_cpu_seconds') != 150
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w7_narrow_score']
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w7_narrow_score', '--validate']):
        raise ValueError('fixed five-point campaign scope differs')
    declared = {r['name']: r['sha256'] for r in spec['staged_inputs']}
    if any(declared.get(name) != sha for name, sha in FIXED.items()):
        raise ValueError('frozen manifest, state or evidence identity differs')
    for name, path in inputs.items():
        if shared.old.digest(path) != declared[name]:
            raise ValueError('staged input hash differs: ' + name)
    if (declared['byte_publication'] != PUBLICATION or declared['pilot_archive'] != PILOT
            or declared['pilot_audit'] != PILOT_AUDIT or declared['historical_review'] != HISTORICAL_REVIEW):
        raise ValueError('proof or prior pilot identity differs')
    states = [json.loads(inputs['state_' + name].read_text()) for name in ('med000', 'med019')]
    merged = {name: state[name] for name, state in zip(('med000', 'med019'), states)}
    state_file = scratch / 'verified-window-state.json'
    state_file.write_text(json.dumps(merged, sort_keys=True))
    inputs['state'] = state_file
    rows = shared.inventory(inputs['inventory'], shared.INVENTORY_SHA)
    rows = shared.bound_rows(rows, json.loads(inputs['bindings'].read_text()))
    chosen = selected(rows, stage.name)
    inputs.update(shared.materialize_proofs(inputs['byte_publication'], PUBLICATION, chosen, scratch / 'proof-inputs'))
    return inputs, rows


def run(stage: Path, scratch: Path) -> None:
    inputs, rows = prepare(stage, scratch)
    roles = {(r['window'], r['name']) for r in selected(rows, stage.name)}
    args = argparse.Namespace(action='narrow5', byte_publication_sha256=PUBLICATION)
    shared.score(args, rows, inputs, pilot=False, selection=roles)


def validate(stage: Path, target: Path) -> None:
    import time
    try:
        staging = json.loads((stage / 'staging.json').read_text())
        execution = json.loads((stage / 'execution.json').read_text())
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        if execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False:
            raise ValueError('invalid smoke execution')
        with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as out:
            report = json.loads((out / 'score.json').read_text())
            events = json.loads((out / 'timing.json').read_text())['events']
            if report['passed'] is not True or {(p['window'], p['name']) for p in report['points']} != CONTROLS:
                raise ValueError('two fixed original controls required')
            for point in report['points']:
                path = out / f"{point['window']}-{point['name']}.npz"
                if shared.old.digest(path) != point['score_sha256'] or not point['control_errors']:
                    raise ValueError('control result missing')
                if max(point['control_errors'].values()) > roi.control_tolerance(point['window']):
                    raise ValueError('historical control tolerance failed')
                with np.load(path) as z:
                    if z['ticks'].tolist() != roi.TICKS or z['values'].shape != (4, 80) or not np.isfinite(z['values']).all():
                        raise ValueError('control arrays differ')
            sources = [e['wall_seconds'] for e in events if e['name'].startswith('source.') and e['completed']]
            points = [e['wall_seconds'] for e in events if e['name'].startswith('point.') and e['completed']]
            if len(sources) != 2 or len(points) != 2:
                raise ValueError('source and control timing coverage differs')
            setup = execution['seconds'] - sum(sources) - sum(points)
            clock = dispatch['campaign_clock']
            now = time.monotonic()
            if not clock['started'] <= clock['phase_started'] <= now < clock['budget_end']:
                raise ValueError('invalid campaign clock')
            actual_overhead = max(0.001, now - clock['started'] - execution['seconds'])
            costs = forecast(setup, max(points), sources, clock['budget_end'] - now, actual_overhead)
        target.write_text(json.dumps({'passed': True, 'checks': {'two_controls': True, 'cost_forecast': True},
                                      'forecast': costs, 'scope': 'three predeclared remaining scores only'}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'two_controls': False}, 'error': str(exc)}) + '\n')
        raise



def make_spec(assets: dict[str, dict[str, str]], deadline: str) -> dict[str, Any]:
    """Render only from complete, actually hashed assets and the accepted old evidence."""
    import tempfile
    from experiments.jobs.inbox import validate_spec
    if set(assets) != REQUIRED:
        raise ValueError('all frozen evidence and both source contexts are required')
    for name, entry in assets.items():
        cf.asset(name, entry['remote'], entry['sha256'], Path(entry['local']), extract=False)
    expected = {'byte_publication': PUBLICATION, 'pilot_archive': PILOT,
                'pilot_audit': PILOT_AUDIT, 'historical_review': HISTORICAL_REVIEW}
    expected.update(FIXED)
    if any(assets[n]['sha256'] != sha for n, sha in expected.items()):
        raise ValueError('required actual historical evidence differs')
    review = json.loads(Path(assets['historical_review']['local']).read_text())
    audit = json.loads(Path(assets['pilot_audit']['local']).read_text())
    if (review.get('approved_historical_reuse') is not True or review.get('approved_full22') is not False
            or review.get('delivery_publication_sha256') != PUBLICATION
            or audit.get('job_id') != '20261105T055056Z-aa3070f0' or audit.get('smoke_published_sha256') != PILOT):
        raise ValueError('historical decision or prior pilot receipt differs')
    forecast(PRIOR_SETUP_LAUNCH, PRIOR_POINT, [CONTEXT, CONTEXT], 950)
    rows = shared.inventory(Path(assets['inventory']['local']), assets['inventory']['sha256'])
    rows = shared.bound_rows(rows, json.loads(Path(assets['bindings']['local']).read_text()))
    for window in ('med000', 'med019'):
        roi.preload_windows(['--windows-file', assets['windows']['local'], '--window-state', assets['state_' + window]['local']])
        shared.check_hybrid_state(roi.WINDOWS[window], assets[window + '_hybrid']['sha256'],
                                  Path(assets[window + '_historical_spec']['local']),
                                  Path(assets[window + '_historical_ready']['local']))
    chosen = selected(rows, 'smoke') + selected(rows, 'full')
    with tempfile.TemporaryDirectory(prefix='w7-narrow-spec-') as tmp:
        proof_inputs = shared.materialize_proofs(Path(assets['byte_publication']['local']), PUBLICATION,
                                                 chosen, Path(tmp) / 'proofs')
        for row in chosen:
            w = row['window']
            shared.check_point(row, json.loads(proof_inputs[row['proof_asset']].read_text()),
                json.loads(proof_inputs[row['validation_asset']].read_text()), proof_inputs[row['mkv_asset']],
                roi.WINDOWS[w], assets[w + '_source']['sha256'], assets[w + '_masks']['sha256'],
                assets[w + '_hybrid']['sha256'], shared.old.digest(Path(shared.old.__file__)), shared.source_bridge())
    files = {name: (entry['remote'], entry['sha256']) for name, entry in assets.items()}
    spec = w7_specs.common('experiments.tennis.w7_narrow_score', ['--stage', '{stage}'], files)
    spec.update({'device': 'gpu', 'hosts': ['gpu2', 'gpu3'], 'gpu_models': ['RTX A6000'],
                 'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True,
                 'local_storage_gib': 30, 'deadline': deadline,
                 'scale': {'stage': {'smoke': 'smoke', 'full': 'full'}},
                 'smoke': {'seconds': 360, 'representative_basis': 'Both original scale-2 controls, 80 ticks, two source contexts and shared models.'},
                 'full': {'seconds': 500}, 'validator_seconds': 20, 'budget_seconds': 950,
                 'validator': ['{python}', '-m', 'experiments.tennis.w7_narrow_score', '--validate'],
                 'inclusive_gpu_seconds': 1000, 'inclusive_cpu_seconds': 1100,
                 'external_gpu_seconds': 50, 'external_cpu_seconds': 150})
    validate_spec(spec)
    return spec

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'full'))
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    elif args.stage == stage.name:
        run(stage, Path(os.environ['PS_SCRATCH_DIR']))
    else:
        raise ValueError('stage mismatch')


if __name__ == '__main__':
    main()
