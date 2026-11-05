"""Build and validate a bounded two-point W7 pilot from actual verified assets.

No submission is performed. A separate reviewed full-22 receipt and spec are
required after a successful pilot; this adapter cannot authorize that run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf, k5d_cost_pilot as cost
from experiments.tennis import w1_roi as roi, w7_shared_score as worker, w7_specs


def pilot_spec(*, assets: dict[str, dict[str, str]], deadline: str, smoke_seconds: int,
               campaign_seconds: int, gpu_budget_seconds: int, cpu_budget_seconds: int) -> dict[str, Any]:
    if (type(smoke_seconds) is not int or not 0 < smoke_seconds <= 600
            or type(campaign_seconds) is not int or campaign_seconds < smoke_seconds + 60
            or type(gpu_budget_seconds) is not int or type(cpu_budget_seconds) is not int
            or gpu_budget_seconds > 350 or cpu_budget_seconds > 450
            or campaign_seconds + 50 > gpu_budget_seconds or campaign_seconds + 100 > cpu_budget_seconds):
        raise ValueError('invalid inclusive pilot bounds')
    for name, entry in assets.items():
        cf.asset(name, entry['remote'], entry['sha256'], Path(entry['local']), extract=False)
    for name in ('inventory', 'bindings', 'windows', 'state', 'lpips', 'dists', 'med000_source',
                 'med000_masks', 'med000_hybrid', 'byte_publication', 'historical_spec', 'historical_ready'):
        if name not in assets:
            raise ValueError('missing pilot asset: ' + name)
    rows = worker.inventory(Path(assets['inventory']['local']), assets['inventory']['sha256'])
    bindings = json.loads(Path(assets['bindings']['local']).read_text())
    rows = worker.bound_rows(rows, bindings)
    roi.preload_windows(['--windows-file', assets['windows']['local'], '--window-state', assets['state']['local']])
    chosen = [r for r in rows if r['window'] == 'med000' and
              ((r['kind'] == 'control' and r['scale'] == 8) or (r['kind'] == 'new' and r['crf'] == 70))]
    if len(chosen) != 2 or 'med000' not in roi.WINDOWS:
        raise ValueError('pilot control and new point are required')
    if assets['byte_publication']['sha256'] != '55739efe3cbf6937b6251e26c2ed1322663f8d1d2d1ee5f36bfb1ace593cdb59':
        raise ValueError('pilot requires the independently reviewed 14f7 full byte-proof publication')
    worker.check_hybrid_state(roi.WINDOWS['med000'], assets['med000_hybrid']['sha256'], Path(assets['historical_spec']['local']), Path(assets['historical_ready']['local']))
    if any(worker.old.digest(Path(assets[n]['local'])) != roi.WEIGHTS[n] for n in ('lpips', 'dists')):
        raise ValueError('model weights differ')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='w7-score-spec-') as temp:
        extracted = worker.materialize_proofs(Path(assets['byte_publication']['local']),
                    assets['byte_publication']['sha256'], chosen, Path(temp) / 'proofs')
        for row in chosen:
            proof = json.loads(extracted[row['proof_asset']].read_text())
            worker.check_point(row, proof, json.loads(extracted[row['validation_asset']].read_text()),
                               extracted[row['mkv_asset']], roi.WINDOWS['med000'],
                               assets['med000_source']['sha256'], assets['med000_masks']['sha256'],
                               assets['med000_hybrid']['sha256'], worker.old.digest(Path(worker.old.__file__)), worker.source_bridge())
    files = {name: (entry['remote'], entry['sha256']) for name, entry in assets.items()}
    args = ['pilot', '--items', '{items}', '--byte-publication-sha256', assets['byte_publication']['sha256'], '--inventory', '{staged:inventory}', '--inventory-sha256', assets['inventory']['sha256'],
            '--bindings', '{staged:bindings}', '--bindings-sha256', assets['bindings']['sha256']]
    spec = w7_specs.common('experiments.tennis.w7_shared_fleet', args, files)
    spec.update({'device': 'gpu', 'hosts': ['gpu2', 'gpu3'], 'gpu_models': ['RTX A6000'],
                 'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True,
                 'local_storage_gib': 30, 'deadline': deadline, 'scale': {'items': {'smoke': 2, 'full': 0}},
                 'smoke': {'seconds': smoke_seconds, 'representative_basis': 'Complete 80-tick med000 control and new CRF70, one shared model load.'},
                 'full': {'seconds': 20}, 'validator_seconds': 20, 'budget_seconds': campaign_seconds,
                 'validator': ['{python}', '-m', 'experiments.tennis.w7_shared_fleet', 'validate']})
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec


def run(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    inputs = {r['name']: Path(r['path']) for r in json.loads((stage / 'staging.json').read_text())['inputs']}
    rows = worker.inventory(args.inventory, args.inventory_sha256)
    if worker.old.digest(args.bindings) != args.bindings_sha256:
        raise ValueError('point bindings identity mismatch')
    rows = worker.bound_rows(rows, json.loads(args.bindings.read_text()))
    if args.action == 'pilot' and args.byte_publication_sha256 != '55739efe3cbf6937b6251e26c2ed1322663f8d1d2d1ee5f36bfb1ace593cdb59':
        raise ValueError('pilot byte-proof publication identity differs')
    if args.action == 'full-study':
        if args.full_gate is None or worker.old.digest(args.full_gate) != args.full_gate_sha256:
            raise ValueError('full gate identity mismatch')
        gate = json.loads(args.full_gate.read_text())
        if (gate.get('approved') is not True or gate.get('purpose') != 'w7-full22-scoring'
                or gate.get('inventory_sha256') != args.inventory_sha256
                or gate.get('execution_identity') != cp.execution_identity()
                or gate.get('pilot_archive_sha256') != worker.old.digest(inputs['pilot_archive'])
                or gate.get('byte_publication_sha256') != args.byte_publication_sha256
                or gate.get('byte_audit_sha256') != worker.old.digest(inputs['byte_audit'])):
            raise ValueError('full gate, frozen source or pilot identity differs')
        for window in worker.WINDOWS:
            if worker.old.digest(inputs['source_bound_' + window]) != gate['source_bound_sha256'][window]:
                raise ValueError('source bound identity mismatch')
    if args.action == 'pilot' and os.environ['PS_STAGE'] == 'smoke':
        selected = [r for r in rows if r['window'] == 'med000' and ((r['kind'] == 'control' and r['scale'] == 8) or (r['kind'] == 'new' and r['crf'] == 70))]
        inputs.update(worker.materialize_proofs(inputs['byte_publication'], args.byte_publication_sha256, selected, scratch / 'proof-inputs'))
    expected_items = {'smoke': 2, 'full': 22 if args.action == 'full-study' else 0}.get(os.environ['PS_STAGE'])
    if args.items != expected_items:
        raise ValueError('stage point scope differs')
    if os.environ['PS_STAGE'] == 'smoke':
        worker.score(args, rows, inputs, pilot=True)
    elif os.environ['PS_STAGE'] == 'full' and args.action == 'full-study':
        worker.score(args, rows, inputs, pilot=False)
    elif os.environ['PS_STAGE'] == 'full':
        output = scratch / 'publish'
        output.mkdir()
        cp.write_new(output / 'no-op.json', {'scope': 'two-point smoke only; full scoring not admitted'})
    else:
        raise ValueError('unknown stage')


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as out:
            record = json.loads((out / 'score.json').read_text())
            if (record['pilot'] is not True or record['window'] != 'med000' or record['point_count'] != 2
                    or record['ticks'] != roi.TICKS or len(record['points']) != 2
                    or {(p['window'], p['name']) for p in record['points']} != {('med000', 'control_s8_crf63'), ('med000', 's2_o20_crf70')}
                    or sum(p['kind'] == 'control' for p in record['points']) != 1
                    or not record['cost']['wall']['points'] or not record['cost']['cpu']['points']):
                raise ValueError('pilot output incomplete')
            for point in record['points']:
                path = out / f"{point['window']}-{point['name']}.npz"
                if worker.old.digest(path) != point['score_sha256']:
                    raise ValueError('score file identity differs')
                with np.load(path) as arrays:
                    if (arrays['ticks'].tolist() != roi.TICKS or arrays['values'].shape != (4, 80)
                            or not np.isfinite(arrays['values']).all()):
                        raise ValueError('score tick or metric coverage differs')
                if point['kind'] == 'control' and (not point['control_errors']
                        or max(point['control_errors'].values()) > roi.control_tolerance('med000')):
                    raise ValueError('historical control tolerance failed')
        target.write_text(json.dumps({'passed': True, 'checks': {'two_complete_points': True},
                                       'scope': 'pilot only; no full-study admission'}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'pilot': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('pilot', 'full-study', 'validate'))
    parser.add_argument('--items', type=int)
    parser.add_argument('--byte-publication-sha256')
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--inventory-sha256')
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--bindings-sha256')
    parser.add_argument('--full-gate', type=Path)
    parser.add_argument('--full-gate-sha256')
    args = parser.parse_args()
    if args.action == 'validate':
        validate()
    elif any(getattr(args, k) is None for k in ('inventory', 'inventory_sha256', 'bindings', 'bindings_sha256', 'byte_publication_sha256', 'items')):
        parser.error('pilot requires inventory and bindings identities')
    elif args.action == 'full-study' and (args.full_gate is None or args.full_gate_sha256 is None):
        parser.error('full study requires a SHA-bound admission gate')
    else:
        run(args)


def full_spec(*, assets: dict[str, dict[str, str]], deadline: str, pilot_job_dir: Path,
              smoke_seconds: int, full_seconds: int, campaign_seconds: int) -> dict[str, Any]:
    """Render full scoring only from actual pilot, 22 proofs, and explicit root gate.

    Each source-bound receipt must be independently reviewed and name its actual
    measurement; a two-point med000 sample alone is never a six-window bound.
    """
    from experiments.jobs.inbox import digest, validate_spec

    if any(type(x) is not int or x <= 0 for x in (smoke_seconds, full_seconds, campaign_seconds)):
        raise ValueError('positive integer stage caps required')
    if smoke_seconds > 600 or full_seconds > 2400 or campaign_seconds < smoke_seconds + full_seconds + 20:
        raise ValueError('full score stage bounds invalid')
    for name, entry in assets.items():
        cf.asset(name, entry['remote'], entry['sha256'], Path(entry['local']), extract=False)
    for name in ('inventory', 'bindings', 'windows', 'state', 'lpips', 'dists', 'full_gate', 'pilot_archive', 'byte_publication', 'byte_audit'):
        if name not in assets:
            raise ValueError('missing full-score asset: ' + name)
    gate = json.loads(Path(assets['full_gate']['local']).read_text())
    if (gate.get('schema') != 1 or gate.get('approved') is not True or gate.get('purpose') != 'w7-full22-scoring'
            or gate.get('inventory_sha256') != assets['inventory']['sha256']
            or gate.get('pilot_archive_sha256') != assets['pilot_archive']['sha256']
            or gate.get('byte_publication_sha256') != assets['byte_publication']['sha256']
            or gate.get('byte_audit_sha256') != assets['byte_audit']['sha256']
            or gate.get('gpu_budget_seconds') != 1800 or gate.get('cpu_budget_seconds') != 1670):
        raise ValueError('full-score admission receipt mismatch')
    byte_audit = json.loads(Path(assets['byte_audit']['local']).read_text())
    if (byte_audit.get('audit_full_passed') is not True or byte_audit.get('logical_roles') != 75
            or byte_audit.get('publication_sha256') != assets['byte_publication']['sha256']):
        raise ValueError('independently audited full byte-proof publication required')
    identity = gate['execution_identity']
    if (set(identity) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}
            or any(not isinstance(v, str) or not cf.digest_string(v) for v in identity.values())):
        raise ValueError('invalid full-score execution identity')
    rows = worker.inventory(Path(assets['inventory']['local']), assets['inventory']['sha256'])
    rows = worker.bound_rows(rows, json.loads(Path(assets['bindings']['local']).read_text()))
    roi.preload_windows(['--windows-file', assets['windows']['local'], '--window-state', assets['state']['local']])
    for row in rows:
        window = row['window']
        for required_asset in (window + '_source', window + '_masks', window + '_hybrid', row.get('proof_asset'), row.get('validation_asset'), row.get('mkv_asset')):
            if not isinstance(required_asset, str) or required_asset not in assets:
                raise ValueError('all 22 actual point proofs and source assets required')
        proof = json.loads(Path(assets[row['proof_asset']]['local']).read_text())
        worker.check_point(row, proof, json.loads(Path(assets[row['validation_asset']]['local']).read_text()),
                           Path(assets[row['mkv_asset']]['local']), roi.WINDOWS[window],
                           assets[window + '_source']['sha256'], assets[window + '_masks']['sha256'],
                           assets[window + '_hybrid']['sha256'], worker.old.digest(Path(worker.old.__file__)), worker.source_bridge())
    worker.check_proof_publication(Path(assets['byte_publication']['local']), assets['byte_publication']['sha256'],
                                   rows, {name: Path(entry['local']) for name, entry in assets.items()}, require_complete=True)
    if any(worker.old.digest(Path(assets[name]['local'])) != roi.WEIGHTS[name] for name in ('lpips', 'dists')):
        raise ValueError('model weights differ')
    job_id = gate['pilot_job_id']
    if pilot_job_dir.name != job_id:
        raise ValueError('pilot job identity differs')
    records = {}
    for name, path in (('spec', pilot_job_dir / 'spec.json'), ('dispatch', pilot_job_dir / 'smoke' / 'dispatch.json'),
                       ('staging', pilot_job_dir / 'smoke' / 'staging.json'), ('execution', pilot_job_dir / 'smoke' / 'execution.json'),
                       ('gate', pilot_job_dir / 'gate.json')):
        if worker.old.digest(path) != gate['pilot_records_sha256'][name]:
            raise ValueError('pilot fleet record hash differs: ' + name)
        records[name] = json.loads(path.read_text())
    spec, dispatch, staging, execution = (records[k] for k in ('spec', 'dispatch', 'staging', 'execution'))
    code = dispatch['code']
    if (code['spec_sha256'] != digest(spec) or code['source_sha256'] != identity['source_sha256']
            or code['snapshot_sha256'] != identity['snapshot_sha256']
            or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
            or staging['published']['sha256'] != assets['pilot_archive']['sha256']
            or Path(staging['published']['path']).parts[-3:] != (job_id, 'smoke', 'published.tar')
            or execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
            or records['gate'].get('passed') is not True or spec['gpu_models'] != ['RTX A6000']
            or 'RTX A6000' not in dispatch['environment']['gpu']['name']
            or spec['entrypoint'] != ['-m', 'experiments.tennis.w7_shared_fleet']
            or spec['arguments'][0] != 'pilot'):
        raise ValueError('pilot provenance, class or gate differs')
    with cost.unpack(Path(assets['pilot_archive']['local']), assets['pilot_archive']['sha256']) as out:
        pilot = json.loads((out / 'score.json').read_text())
        if (pilot.get('source_identity') != code or pilot.get('passed') is not True
                or pilot.get('point_count') != 2 or pilot.get('window') != 'med000'
                or pilot.get('pilot') is not True or pilot['cost']['wall']['pilot_total'] > execution['seconds']):
            raise ValueError('pilot output or elapsed measurement differs')
        events = json.loads((out / 'timing.json').read_text())['events']
        for kind, fields in (('wall', ('wall_seconds',)), ('cpu', ('self_cpu_seconds', 'waited_children_cpu_seconds'))):
            source = [e for e in events if e['name'] == 'source.med000' and e['completed'] is True]
            points = [e for e in events if e['name'].startswith('point.med000.') and e['completed'] is True]
            if (len(source) != 1 or len(points) != 2
                    or pilot['cost'][kind]['source'] != sum(source[0][f] for f in fields)
                    or pilot['cost'][kind]['points'] != [sum(e[f] for f in fields) for e in points]):
                raise ValueError('pilot timed source or points differ from output')
        for point in pilot['points']:
            path = out / f"{point['window']}-{point['name']}.npz"
            if worker.old.digest(path) != point['score_sha256']:
                raise ValueError('pilot score hash differs')
        if ({(p['kind'], p['scale']) for p in pilot['points'] if p['kind'] == 'control'} != {('control', 8)}
                or not any(p['kind'] == 'new' and p['name'].endswith('crf70') for p in pilot['points'])):
            raise ValueError('pilot controls or new CRF70 differ')
    bounds = {}
    for window in worker.WINDOWS:
        name = 'source_bound_' + window
        if name not in assets or gate['source_bound_sha256'][window] != assets[name]['sha256']:
            raise ValueError('six independently reviewed source-bound receipts required')
        bound = json.loads(Path(assets[name]['local']).read_text())
        if (bound.get('schema') != 1 or bound.get('reviewed') is not True or bound.get('window') != window
                or bound.get('execution_identity') != identity or bound.get('gpu_model') != 'RTX A6000'
                or bound.get('source_sha256') != assets[window + '_source']['sha256']
                or not cf.digest_string(bound.get('measurement_evidence_sha256', ''))):
            raise ValueError('source-bound evidence or provenance differs')
        bounds[window] = {'wall': bound['wall_seconds'], 'cpu': bound['cpu_seconds']}
    projection = worker.forecast(pilot, source_bounds=bounds, gpu_remaining=1800, cpu_remaining=1670,
        gpu_staging=gate['gpu_staging_seconds'], cpu_staging=gate['cpu_staging_seconds'],
        gpu_reserve=gate['gpu_audit_reserve_seconds'], cpu_reserve=gate['cpu_audit_reserve_seconds'])
    # Full job's smoke repeats the two-point pilot and must be charged too.
    for kind, maximum in (('wall', 1800), ('cpu', 1670)):
        if projection[kind + '_seconds'] + 1.5 * pilot['cost'][kind]['pilot_total'] > maximum:
            raise ValueError('full job including repeated smoke exceeds inclusive budget')
    predicted_full = projection['wall_seconds'] - pilot['cost']['wall']['pilot_total'] - gate['gpu_staging_seconds'] - gate['gpu_audit_reserve_seconds']
    if full_seconds < predicted_full or smoke_seconds < 1.5 * pilot['cost']['wall']['pilot_total']:
        raise ValueError('stage limits below measured conservative forecast')
    if (campaign_seconds + pilot['cost']['wall']['pilot_total'] + gate['gpu_audit_reserve_seconds'] > 1800
            or campaign_seconds + pilot['cost']['cpu']['pilot_total'] + gate['cpu_audit_reserve_seconds'] > 1670):
        raise ValueError('declared campaign exceeds inclusive budgets')
    files = {name: (entry['remote'], entry['sha256']) for name, entry in assets.items()}
    arguments = ['full-study', '--items', '{items}', '--byte-publication-sha256', assets['byte_publication']['sha256'], '--inventory', '{staged:inventory}', '--inventory-sha256', assets['inventory']['sha256'],
                 '--bindings', '{staged:bindings}', '--bindings-sha256', assets['bindings']['sha256'],
                 '--full-gate', '{staged:full_gate}', '--full-gate-sha256', assets['full_gate']['sha256']]
    result = w7_specs.common('experiments.tennis.w7_shared_fleet', arguments, files)
    result.update({'device': 'gpu', 'hosts': w7_specs.HOSTS_GPU, 'gpu_models': ['RTX A6000'],
                   'gpu_memory_mib': 8000, 'cpu_threads': 1, 'single_cpu_campaign': True,
                   'local_storage_gib': 30, 'deadline': deadline, 'scale': {'items': {'smoke': 2, 'full': 22}},
                   'smoke': {'seconds': smoke_seconds, 'representative_basis': 'Full-80-tick med000 control and new CRF70; verify before all 22 points.'},
                   'full': {'seconds': full_seconds}, 'validator_seconds': 20, 'budget_seconds': campaign_seconds,
                   'validator': ['{python}', '-m', 'experiments.tennis.w7_shared_fleet', 'validate']})
    validate_spec(result)
    return result


if __name__ == '__main__':
    main()
