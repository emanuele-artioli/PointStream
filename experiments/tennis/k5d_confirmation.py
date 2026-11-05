"""Prospective Clip3 16/188 confirmation; no corpus admission."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as encoder, k5d_fleet as fleet, k5d_source_oracle as oracle
from experiments.tennis.k5d_timing import Timer

AUDIT_SHA = '66cbbd2ef1480524b0327effb9c760d7ef726abe40e338c3246c434bc9640d66'
AUDIT = fleet.ROOT / 'docs/experiments/k5d-inputs/cost-d8b5-independent-audit-20261105.json'
D8_SHA = '8758597ba524cafc017e25cbd20c15ecbeaeed534d5622dfe67d66eb16521377'
D8_JOB = '20261105T000550Z-d8b5c76b'
FIXED = 138.72209787368774
MARGINAL = 4.190426766872406
PURPOSE = '188-frame-confirmation'


def forecast(stage: Path, now: float) -> dict[str, float]:
    execution = json.loads((stage / 'execution.json').read_text())
    clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
    sec = execution['seconds']
    started, phase, end = clock['started'], clock['phase_started'], clock['budget_end']
    values = (sec, started, phase, end, now)
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError('invalid campaign timing')
    if execution.get('exit_code') != 0 or sec <= 0 or not (0 < started <= phase <= now < end):
        raise ValueError('inconsistent campaign timing')
    baseline16 = FIXED + MARGINAL * 16
    slowdown = max(1., sec / baseline16)
    estimate = 1.5 * (FIXED + MARGINAL * 188) * slowdown + 200 + 60
    if estimate > 1750 or estimate + 60 > end - now:
        raise ValueError('188-frame forecast exceeds stage or campaign budget')
    return {'smoke_seconds': sec, 'observed_slowdown': slowdown, 'fixed': FIXED, 'per_frame': MARGINAL,
            'margin': 1.5, 'overhead_seconds': 200., 'reserve_seconds': 60.,
            'full_estimate_seconds': estimate, 'campaign_remaining_seconds': end - now}


def check_timing(path: Path, frames: int) -> None:
    report = json.loads((path / 'pilot.json').read_text())
    source = json.loads((path / 'source-qualification.json').read_text())
    if (report['frames'] != frames or report.get('cost_pilot') or report.get('benchmark')
            or report['clip'] != cost.CLIP or report['fidelity_gate']['purpose'] != PURPOSE
            or report['source_task_qualified'] != source['qualified']):
        raise ValueError('confirmation scope or source qualification mismatch')
    events = json.loads((path / 'timing.json').read_text())['events']
    names = {event['name'] for event in events}
    required = {'encoder.run_total', 'reference.compare', 'source.oracle_total', 'judge.wasb', 'judge.balltrack',
                'model.lpips_init', 'model.dists_init', 'model.wasb_init', 'model.balltrack_init'}
    for kind, point in encoder.POINTS:
        required.update({f'native.{kind}.{point}.total', f'receiver.{kind}.{point}',
                         f'metrics.{kind}.{point}.baseline', f'judges.{kind}.{point}.baseline'})
    if not required <= names:
        raise ValueError(f'missing timing events: {sorted(required - names)}')
    for event in events:
        if not event['completed'] or any(isinstance(event[k], bool) or not isinstance(event[k], (int, float))
                or not math.isfinite(event[k]) or event[k] < 0
                for k in ('wall_seconds', 'self_cpu_seconds', 'waited_children_cpu_seconds')):
            raise ValueError('invalid timing event')


def validate_smoke() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        if oracle.sha(AUDIT) != AUDIT_SHA:
            raise ValueError('independent cost audit identity mismatch')
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        entries = {entry['name']: entry for entry in staging['inputs']}
        with (cost.unpack(Path(entries['accepted16']['path']), cost.ACCEPTED_SHA) as old,
              cost.unpack(Path(entries['cost16']['path']), D8_SHA) as d8,
              cost.unpack(stage / 'published.tar', staging['published']['sha256']) as new):
            prior, original = cost.reference(old)
            historical = encoder.validate(d8)
            current = encoder.validate(new)
            if any(r['frames'] != 16 or r['points'] != 6 or not r['full_metrics'] for r in (historical, current)):
                raise ValueError('incomplete six-point smoke')
            current_report = json.loads((new / 'pilot.json').read_text())
            current_source = json.loads((new / 'source-qualification.json').read_text())
            for reference_dir, report, source in ((old, prior, original),
                    (d8, json.loads((d8 / 'pilot.json').read_text()), json.loads((d8 / 'source-qualification.json').read_text()))):
                del reference_dir
                cost.compare_contract(report, source, current_report, current_source, exact=True)
            check_timing(new, 16)
        projection = forecast(stage, time.monotonic())
        result = {'passed': True, 'scope': 'Clip3 188 confirmation only; no corpus admission',
                  'checks': {'six_native_and_receiver': True, 'ed28_and_d8b5_exact': True,
                             'source_task_flag_rechecked': True, 'component_timing': True},
                  'forecast': projection, 'cost_audit_sha256': AUDIT_SHA}
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    target.write_text(json.dumps(result, allow_nan=False) + '\n')


def make_spec(base: dict[str, Any], gate_path: str, gate_sha: str, gate_copy: Path, deadline: str) -> dict[str, Any]:
    data_root = Path('/home/itec/emanuele/Datasets/pointstream-data')
    if not Path(gate_path).is_absolute() or not Path(gate_path).is_relative_to(data_root):
        raise ValueError('receipt path must be an absolute data-root file')
    if len(gate_sha) != 64 or any(c not in '0123456789abcdef' for c in gate_sha) or oracle.sha(gate_copy) != gate_sha:
        raise ValueError('actual receipt SHA256 mismatch')
    encoder.fidelity_gate(gate_copy, gate_sha, fleet.EXPECTED['archive'], purpose=PURPOSE, clip=cost.CLIP, frames=188)
    if oracle.sha(AUDIT) != AUDIT_SHA:
        raise ValueError('independent cost audit identity mismatch')
    spec = json.loads(json.dumps(base))
    spec['entrypoint'] = ['-m', 'experiments.tennis.k5d_confirmation']
    spec['gpu_models'] = ['RTX A6000']
    spec['arguments'] += ['--accepted16', '{staged:accepted16}', '--cost16', '{staged:cost16}',
                          '--fidelity-gate', '{staged:fidelity_gate}', '--fidelity-gate-sha256', gate_sha]
    spec['scale']['frames'] = {'smoke': 16, 'full': 188}
    entries = [
        {'name': 'accepted16', 'path': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261104T223611Z-ed28cf8e/smoke/published.tar', 'sha256': cost.ACCEPTED_SHA},
        {'name': 'cost16', 'path': '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261105T000550Z-d8b5c76b/smoke/published.tar', 'sha256': D8_SHA},
        {'name': 'fidelity_gate', 'path': gate_path, 'sha256': gate_sha},
    ]
    spec['staged_inputs'].extend(entries)
    spec['inputs'].extend([{k: e[k] for k in ('path', 'sha256')} for e in entries])
    spec['smoke'] = {'seconds': 500, 'representative_basis': '16 Clip3 frames, six real codec paths, both judges and exact ed28 and d8b5 parity; measured 16-frame slowdown scales frozen affine 188 forecast with 1.5 margin and reserves.'}
    spec['full'] = {'seconds': 1750}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_confirmation', '--validate-smoke']
    spec['validator_seconds'] = 60
    spec['budget_seconds'] = 2400
    spec['deadline'] = deadline
    return spec


def main() -> None:
    if sys.argv[1:] == ['--validate-smoke']:
        validate_smoke()
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--make-spec':
        p = argparse.ArgumentParser()
        for name in ('gate-path', 'gate-sha256', 'gate-copy', 'deadline', 'output'):
            p.add_argument('--' + name, required=True)
        a = p.parse_args(sys.argv[2:])
        base = json.loads((fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json').read_text())
        spec = make_spec(base, a.gate_path, a.gate_sha256, Path(a.gate_copy), a.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with Path(a.output).open('x') as out:
            json.dump(spec, out, indent=2, allow_nan=False)
            out.write('\n')
        return
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config', 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'accepted16', 'cost16', 'fidelity-gate', 'fidelity-gate-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--frames', type=int, required=True)
    args = p.parse_args()
    if {'smoke': 16, 'full': 188}.get(os.environ.get('PS_STAGE', '')) != args.frames:
        raise ValueError('stage and frame count mismatch')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    timer = Timer(scratch / 'confirmation-timing.json')
    if timer.measure('bootstrap.source_manifest_hash', lambda: oracle.sha(fleet.SOURCE)) != fleet.SOURCE_SHA or timer.measure('bootstrap.exposure_hash', lambda: oracle.sha(fleet.EXPOSURE)) != fleet.EXPOSURE_SHA or oracle.sha(AUDIT) != AUDIT_SHA:
        raise ValueError('source manifest, exposure or cost audit differs')
    data = vars(args).copy()
    stage = Path(os.environ['PS_STAGE_DIR'])
    dispatch_code = json.loads((stage / 'dispatch.json').read_text())['code']
    data['snapshot_identity'] = {name: dispatch_code.get(name) for name in ('git_head', 'source_sha256', 'snapshot_sha256', 'tracked_worktree_patch_sha256', 'included_untracked_sha256')}
    data.update(code_revision=fleet.snapshot_head(stage), manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA, clip=cost.CLIP, start_frame=0,
                output=scratch / 'publish', feasibility_no_metrics=False, cost_pilot=False, gate_purpose=PURPOSE, timer=timer)
    for name, digest in fleet.EXPECTED.items():
        data[name + '_sha256'] = digest
    timer.measure('bootstrap.fidelity_gate_validate', lambda: encoder.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'], purpose=PURPOSE, clip=cost.CLIP, frames=args.frames))
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        entry = entries[name]
        archive = Path(data[name]).parent / Path(entry['source']).name
        if timer.measure(f'bootstrap.hash.{name}_archive', lambda: oracle.sha(archive)) != entry['sha256']:
            raise ValueError('model code archive identity differs')
        expected = timer.measure(f'bootstrap.tar_tree.{name}', lambda: fleet.archive_tree_sha(archive))
        if timer.measure(f'bootstrap.installed_tree.{name}', lambda: oracle.tree_sha(Path(data[name]))) != expected:
            raise ValueError('cached code tree differs')
        data[name + '_sha256'] = expected
    for name, binary in (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')):
        data[name] = str(Path(sys.prefix) / 'bin' / binary)
    with cost.unpack(Path(args.accepted16), cost.ACCEPTED_SHA, timer, 'bootstrap.accepted16') as old, cost.unpack(Path(args.cost16), D8_SHA, timer, 'bootstrap.cost16') as d8:
        prior, original = timer.measure('bootstrap.accepted16_validate', lambda: cost.reference(old))
        timer.measure('bootstrap.cost16_validate', lambda: encoder.validate(d8))
        timer.measure('encoder.run_total', lambda: encoder.run(argparse.Namespace(**data)))
        current = json.loads((data['output'] / 'pilot.json').read_text())
        source = json.loads((data['output'] / 'source-qualification.json').read_text())
        def compare() -> None:
            cost.compare_contract(prior, original, current, source, exact=args.frames == 16)
            cost.compare_contract(json.loads((d8 / 'pilot.json').read_text()), json.loads((d8 / 'source-qualification.json').read_text()), current, source, exact=args.frames == 16)
        timer.measure('reference.compare', compare)
        timer.relocate(data['output'] / 'timing.json')
        check_timing(data['output'], args.frames)
        (data['output'] / 'reference-parity.json').write_text(json.dumps({'reference_jobs': [cost.ACCEPTED_JOB, D8_JOB], 'reference_sha256': [cost.ACCEPTED_SHA, D8_SHA], 'mode': 'exact-16' if args.frames == 16 else 'source-prefix-188', 'passed': True}) + '\n')


if __name__ == '__main__':
    main()
