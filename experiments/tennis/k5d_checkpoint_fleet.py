"""Fleet adapter for bounded K5d checkpoint equivalence, not corpus admission.

Specs are generated in sequence only from actual hash-verified publications.
Each job uses the same frozen source snapshot and packed runtime. No dispatch is
performed here; the caller owns admission, budgets, lifecycle and closeout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint as cp, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle

DATA_ROOT = Path('/home/itec/emanuele/Datasets/pointstream-data')
BASE = fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json'
AUDIT_PATH = fleet.ROOT / 'docs/experiments/k5d-inputs/cost-d8b5-independent-audit-20261105.json'
AUDIT_SHA = '66cbbd2ef1480524b0327effb9c760d7ef726abe40e338c3246c434bc9640d66'
PREPARE_RECEIPT = fleet.ROOT / 'docs/experiments/k5d-inputs/checkpoint-prepare-forecast-20261105.json'
PREPARE_RECEIPT_SHA = '53b15df0f4cc067b80320b45555dd9ebb501b5553f98112d4d99c9119069be11'
D8_SMOKE_SHA = '8758597ba524cafc017e25cbd20c15ecbeaeed534d5622dfe67d66eb16521377'
D8_SMOKE_PATH = str(DATA_ROOT / 'jobs/fleet/inbox/20261105T000550Z-d8b5c76b/smoke/published.tar')


def digest_string(value: str) -> bool:
    return len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def asset(name: str, remote: str, digest: str, local: Path, *, extract: bool) -> dict[str, Any]:
    """Specs cannot be rendered until the complete actual artifact was read."""
    if (not Path(remote).is_absolute() or not Path(remote).is_relative_to(DATA_ROOT)
            or not digest_string(digest) or oracle.sha(local) != digest):
        raise ValueError('actual staged artifact path or checksum mismatch: ' + name)
    return {'name': name, 'path': remote, 'sha256': digest, 'extract': extract}


def checkpoint_archive(name: str, remote: str, archive_sha: str, local: Path, checkpoint_sha: str,
                       frames: int) -> dict[str, Any]:
    item = asset(name, remote, archive_sha, local, extract=True)
    if not digest_string(checkpoint_sha):
        raise ValueError('checkpoint SHA256 required')
    with tarfile.open(local) as archive:
        members = [m for m in archive if m.name == 'publish/checkpoint.json']
        if len(members) != 1 or not members[0].isfile() or members[0].size > 1_000_000:
            raise ValueError('checkpoint manifest missing or ambiguous')
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError('checkpoint manifest unavailable')
        raw = stream.read()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != checkpoint_sha:
        raise ValueError('checkpoint manifest digest differs')
    record = json.loads(raw)
    if record.get('schema') != 1 or record.get('clip') != cost.CLIP or record.get('frames') != frames:
        raise ValueError('checkpoint timeline differs')
    return item


def common_spec(base: dict[str, Any], extra: list[dict[str, Any]], deadline: str) -> dict[str, Any]:
    if oracle.sha(AUDIT_PATH) != AUDIT_SHA:
        raise ValueError('independent d8b5 audit identity mismatch')
    prepare_baseline()
    spec = json.loads(json.dumps(base))
    spec['entrypoint'] = ['-m', 'experiments.tennis.k5d_checkpoint_fleet']
    spec['gpu_models'] = ['RTX A6000']
    spec['scale']['frames'] = {'smoke': 16, 'full': 48}
    spec['deadline'] = deadline
    spec['staged_inputs'].extend(extra)
    spec['inputs'].extend([{k: item[k] for k in ('path', 'sha256')} for item in extra])
    spec['validator_seconds'] = 20
    spec['verify_staged_inputs'] = True
    return spec


def prepare_spec(base: dict[str, Any], *, gate_remote: str, gate_sha: str, gate_local: Path,
                 reference_local: Path, deadline: str) -> dict[str, Any]:
    gate = asset('fidelity_gate', gate_remote, gate_sha, gate_local, extract=False)
    enc.fidelity_gate(gate_local, gate_sha, fleet.EXPECTED['archive'], purpose='48-frame-cost-pilot',
                      clip=cost.CLIP, frames=48)
    reference = asset('accepted16', D8_SMOKE_PATH, D8_SMOKE_SHA, reference_local, extract=False)
    spec = common_spec(base, [gate, reference], deadline)
    spec['arguments'] = ['prepare', *base['arguments'], '--fidelity-gate', '{staged:fidelity_gate}',
                         '--fidelity-gate-sha256', gate_sha, '--accepted16', '{staged:accepted16}']
    spec['smoke'] = {'seconds': 200, 'representative_basis': '16 frames, all six full native contexts, one joint fit and receiver parity; exact d8b5 16-frame control before scale.'}
    spec['full'] = {'seconds': 420}
    spec['budget_seconds'] = 650
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_checkpoint_fleet', 'validate-prepare']
    return spec


def score_spec(base: dict[str, Any], *, point: tuple[str, int], small: dict[str, Any],
               full: dict[str, Any], reference_local: Path, deadline: str) -> dict[str, Any]:
    if point not in enc.POINTS:
        raise ValueError('unregistered point')
    a = checkpoint_archive('prepare16', small['remote'], small['archive_sha256'], Path(small['local']),
                           small['checkpoint_sha256'], 16)
    b = checkpoint_archive('prepare48', full['remote'], full['archive_sha256'], Path(full['local']),
                           full['checkpoint_sha256'], 48)
    prior = asset('accepted16', D8_SMOKE_PATH, D8_SMOKE_SHA, reference_local, extract=False)
    spec = common_spec(base, [a, b, prior], deadline)
    spec['arguments'] = ['score', *base['arguments'], '--codec', point[0], '--point', str(point[1]),
                         '--prepare16', '{staged:prepare16}', '--prepare48', '{staged:prepare48}',
                         '--checkpoint16-sha256', small['checkpoint_sha256'], '--checkpoint48-sha256', full['checkpoint_sha256'],
                         '--accepted16', '{staged:accepted16}']
    spec['smoke'] = {'seconds': 120, 'representative_basis': 'Full-context 16-frame single-point receiver replay, both judges and perceptual metrics; exact d8b5 16-frame control.'}
    spec['full'] = {'seconds': 240}
    spec['budget_seconds'] = 400
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_checkpoint_fleet', 'validate-score']
    return spec


def reference16(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = enc.validate(root)
    report = json.loads((root / 'pilot.json').read_text())
    source = json.loads((root / 'source-qualification.json').read_text())
    if (validated['frames'] != 16 or validated['points'] != 6 or not validated['full_metrics']
            or report['clip'] != cost.CLIP or source['clip'] != cost.CLIP
            or source['device']['capability'] != [8, 6]):
        raise ValueError('d8b5 reference scope or GPU class changed')
    return report, source


def prepare_parity(checkpoint: Path, prior: Path, identity: dict[str, str], *,
                   capability: tuple[int, int] = (8, 6)) -> None:
    cp.verify(checkpoint, oracle.sha(checkpoint / 'checkpoint.json'), identity)
    reference, old_source = reference16(prior)
    new = json.loads((checkpoint / 'pilot.json').read_text())
    source = json.loads((checkpoint / 'source-qualification.json').read_text())
    cost.compare_contract(reference, old_source, new, source, exact=False, capability=capability)
    if (new['frames'] != 16 or source['rows'] != old_source['rows'] or source['scores'] != old_source['scores']
            or new['conversion']['yuv_sha256'] != reference['conversion']['yuv_sha256']):
        raise ValueError('prepare source or converted pixels differ from accepted16')
    for left, right in zip(reference['points'], new['points']):
        for key in ('codec', 'point', 'native_sha256', 'wire_sha256', 'encoder_decoded_sha256', 'encoder_rendered_sha256'):
            if left[key] != right[key]:
                raise ValueError('prepare native, fit payload or pixel differs: ' + key)
        if (left['baseline']['zero_ball_wire_sha256'] != right['baseline']['zero_ball_wire_sha256']
                or left['streak']['guard_off'] != right['streak']['guard_off']):
            raise ValueError('prepare zero packet or guard differs')


def score_parity(record: dict[str, Any], prior: Path, point: tuple[str, int], checkpoint_sha: str,
                 identity: dict[str, str]) -> None:
    reference, _ = reference16(prior)
    expected = next(p for p in reference['points'] if (p['codec'], p['point']) == point)
    actual = record['point']
    if (record['checkpoint_sha256'] != checkpoint_sha or record['execution_identity'] != identity
            or (actual['codec'], actual['point']) != point):
        raise ValueError('point identity differs')
    for key in ('native_sha256', 'wire_sha256', 'encoder_decoded_sha256', 'encoder_rendered_sha256'):
        if actual[key] != expected[key]:
            raise ValueError('point native or pixel differs: ' + key)
    for arm in ('baseline', 'streak'):
        if actual[arm]['metrics'] != expected[arm]['metrics'] or actual[arm]['judges'] != expected[arm]['judges']:
            raise ValueError('point metrics or judges differ')
    if actual['streak']['guard_off'] != expected['streak']['guard_off']:
        raise ValueError('point guard differs')
    replay = record['receiver_replay']
    if (replay['decoded_yuv_sha256'] != actual['encoder_decoded_sha256']
            or replay['rendered_rgb_sha256'] != actual['encoder_rendered_sha256']):
        raise ValueError('point replay pixels differ')


def forecast(stage: Path, limit: float, multiplier: float, now: float) -> dict[str, float]:
    execution = json.loads((stage / 'execution.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    clock = dispatch['campaign_clock']
    seconds = execution['seconds']
    if (execution.get('exit_code') != 0 or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
            or not math.isfinite(seconds) or seconds <= 0
            or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
        raise ValueError('invalid stage timing')
    estimate = 1.25 * multiplier * seconds + 20
    if estimate > limit or estimate + 20 > clock['budget_end'] - now:
        raise ValueError('48-frame forecast exceeds stage or remaining campaign')
    return {'smoke_seconds': seconds, 'estimate_seconds': estimate, 'remaining_seconds': clock['budget_end'] - now}


def prepare_baseline() -> tuple[float, float]:
    """Recompute the fixed/marginal preparation estimate from pinned outer intervals."""
    if oracle.sha(PREPARE_RECEIPT) != PREPARE_RECEIPT_SHA:
        raise ValueError('prepare timing evidence identity mismatch')
    receipt = json.loads(PREPARE_RECEIPT.read_text())
    if (receipt['job'] != '20261105T000550Z-d8b5c76b'
            or receipt['source_cost_audit_sha256'] != AUDIT_SHA):
        raise ValueError('prepare timing provenance mismatch')
    remaining = []
    for frames, expected_archive, expected_timing in (
        (16, D8_SMOKE_SHA, '77c0ff99c668c8c46e512b0bd732a4bbbf271d74f348b0d0202fddb8cf20c850'),
        (48, 'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727',
         'f413eb28cded1a10d229c3965b2d3701ef30b86410c89a7e5673fabfcc87faae'),
    ):
        stage = receipt['stages'][str(frames)]
        if stage['archive_sha256'] != expected_archive or stage['timing_sha256'] != expected_timing:
            raise ValueError('prepare timing archive identity mismatch')
        events = stage['scoring_events']
        expected_names = {f'{kind}.{codec}.{point}.{arm}'
                          for codec, point in enc.POINTS for kind in ('metrics', 'judges')
                          for arm in ('baseline', 'streak')}
        if len(events) != 24 or {e['name'] for e in events} != expected_names:
            raise ValueError('outer metric/judge timing coverage differs')
        durations = [e['wall_seconds'] for e in events]
        wall = stage['command_wall_seconds']
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                for x in [wall, *durations]) or wall <= sum(durations)):
            raise ValueError('invalid prepare timing intervals')
        rest = wall - sum(durations)
        if (abs(sum(durations) - stage['scoring_wall_seconds']) > 1e-9
                or abs(rest - stage['remaining_wall_seconds']) > 1e-9):
            raise ValueError('prepare timing arithmetic differs')
        remaining.append(rest)
    marginal = (remaining[1] - remaining[0]) / 32
    fixed = remaining[0] - 16 * marginal
    if marginal < 0 or fixed < 0:
        raise ValueError('nonphysical prepare fixed/marginal timing fit')
    if (abs(fixed - receipt['fixed_seconds']) > 1e-9
            or abs(marginal - receipt['marginal_seconds_per_frame']) > 1e-9):
        raise ValueError('prepare fixed/marginal arithmetic differs')
    return remaining[0], remaining[1]


def prepare_forecast(stage: Path, now: float) -> dict[str, float]:
    base16, base48 = prepare_baseline()
    execution = json.loads((stage / 'execution.json').read_text())
    clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
    seconds = execution['seconds']
    if (execution.get('exit_code') != 0 or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0
            or not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']):
        raise ValueError('invalid prepare campaign timing')
    spec = json.loads((stage.parent / 'spec.json').read_text())
    full_cap = spec['full']['seconds']
    if (isinstance(full_cap, bool) or not isinstance(full_cap, (int, float))
            or not math.isfinite(full_cap) or not 0 < full_cap <= 500):
        raise ValueError('prepare full stage cap must be positive and at most 500 seconds')
    slowdown = max(1., seconds / base16)
    estimate = 1.5 * base48 * slowdown + 40
    remaining = clock['budget_end'] - now
    if estimate > full_cap or estimate + 20 > remaining:
        raise ValueError('prepare 48-frame forecast exceeds stage or remaining campaign')
    return {'smoke_seconds': seconds, 'baseline16': base16, 'baseline48': base48,
            'observed_slowdown': slowdown, 'margin': 1.5, 'seal_staging_validation_reserve_seconds': 40.,
            'estimate_seconds': estimate, 'full_stage_cap_seconds': full_cap,
            'remaining_seconds': remaining, 'campaign_reserve_seconds': 20.}


def option(arguments: list[str], name: str) -> str:
    positions = [i for i, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError('missing or duplicate spec option: ' + name)
    return arguments[positions[0] + 1]


def validate_smoke(kind: str) -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        if oracle.sha(AUDIT_PATH) != AUDIT_SHA:
            raise ValueError('independent d8b5 audit identity mismatch')
        stage = Path(os.environ['PS_STAGE_DIR'])
        spec = json.loads((stage.parent / 'spec.json').read_text())
        staging = json.loads((stage / 'staging.json').read_text())
        entries = {item['name']: item for item in staging['inputs']}
        identity = cp.execution_identity()
        with (cost.unpack(Path(entries['accepted16']['path']), D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', staging['published']['sha256']) as produced):
            if kind == 'prepare':
                prepare_parity(produced, prior, identity)
                projection = prepare_forecast(stage, time.monotonic())
                checks = {'six_native_and_fit_parity': True, 'source_pixels_and_qualification': True,
                          'all_receiver_pixels': True, 'checkpoint_content': True, 'cost_forecast': True}
            elif kind == 'score':
                codec = option(spec['arguments'], '--codec')
                setting = int(option(spec['arguments'], '--point'))
                checkpoint_sha = option(spec['arguments'], '--checkpoint16-sha256')
                root = Path(entries['prepare16']['path']) / 'publish'
                cp.verify(root, checkpoint_sha, identity)
                score_parity(json.loads((produced / 'point.json').read_text()), prior,
                             (codec, setting), checkpoint_sha, identity)
                projection = forecast(stage, 240, 3, time.monotonic())
                checks = {'native_and_receiver_pixels': True, 'both_judges_and_all_metrics': True,
                          'checkpoint_content': True, 'cost_forecast': True}
            else:
                raise ValueError('unknown validation scope')
        target.write_text(json.dumps({'passed': True, 'checks': checks, 'forecast': projection,
                                      'scope': '16/48 cost and equivalence study only'}, allow_nan=False) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'checkpoint_validation': False},
                                      'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise


def run(args: argparse.Namespace) -> None:
    if oracle.sha(AUDIT_PATH) != AUDIT_SHA:
        raise ValueError('independent d8b5 audit identity mismatch')
    if {'smoke': 16, 'full': 48}.get(os.environ.get('PS_STAGE', '')) != args.frames:
        raise ValueError('stage frame scope differs')
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    identity = cp.execution_identity()
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError('source manifest or provenance changed')
    data = vars(args).copy()
    for name, digest in fleet.EXPECTED.items():
        data[name + '_sha256'] = digest
    data.update(clip=cost.CLIP, start_frame=0, manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA,
                code_revision=fleet.snapshot_head(stage), output=scratch / 'publish',
                feasibility_no_metrics=False, cost_pilot=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    data['snapshot_identity'] = {name: dispatch['code'].get(name) for name in
                                 ('git_head', 'source_sha256', 'snapshot_sha256', 'tracked_worktree_patch_sha256', 'included_untracked_sha256')}
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        archive = Path(data[name]).parent / Path(entries[name]['source']).name
        if oracle.sha(archive) != entries[name]['sha256']:
            raise ValueError('model code archive mismatch')
        expected = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(Path(data[name])) != expected:
            raise ValueError('installed source tree differs')
        data[name + '_sha256'] = expected
    for name, binary in (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')):
        data[name] = str(Path(sys.prefix) / 'bin' / binary)
    if args.action == 'prepare':
        prepare_baseline()
        enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                          purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=args.frames)
        cp.prepare(argparse.Namespace(**data), identity)
    elif args.action == 'score':
        which = '16' if args.frames == 16 else '48'
        data['checkpoint'] = str(Path(getattr(args, 'prepare' + which)) / 'publish')
        checkpoint_sha = getattr(args, 'checkpoint' + which + '_sha256')
        cp.score_point(argparse.Namespace(**data), checkpoint_sha, identity)
    else:
        raise ValueError('unknown study action')


def point_archive(name: str, entry: dict[str, str], key: tuple[str, int], checkpoint_sha: str) -> tuple[dict[str, Any], str]:
    import hashlib

    item = asset(name, entry['remote'], entry['archive_sha256'], Path(entry['local']), extract=True)
    with tarfile.open(entry['local']) as archive:
        members = [m for m in archive if m.name == 'publish/point.json']
        if len(members) != 1 or not members[0].isfile() or members[0].size > 50_000_000:
            raise ValueError('score receipt missing or ambiguous')
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError('score receipt absent')
        raw = handle.read()
    receipt = json.loads(raw)
    point = receipt['point']
    if receipt['checkpoint_sha256'] != checkpoint_sha or (point['codec'], point['point']) != key:
        raise ValueError('point archive belongs to another checkpoint or setting')
    sha = hashlib.sha256(raw).hexdigest()
    if sha != entry['point_sha256']:
        raise ValueError('point receipt SHA256 mismatch')
    return item, sha


def assemble_spec(base: dict[str, Any], *, small: dict[str, str], full: dict[str, str],
                  points16: list[dict[str, str]], points48: list[dict[str, str]],
                  reference16_local: Path, reference48_local: Path, deadline: str) -> dict[str, Any]:
    if len(points16) != 6 or len(points48) != 6:
        raise ValueError('actual sixteen and forty-eight frame point archives required')
    a = checkpoint_archive('prepare16', small['remote'], small['archive_sha256'], Path(small['local']), small['checkpoint_sha256'], 16)
    b = checkpoint_archive('prepare48', full['remote'], full['archive_sha256'], Path(full['local']), full['checkpoint_sha256'], 48)
    ref16 = asset('reference16', D8_SMOKE_PATH, D8_SMOKE_SHA, reference16_local, extract=False)
    sha48 = 'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727'
    ref48 = asset('reference48', '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/'
                  '20261105T000550Z-d8b5c76b/full/published.tar', sha48, reference48_local, extract=False)
    extra = [a, b, ref16, ref48]
    arguments = ['assemble', '--frames', '{frames}', '--prepare16', '{staged:prepare16}',
                 '--prepare48', '{staged:prepare48}', '--checkpoint16-sha256', small['checkpoint_sha256'],
                 '--checkpoint48-sha256', full['checkpoint_sha256'], '--reference16', '{staged:reference16}',
                 '--reference48', '{staged:reference48}']
    for n, entries, expected_sha in ((16, points16, small['checkpoint_sha256']), (48, points48, full['checkpoint_sha256'])):
        for i, (key, entry) in enumerate(zip(enc.POINTS, entries)):
            name = f'point{n}_{i}'
            item, sha = point_archive(name, entry, key, expected_sha)
            extra.append(item)
            arguments.extend([f'--point{n}', '{staged:' + name + '}', f'--point{n}-sha256', sha])
    spec = common_spec(base, extra, deadline)
    # Assembly consumes only immutable small prior outputs and the original environment.
    spec['staged_inputs'] = extra
    spec['inputs'] = [{k: x[k] for k in ('path', 'sha256')} for x in extra]
    spec['device'], spec['gpu_models'], spec['gpu_memory_mib'] = 'cpu', [], 0
    spec.pop('gpu_sharing', None)
    spec['arguments'] = arguments
    spec['required_commands'] = []
    spec['smoke'] = {'seconds': 35, 'representative_basis': 'Assemble and independently compare all six immutable 16-frame checkpoints with d8b5 monolithic output.'}
    spec['full'] = {'seconds': 65}
    spec['budget_seconds'] = 120
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.k5d_checkpoint_fleet', 'validate-assembly']
    return spec


def assembly_run(args: argparse.Namespace) -> None:
    if oracle.sha(AUDIT_PATH) != AUDIT_SHA:
        raise ValueError('independent d8b5 audit identity mismatch')
    if {'smoke': 16, 'full': 48}.get(os.environ.get('PS_STAGE', '')) != args.frames:
        raise ValueError('assembly stage frame count differs')
    which = str(args.frames)
    checkpoint = Path(getattr(args, 'prepare' + which)) / 'publish'
    reference = Path(getattr(args, 'reference' + which))
    checkpoint_sha = getattr(args, 'checkpoint' + which + '_sha256')
    paths = getattr(args, 'point' + which)
    shas = getattr(args, 'point' + which + '_sha256')
    if len(paths) != 6 or len(shas) != 6:
        raise ValueError('six point archives required')
    receipts = [{'path': str(Path(path) / 'publish' / 'point.json'), 'sha256': sha}
                for path, sha in zip(paths, shas)]
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    result = cp.assemble(checkpoint, checkpoint_sha, receipts, output / 'assembled', cp.execution_identity())
    old_archive_sha = D8_SMOKE_SHA if args.frames == 16 else 'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727'
    with cost.unpack(reference, old_archive_sha) as prior:
        old_sha = oracle.sha(prior / 'pilot.json')
        comparison = cp.compare_equivalence(prior, output / 'assembled', old_sha)
    cp.write_new(output / 'assembly-validation.json', {'result': result, 'comparison': comparison,
                                                       'checkpoint_sha256': checkpoint_sha,
                                                       'point_sha256': shas})


def validate_assembly() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        if oracle.sha(AUDIT_PATH) != AUDIT_SHA:
            raise ValueError('independent d8b5 audit identity mismatch')
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        entries = {item['name']: item for item in staging['inputs']}
        with (cost.unpack(Path(entries['reference16']['path']), D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output):
            record = json.loads((output / 'assembly-validation.json').read_text())
            result = cp.compare_equivalence(prior, output / 'assembled', oracle.sha(prior / 'pilot.json'))
            if record['comparison'] != result or not result['passed'] or result['frames'] != 16:
                raise ValueError('assembled 16-frame parity failed')
        projection = forecast(stage, 65, 3, time.monotonic())
        target.write_text(json.dumps({'passed': True, 'checks': {'six_point_assembly': True,
                          'exact_accepted16': True, 'full_cost_forecast': True}, 'forecast': projection}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'assembly': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == '--make-spec':
        parser = argparse.ArgumentParser(description='Build spec only from actual hash-verified publications')
        parser.add_argument('--config', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        options = json.loads(args.config.read_text())
        kind = options.pop('kind')
        base = json.loads(BASE.read_text())
        if kind == 'prepare':
            for name in ('gate_local', 'reference_local'):
                options[name] = Path(options[name])
            spec = prepare_spec(base, **options)
        elif kind == 'score':
            options['reference_local'] = Path(options['reference_local'])
            options['point'] = tuple(options['point'])
            spec = score_spec(base, **options)
        elif kind == 'assemble':
            for name in ('reference16_local', 'reference48_local'):
                options[name] = Path(options[name])
            spec = assemble_spec(base, **options)
        else:
            parser.error('unknown spec kind')
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        with args.output.open('x') as output:
            json.dump(spec, output, indent=2, allow_nan=False)
            output.write('\n')
        return
    if sys.argv[1:] == ['validate-assembly']:
        validate_assembly()
        return
    if sys.argv[1:] in (['validate-prepare'], ['validate-score']):
        validate_smoke(sys.argv[1].removeprefix('validate-'))
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'assemble':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action')
        parser.add_argument('--frames', type=int, required=True)
        for name in ('prepare16', 'prepare48', 'checkpoint16-sha256', 'checkpoint48-sha256', 'reference16', 'reference48'):
            parser.add_argument('--' + name, required=True)
        for n in (16, 48):
            parser.add_argument(f'--point{n}', action='append', required=True)
            parser.add_argument(f'--point{n}-sha256', action='append', required=True)
        assembly_run(parser.parse_args())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'score'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'accepted16'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--frames', required=True, type=int)
    for name in ('fidelity-gate', 'fidelity-gate-sha256', 'prepare16', 'prepare48',
                 'checkpoint16-sha256', 'checkpoint48-sha256', 'codec'):
        parser.add_argument('--' + name)
    parser.add_argument('--point', type=int)
    args = parser.parse_args()
    required = ('fidelity_gate', 'fidelity_gate_sha256') if args.action == 'prepare' else (
        'prepare16', 'prepare48', 'checkpoint16_sha256', 'checkpoint48_sha256', 'codec', 'point')
    if any(getattr(args, name) is None for name in required):
        parser.error('missing required action argument')
    run(args)


if __name__ == '__main__':
    main()
