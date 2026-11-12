"""Prospective isolated Ada 16/48 producer and sealed-pixel scorer.

Requires root-authenticated bridge, CPU proof, config and actual witness. This
module does not authorize dispatch or establish a forecast before its smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_ada_scale_adapter as adapter, k5d_checkpoint as cp
from experiments.tennis import k5d_checkpoint_fleet as cf, k5d_coalesced as co
from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as enc
from experiments.tennis import k5d_fleet as fleet, k5d_real_parity as parity
from experiments.tennis import k5d_source_oracle as oracle


def digest(value: Any) -> bool:
    return isinstance(value, str) and cf.digest_string(value)


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def checked(path: Path, digest: str) -> dict[str, Any]:
    if not cf.digest_string(digest) or oracle.sha(path) != digest:
        raise ValueError('authenticated asset checksum differs')
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError('authenticated JSON object required')
    return result


def base_admission(args: argparse.Namespace, identity: dict[str, str]) -> dict[str, Any]:
    """Authenticate the source bridge, configuration and native CPU proof."""
    from experiments.tennis import k5d_scoped_namespace as namespace

    bridge = checked(Path(args.bridge), args.bridge_sha256)
    if (bridge.get('approved') is not True or bridge.get('consumer_execution_identity') != identity
            or bridge.get('calibration_execution_identity') != adapter.REFERENCE_IDENTITY
            or bridge.get('consumer_snapshot_sha256') != identity['snapshot_sha256']):
        raise ValueError('approved exact-source Ada bridge required')
    config = checked(Path(args.receiver_isolation_config), args.receiver_isolation_config_sha256)
    if (config.get('schema') != 1 or config.get('execution_identity') != identity
            or not isinstance(config.get('profiles'), dict) or 'gpu6' not in config['profiles']):
        raise ValueError('authenticated isolation configuration differs')
    proof = checked(Path(args.native_cpu_receipt), args.native_cpu_receipt_sha256)
    if (proof.get('schema') != 1 or proof.get('audited') is not True or proof.get('passed') is not True
            or proof.get('scope') != 'native CPU PID containment and accounting' or proof.get('gpu_used') is not False
            or proof.get('consumer_execution_identity') != identity
            or proof.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))
            or proof.get('unshare_sha256') != args.unshare_sha256
            or not isinstance(proof.get('job_id'), str) or not proof['job_id'].startswith('2026')
            or proof.get('all_owned_pids_absent') is not True or proof.get('independent_cpu_match') is not True
            or proof.get('cases') != ['normal', 'timeout', 'sigterm', 'sigkill']):
        raise ValueError('actual source-bound native CPU proof required')
    return config


def admission(args: argparse.Namespace, identity: dict[str, str]) -> None:
    config = base_admission(args, identity)
    witness = checked(Path(args.denial_witness), args.denial_witness_sha256)
    context = witness.get('producer_context')
    gpu = context.get('gpu', {}) if isinstance(context, dict) else {}
    if (witness.get('approved') is not True or witness.get('actual_filesystem_denial_passed') is not True
            or witness.get('host') != 'gpu6' or sorted(witness.get('codecs', [])) != ['dcvc', 'svt']
            or not digest(witness.get('actual_policy_sha256')) or not digest(witness.get('profile_sha256')) or not digest(witness.get('runtime_inventory_sha256'))
            or witness.get('consumer_execution_identity') != identity
            or witness.get('config_sha256') != args.receiver_isolation_config_sha256
            or not isinstance(context, dict) or context.get('execution_identity') != identity
            or not str(context.get('job_id', '')).startswith('2026') or context.get('stage') not in ('smoke', 'full')
            or gpu.get('name') != adapter.GPU_NAME or gpu.get('capability') != [8, 9]
            or not str(gpu.get('uuid', '')).startswith('GPU-')):
        raise ValueError('root-approved actual same-job Ada denial witness required')
    isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
    verified = isolated.verify_receipt(witness.get('receipt'), expected_context=context,
                                       expected_config_sha256=args.receiver_isolation_config_sha256)
    profile = config['profiles']['gpu6']
    if (not isinstance(verified, dict) or type(verified.get('landlock_abi')) is not int
            or verified['landlock_abi'] < 3 or type(verified.get('handled_mask')) is not int
            or not isinstance(profile, dict) or not isinstance(profile.get('runtime_files'), dict)
            or witness['profile_sha256'] != canonical_sha(profile)
            or witness['runtime_inventory_sha256'] != canonical_sha(profile['runtime_files'])
            or verified.get('profile_sha256') != witness['profile_sha256']
            or verified.get('actual_policy_sha256') != witness['actual_policy_sha256']
            or verified.get('runtime_inventory_sha256') != witness['runtime_inventory_sha256']):
        raise ValueError('witness policy or runtime inventory binding differs')



def producer_attestation(args: argparse.Namespace, identity: dict[str, str], frames: int,
                         marker: dict[str, Any]) -> None:
    """Bind a root-reviewed fleet audit to the exact immutable producer archive."""
    suffix = str(frames)
    audit = checked(Path(getattr(args, 'producer' + suffix + '_attestation')),
                    getattr(args, 'producer' + suffix + '_attestation_sha256'))
    config = checked(Path(args.receiver_isolation_config), args.receiver_isolation_config_sha256)
    profile = config['profiles']['gpu6']
    expected_gpu = {**marker['gpu'], 'capability': [8, 9]}
    if (audit.get('schema') != 1 or audit.get('approved') is not True
            or audit.get('consumer_execution_identity') != identity
            or audit.get('producer_job') != marker['producer_job']
            or audit.get('producer_stage') != marker['producer_stage']
            or audit.get('publication_sha256') != getattr(args, 'prepare' + suffix + '_sha256')
            or audit.get('gpu') != expected_gpu or audit.get('receiver_denial_passed') is not True
            or audit.get('config_sha256') != args.receiver_isolation_config_sha256
            or audit.get('profile_sha256') != canonical_sha(profile)
            or audit.get('runtime_inventory_sha256') != canonical_sha(profile['runtime_files'])):
        raise ValueError('approved producer audit does not bind this job and publication')

def data_args(args: argparse.Namespace, stage: Path, scratch: Path, dispatch: dict) -> argparse.Namespace:
    data = vars(args).copy()
    for name, digest in fleet.EXPECTED.items():
        data[name + '_sha256'] = digest
    data.update(clip=cost.CLIP, start_frame=0, manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA,
                code_revision=fleet.snapshot_head(stage), output=scratch / 'publish',
                feasibility_no_metrics=False, cost_pilot=True, same_job_receiver=True, isolated_receiver=True)
    data['snapshot_identity'] = {name: dispatch['code'].get(name) for name in
                                 ('git_head', 'source_sha256', 'snapshot_sha256', 'tracked_worktree_patch_sha256', 'included_untracked_sha256')}
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        archive = Path(data[name]).parent / Path(entries[name]['source']).name
        if oracle.sha(archive) != entries[name]['sha256']:
            raise ValueError('model source archive differs')
        expected = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(Path(data[name])) != expected:
            raise ValueError('installed model source differs')
        data[name + '_sha256'] = expected
    for name, binary in (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')):
        data[name] = str(Path(sys.prefix) / 'bin' / binary)
    return argparse.Namespace(**data)


def validate_score(scored: Path, prepared: Path, sha: str, identity: dict[str, str]) -> None:
    record = json.loads((scored / 'coalesced-validation.json').read_text())
    names = [f'point-{codec}-{point}.json' for codec, point in enc.POINTS]
    if (record.get('checkpoint_sha256') != sha or record.get('execution_identity') != identity
            or [item['name'] for item in record['points']] != names):
        raise ValueError('six score proofs or checkpoint identity differ')
    files = []
    for item in record['points']:
        path = scored / item['name']
        if oracle.sha(path) != item['sha256']:
            raise ValueError('score proof hash differs')
        files.append({'path': str(path), 'sha256': item['sha256']})
    with tempfile.TemporaryDirectory(prefix='ada-score-verify-', dir=os.environ.get('TMPDIR')) as folder:
        rebuilt = Path(folder) / 'assembled'
        result = cp.assemble(prepared, sha, files, rebuilt, identity)
        if result != record['result'] or oracle.sha(rebuilt / 'pilot.json') != oracle.sha(scored / 'assembled' / 'pilot.json'):
            raise ValueError('scored assembly differs from sealed points')


def smoke_forecast(stage: Path) -> dict[str, float]:
    """Conservative conditional 48-frame admission, including reserves."""
    from experiments.tennis import k5d_ada_fresh as fresh_worker
    wall, cpu = fresh_worker.stage_receipt(stage)
    clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
    now = time.monotonic()
    if not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']:
        raise ValueError('campaign clock missing or invalid')
    estimate = 1.5 * 3 * wall + 50
    remaining = clock['budget_end'] - now
    if estimate > 580 or remaining < max(600, estimate + 130):
        raise ValueError('complete smoke forecast cannot support 48-frame stage')
    return {'wall_seconds': wall, 'cpu_seconds': cpu, 'full_estimate_seconds': estimate,
            'remaining_seconds': remaining, 'setup_seconds': clock['phase_started'] - clock['started']}


def run(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    phase = os.environ['PS_STAGE']
    if {'smoke': 16, 'full': 48}.get(phase) != args.frames:
        raise ValueError('only paired 16/48 stages are permitted')
    identity = cp.execution_identity()
    admission(args, identity)
    if phase == 'full':
        gate = json.loads((stage.parent / 'gate.json').read_text())
        if gate.get('passed') is not True or gate.get('validation', {}).get('checks', {}).get('forecast') is not True:
            raise ValueError('authenticated smoke forecast gate required')
        smoke_forecast(stage.parent / 'smoke')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    specification = json.loads((stage.parent / 'spec.json').read_text())
    import torch
    device = parity.check_gpu_profile(parity.ADA_PROFILE, specification, dispatch, torch, os.environ)
    if (oracle.sha(cf.AUDIT_PATH) != cf.AUDIT_SHA or oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA
            or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA):
        raise ValueError('source or exposure identity differs')
    data = data_args(args, stage, scratch, dispatch)
    data.receiver_isolation_config = Path(args.receiver_isolation_config)
    data.receiver_producer_context = {'job_id': stage.parent.name, 'stage': phase,
                                      'execution_identity': identity, 'gpu': device}
    data.receiver_protected_paths = {
        'source_archive': [Path(args.archive)],
        'ground_truth': [fleet.SOURCE, fleet.EXPOSURE],
        'private_detector': [Path(args.balltrack_weight), Path(args.wasb_config),
                             Path(args.wasb_weights) / 'wasb_tennis_best.pth.tar'],
    }
    if args.action == 'prepare':
        enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                          purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=args.frames)
        cp.prepare(data, identity)
        prepared = Path(data.output)
        adapter.producer(prepared, identity, job=stage.parent.name, stage=phase, frames=args.frames,
                         config_sha256=args.receiver_isolation_config_sha256,
                         config_path=Path(args.receiver_isolation_config))
        if args.frames == 16:
            adapter.prepare16(Path(args.ada_reference), prepared, identity)
        else:
            adapter.prepare48(Path(args.ada_reference), prepared, identity)
    else:
        archive = Path(getattr(args, 'prepare' + str(args.frames)))
        archive_sha = getattr(args, 'prepare' + str(args.frames) + '_sha256')
        expected_job = getattr(args, 'producer' + str(args.frames) + '_job')
        with cost.unpack(archive, archive_sha) as prepared:
            marker = adapter.producer(prepared, identity, job=expected_job, stage=phase, frames=args.frames,
                             config_sha256=args.receiver_isolation_config_sha256,
                             config_path=Path(args.receiver_isolation_config))
            producer_attestation(args, identity, args.frames, marker)
            if args.frames == 16:
                adapter.prepare16(Path(args.ada_reference), prepared, identity)
            else:
                adapter.prepare48(Path(args.ada_reference), prepared, identity)
            data.checkpoint = str(prepared)
            data.receiver_producer_job = expected_job
            checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
            co.score_all(data, checkpoint_sha, identity)
            validate_score(Path(data.output), prepared, checkpoint_sha, identity)
            if args.frames == 16:
                with cost.unpack(Path(args.accepted16), cf.D8_SMOKE_SHA) as accepted:
                    adapter.score16(Path(args.ada_reference), Path(data.output) / 'assembled', accepted)


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        if os.environ['PS_STAGE'] != 'smoke':
            raise ValueError('fleet validator requires smoke stage')
        staging = json.loads((stage / 'staging.json').read_text())
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        command = dispatch['command']
        modes = [(i, x) for i, x in enumerate(command) if x in ('ada-scale-prepare', 'ada-scale-score')]
        if len(modes) != 1:
            raise ValueError('fixed worker mode missing')
        index, mode = modes[0]
        action = 'prepare' if mode == 'ada-scale-prepare' else 'score'
        args = parser().parse_args([action, *command[index + 1:]])
        if args.frames != 16:
            raise ValueError('smoke frame scope differs')
        identity = cp.execution_identity()
        admission(args, identity)
        published = staging['published']['sha256']
        with cost.unpack(stage / 'published.tar', published) as output:
            if action == 'prepare':
                adapter.producer(output, identity, job=stage.parent.name, stage='smoke', frames=16,
                                 config_sha256=args.receiver_isolation_config_sha256,
                             config_path=Path(args.receiver_isolation_config))
                adapter.prepare16(Path(args.ada_reference), output, identity)
            else:
                with cost.unpack(Path(args.prepare16), args.prepare16_sha256) as prepared:
                    marker = adapter.producer(prepared, identity, job=args.producer16_job, stage='smoke', frames=16,
                                     config_sha256=args.receiver_isolation_config_sha256,
                             config_path=Path(args.receiver_isolation_config))
                    producer_attestation(args, identity, 16, marker)
                    adapter.prepare16(Path(args.ada_reference), prepared, identity)
                    validate_score(output, prepared, oracle.sha(prepared / 'checkpoint.json'), identity)
                    with cost.unpack(Path(args.accepted16), cf.D8_SMOKE_SHA) as accepted:
                        adapter.score16(Path(args.ada_reference), output / 'assembled', accepted)
        forecast = smoke_forecast(stage)
        target.write_text(json.dumps({'passed': True, 'checks': {'forecast': True, 'six_real_points': True,
                          'actual_denial_receipts': True, 'ada_reference': True}, 'forecast': forecast}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'forecast': False}, 'error': str(exc)}) + '\n')
        raise


def spec(base: dict, *, action: str, assets: dict[str, dict[str, str]], deadline: str,
         unshare_sha256: str, identity: dict[str, str], producer16_job: str = '', producer48_job: str = '') -> dict:
    """Build a bounded candidate from actual authenticated assets, not a release."""
    from experiments.jobs.inbox import validate_spec

    required = {'bridge', 'receiver_isolation_config', 'native_cpu_receipt', 'denial_witness',
                'fidelity_gate', 'accepted16', 'ada_reference'}
    if action == 'score':
        required |= {'prepare16', 'prepare48', 'producer16_attestation', 'producer48_attestation'}
        if not producer16_job.startswith('2026') or not producer48_job.startswith('2026'):
            raise ValueError('actual producer jobs required for sealed-pixel scoring')
    elif action != 'prepare':
        raise ValueError('only prepare or score is supported')
    if set(assets) != required or not cf.digest_string(unshare_sha256):
        raise ValueError('exact reviewed asset inventory required')
    if (assets['accepted16']['sha256'] != cf.D8_SMOKE_SHA
            or assets['ada_reference']['sha256'] != adapter.REFERENCE_SHA256):
        raise ValueError('immutable historical or Ada reference differs')
    kwargs = {name: assets[name]['local'] for name in ('bridge', 'receiver_isolation_config',
                                                       'native_cpu_receipt', 'denial_witness')}
    kwargs.update({name + '_sha256': assets[name]['sha256'] for name in kwargs.copy()})
    admission(argparse.Namespace(**kwargs, unshare_sha256=unshare_sha256), identity)
    extra = [cf.asset(name, entry['remote'], entry['sha256'], Path(entry['local']), extract=False)
             for name, entry in assets.items()]
    result = cf.common_spec(base, extra, deadline)
    result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
    result['scale']['frames'] = {'smoke': 16, 'full': 48}
    result['scale']['proof_mode'] = {'smoke': 'ada-scale-' + action, 'full': 'ada-scale-' + action}
    result['scale']['namespace_timeout'] = {'smoke': 280, 'full': 580}
    result['arguments'] = ['outer', '--unshare-sha', unshare_sha256, '--timeout', '{namespace_timeout}', '--',
                           '{proof_mode}', *base['arguments'], '--unshare-sha256', unshare_sha256]
    for name, entry in assets.items():
        result['arguments'] += ['--' + name.replace('_', '-'), '{staged:' + name + '}',
                                '--' + name.replace('_', '-') + '-sha256', entry['sha256']]
    if action == 'score':
        result['arguments'] += ['--producer16-job', producer16_job, '--producer48-job', producer48_job]
    result.update(hosts=['gpu6'], gpu_models=['RTX 6000 Ada'], gpu_sharing={'mode': 'exclusive'},
                  single_cpu_campaign=True, budget_seconds=900, validator_seconds=30,
                  smoke={'seconds': 300, 'representative_basis':
                         'All six isolated 16-frame producer paths or sealed-pixel full scoring; conditional 48-frame forecast.'},
                  full={'seconds': 600}, validator=['{python}', '-m', 'experiments.tennis.k5d_ada_scale_worker', 'validate'])
    validate_spec(result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'score', 'validate'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone'):
        p.add_argument('--' + name)
    for name in ('bridge', 'receiver-isolation-config', 'native-cpu-receipt', 'denial-witness',
                 'fidelity-gate', 'accepted16', 'ada-reference', 'prepare16', 'prepare48',
                 'producer16-attestation', 'producer48-attestation'):
        p.add_argument('--' + name)
        p.add_argument('--' + name + '-sha256')
    p.add_argument('--unshare-sha256')
    p.add_argument('--producer16-job')
    p.add_argument('--producer48-job')
    p.add_argument('--frames', type=int)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == 'validate':
        validate()
    else:
        run(args)


if __name__ == '__main__':
    main()
