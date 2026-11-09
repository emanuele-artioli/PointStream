"""Proposed real 16-frame K5d parity proof; not a dispatch or admission.

The fixed worker reuses the frozen numerical producer and shared scorer, but
runs them with a new strict execution identity under the namespace launcher.
It compares the complete real path to the pre-existing 16-frame d8 reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_coalesced as co, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle

SHARED_PROFILE = 'k5d-parity16-shared-numerical-v1'
LOW_MEMORY_PROFILE = 'k5d-parity16-shared-numerical-low-memory-v1'
LOW_MEMORY_LIMIT_MIB = 6144
LOW_MEMORY_PARENT_MIB = 4096
LOW_MEMORY_CHILD_MIB = 2048
SHARED_LIMIT_MIB = 16384
PER_PROCESS_MIB = 8192
SHARED_BASIS = 'Exact numerical parity only; shared execution is not timing, scale, or cost evidence.'


def spec(base: dict, *, assets: dict[str, dict[str, str]], deadline: str, unshare_sha256: str,
         identity: dict[str, str], shared_numerical: bool = False, shared_low_memory: bool = False) -> dict:
    """Build a bounded candidate only after actual assets are available.

    This does not reserve resources or approve a fleet submission.
    """
    if shared_numerical and shared_low_memory:
        raise ValueError('shared profiles are mutually exclusive')
    if set(assets) != {'fidelity_gate', 'accepted16', 'native_cpu_receipt'}:
        raise ValueError('fidelity gate, accepted reference and audited CPU receipt required')
    if not cf.digest_string(unshare_sha256) or assets['accepted16']['sha256'] != cf.D8_SMOKE_SHA:
        raise ValueError('unshare or historical reference identity differs')
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items()]
    cpu = json.loads(Path(assets['native_cpu_receipt']['local']).read_text())
    from experiments.tennis import k5d_scoped_namespace as namespace
    if (cpu.get('audited') is not True or cpu.get('passed') is not True
            or cpu.get('scope') != 'native CPU PID containment and accounting' or cpu.get('gpu_used') is not False
            or cpu.get('consumer_execution_identity') != identity
            or cpu.get('unshare_sha256') != unshare_sha256
            or cpu.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))):
        raise ValueError('audited actual CPU proof does not match namespace')
    result = cf.common_spec(base, extra, deadline)
    result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
    result['arguments'] = ['outer', '--unshare-sha', unshare_sha256, '--timeout', '{namespace_timeout}', '--', '{proof_mode}',
                           *base['arguments'], '--fidelity-gate', '{staged:fidelity_gate}',
                           '--fidelity-gate-sha256', assets['fidelity_gate']['sha256'], '--accepted16', '{staged:accepted16}',
                           '--native-cpu-receipt', '{staged:native_cpu_receipt}',
                           '--native-cpu-receipt-sha256', assets['native_cpu_receipt']['sha256'],
                           '--unshare-sha256', unshare_sha256]
    result['scale']['frames'] = {'smoke': 16, 'full': 16}
    result['scale']['proof_mode'] = {'smoke': 'parity16', 'full': 'parity-noop'}
    result['scale']['namespace_timeout'] = {'smoke': 580, 'full': 4}
    result.update(gpu_models=['RTX A6000'], gpu_sharing={'mode': 'exclusive'}, budget_seconds=800,
                  validator_seconds=30, smoke={'seconds': 600, 'representative_basis':
                  'Actual 16-frame preparation and all six native replay, metric and judge paths with exact saved parity.'},
                  full={'seconds': 5}, validator=['{python}', '-m', 'experiments.tennis.k5d_real_parity', 'validate'])
    if shared_numerical:
        result['arguments'].append('--shared-numerical-only')
        result.update(k5d_shared_parity_profile=SHARED_PROFILE, gpu_memory_mib=SHARED_LIMIT_MIB,
                      gpu_sharing={'mode': 'shared', 'basis': SHARED_BASIS, 'margin_mib': 8192},
                      contention={'policy': 'continue', 'basis': SHARED_BASIS, 'resume_attempts': 0})
    if shared_low_memory:
        result['arguments'].append('--shared-low-memory')
        result.update(k5d_shared_parity_profile=LOW_MEMORY_PROFILE, gpu_memory_mib=LOW_MEMORY_LIMIT_MIB,
                      gpu_sharing={'mode': 'shared', 'basis': SHARED_BASIS, 'margin_mib': 8192},
                      contention={'policy': 'continue', 'basis': SHARED_BASIS, 'resume_attempts': 0})
    from experiments.jobs.inbox import validate_spec
    validate_spec(result)
    return result


def check_resource_profile(specification: dict, dispatch: dict, *, requested: bool, environ: Mapping[str, str], low_memory: bool = False) -> None:
    """Bind the opt-in to the authenticated fleet declaration and claim."""
    if low_memory and not requested:
        raise ValueError('low-memory profile requires explicit opt-in')
    profile = LOW_MEMORY_PROFILE if low_memory else SHARED_PROFILE
    limit = LOW_MEMORY_LIMIT_MIB if low_memory else SHARED_LIMIT_MIB
    if not requested:
        if specification.get('k5d_shared_parity_profile') is not None or environ.get('PS_GPU_SHARING') == 'shared':
            raise ValueError('shared parity requires explicit opt-in')
        return
    if (specification.get('k5d_shared_parity_profile') != profile
            or specification.get('gpu_memory_mib') != limit
            or specification.get('gpu_models') != ['RTX A6000']
            or specification.get('gpu_sharing') != {'mode': 'shared', 'basis': SHARED_BASIS, 'margin_mib': 8192}
            or specification.get('contention') != {'policy': 'continue', 'basis': SHARED_BASIS, 'resume_attempts': 0}
            or dispatch.get('gpu_sharing') != specification['gpu_sharing']
            or dispatch.get('contention_policy') != specification['contention']
            or environ.get('PS_GPU_SHARING') != 'shared'
            or environ.get('PS_GPU_MEMORY_LIMIT_MIB') != str(limit)
            or not environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
            or environ.get('PS_CLAIMED_GPU_UUID') != environ.get('CUDA_VISIBLE_DEVICES')
            or dispatch.get('environment', {}).get('gpu', {}).get('uuid') != environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('shared numerical parity claim or policy differs')


def apply_parent_limit(torch: Any, *, low_memory: bool = False) -> dict:
    """Cap this allocator before loading models; DCVC children use a separate cap."""
    limit = LOW_MEMORY_LIMIT_MIB if low_memory else SHARED_LIMIT_MIB
    budget = LOW_MEMORY_PARENT_MIB if low_memory else PER_PROCESS_MIB
    cuda = torch.cuda
    if cuda.memory_allocated(0) != 0:
        raise ValueError('shared parent allocated CUDA before its limit')
    total = cuda.get_device_properties(0).total_memory
    if total < limit * 2**20:
        raise ValueError('shared A6000 capacity is insufficient')
    cuda.set_per_process_memory_fraction(budget * 2**20 / total, 0)
    from experiments.tennis.k5d_scoped_namespace import host_group_identity
    host = host_group_identity(Path(os.environ['PS_STAGE_DIR']).parent / 'run' / 'status.json')
    return {'role': 'parent', 'pid': os.getpid(), 'pgid': os.getpgrp(), 'budget_mib': budget,
            'total_bytes': total, 'fraction': budget * 2**20 / total, 'allocated_before': 0,
            'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], **host}


def configure_child_environment(saved: MutableMapping[str, str], current: Mapping[str, str], *, low_memory: bool = False) -> None:
    """Propagate the already-authenticated opt-in into K5's frozen child env."""
    profile = LOW_MEMORY_PROFILE if low_memory else SHARED_PROFILE
    limit = LOW_MEMORY_LIMIT_MIB if low_memory else SHARED_LIMIT_MIB
    if (current.get('PS_K5D_SHARED_NUMERICAL') != profile
            or current.get('PS_GPU_SHARING') != 'shared'
            or current.get('PS_GPU_MEMORY_LIMIT_MIB') != str(limit)
            or not current.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
            or current.get('PS_CLAIMED_GPU_UUID') != current.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('shared child environment differs')
    for key in ('PS_K5D_SHARED_NUMERICAL', 'PS_GPU_SHARING', 'PS_GPU_MEMORY_LIMIT_MIB', 'PS_K5D_RESOURCE_RECEIPTS',
                'CUDA_VISIBLE_DEVICES', 'PS_CLAIMED_GPU_UUID'):
        saved[key] = current[key]


def validate_resource_receipts(directory: Path, *, low_memory: bool = False) -> list[dict]:
    paths = sorted(directory.glob('*.json'))
    if not (directory / 'parent.json').is_file() or len(paths) < 2:
        raise ValueError('shared parent and DCVC allocator receipts required')
    records = []
    for path in paths:
        value = json.loads(path.read_text())
        role = 'parent' if path.name == 'parent.json' else 'dcvc'
        budget = (LOW_MEMORY_PARENT_MIB if role == 'parent' else LOW_MEMORY_CHILD_MIB) if low_memory else PER_PROCESS_MIB
        limit = LOW_MEMORY_LIMIT_MIB if low_memory else SHARED_LIMIT_MIB
        total = value.get('total_bytes')
        if (value.get('role') != role or not isinstance(total, int) or total < limit * 2**20
                or value.get('budget_mib') != budget or value.get('allocated_before') != 0
                or value.get('fraction') != budget * 2**20 / total
                or not isinstance(value.get('pid'), int) or value['pid'] <= 0
                or not isinstance(value.get('pgid'), int) or value['pgid'] < 0
                or not isinstance(value.get('gpu_uuid'), str) or not value['gpu_uuid'].startswith('GPU-')
                or any(type(value.get(key)) is not int or value[key] <= 0 for key in
                       ('host_pid', 'host_pgid', 'start_ticks', 'campaign_pgid'))
                or value['host_pgid'] != value['campaign_pgid']
                or (role == 'dcvc' and path.name != f"child-{value['pid']}.json")):
            raise ValueError('invalid shared allocator receipt')
        records.append(value)
    if len({(item['host_pid'], item['start_ticks']) for item in records}) != len(records):
        raise ValueError('duplicate allocator process identity')
    expected = {(action, f'prepared-work/dcvc-{point}/plan.json')
                for kind, point in enc.POINTS if kind == 'dcvc' for action in ('encode', 'decode')}
    expected.update(('decode', f'coalesced-dcvc-{point}/receiver-plan.json')
                    for kind, point in enc.POINTS if kind == 'dcvc')
    expected.update(('decode', f'prepared-work/receiver-dcvc-{point}/receiver-plan.json')
                    for kind, point in enc.POINTS if kind == 'dcvc')
    actual = [(item.get('action'), item.get('plan')) for item in records if item['role'] == 'dcvc']
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('shared allocator receipts do not cover every DCVC action')
    return records


def run(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.environ['PS_STAGE'] != 'smoke' or args.frames != 16:
        raise ValueError('only the 16-frame calibration clip is admitted')
    identity = cp.execution_identity()
    cpu = json.loads(Path(args.native_cpu_receipt).read_text())
    if (oracle.sha(Path(args.native_cpu_receipt)) != args.native_cpu_receipt_sha256
            or cpu.get('schema') != 1 or cpu.get('audited') is not True or cpu.get('passed') is not True
            or cpu.get('scope') != 'native CPU PID containment and accounting' or cpu.get('gpu_used') is not False
            or cpu.get('consumer_execution_identity') != identity
            or not isinstance(cpu.get('job_id'), str) or not cpu['job_id'].startswith('2026')
            or cpu.get('all_owned_pids_absent') is not True or cpu.get('independent_cpu_match') is not True
            or cpu.get('cases') != ['normal', 'timeout', 'sigterm', 'sigkill']):
        raise ValueError('audited native CPU proof required')
    from experiments.tennis import k5d_scoped_namespace as namespace
    if (cpu.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))
            or cpu.get('unshare_sha256') != args.unshare_sha256):
        raise ValueError('CPU proof does not bind this exact namespace wrapper')
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    specification = json.loads((stage.parent / 'spec.json').read_text())
    if args.shared_numerical_only and args.shared_low_memory:
        raise ValueError('shared profiles are mutually exclusive')
    shared = args.shared_numerical_only or args.shared_low_memory
    check_resource_profile(specification, dispatch, requested=shared, environ=os.environ, low_memory=args.shared_low_memory)
    import torch
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000'
            or tuple(torch.cuda.get_device_capability(0)) != (8, 6)):
        raise ValueError('A6000 CUDA device required')
    if shared:
        receipt_dir = scratch / 'shared-resource-receipts'
        receipt_dir.mkdir()
        cp.write_new(receipt_dir / 'parent.json', apply_parent_limit(torch, low_memory=args.shared_low_memory))
        os.environ['PS_K5D_SHARED_NUMERICAL'] = LOW_MEMORY_PROFILE if args.shared_low_memory else SHARED_PROFILE
        os.environ['PS_K5D_RESOURCE_RECEIPTS'] = str(receipt_dir)
        # K5 snapshots its child environment at import time; update only this
        # reviewed opt-in so native DCVC encode/decode workers receive the cap.
        from experiments.tennis import k5
        configure_child_environment(k5.STAGE_ENV, os.environ, low_memory=args.shared_low_memory)
    if oracle.sha(cf.AUDIT_PATH) != cf.AUDIT_SHA or oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError('historical audit, source manifest or exposure differs')
    data = vars(args).copy()
    for name, digest in fleet.EXPECTED.items():
        data[name + '_sha256'] = digest
    data.update(clip=cost.CLIP, start_frame=0, manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA,
                code_revision=fleet.snapshot_head(stage), output=scratch / 'prepared',
                feasibility_no_metrics=False, cost_pilot=True)
    data['snapshot_identity'] = {key: dispatch['code'].get(key) for key in
                                 ('git_head', 'source_sha256', 'snapshot_sha256', 'tracked_worktree_patch_sha256', 'included_untracked_sha256')}
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    for name in ('wasb_code', 'balltrack_code'):
        archive = Path(data[name]).parent / Path(entries[name]['source']).name
        if oracle.sha(archive) != entries[name]['sha256']:
            raise ValueError('model code archive differs')
        expected = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(Path(data[name])) != expected:
            raise ValueError('model code tree differs')
        data[name + '_sha256'] = expected
    for name, binary in (('ffmpeg', 'ffmpeg'), ('encoder', 'SvtAv1EncApp'), ('decoder', 'dav1d')):
        data[name] = str(Path(sys.prefix) / 'bin' / binary)
    cf.prepare_baseline()
    enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                      purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=16)
    with cost.unpack(Path(args.accepted16), cf.D8_SMOKE_SHA) as prior:
        cp.prepare(argparse.Namespace(**data), identity)
        prepared = Path(data['output'])
        cf.prepare_parity(prepared, prior, identity)
        checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
        data.update(checkpoint=str(prepared), output=str(scratch / 'scored'))
        co.score_all(argparse.Namespace(**data), checkpoint_sha, identity)
        scored = Path(data['output'])
        comparison = cp.compare_equivalence(prior, scored / 'assembled', oracle.sha(prior / 'pilot.json'))
    publish = scratch / 'publish'
    publish.mkdir()
    shutil.move(str(prepared), publish / 'prepared')
    shutil.move(str(scored), publish / 'scored')
    if shared:
        shutil.move(str(receipt_dir), publish / 'shared-resource-receipts')
    cp.write_new(publish / 'parity.json', {'comparison': comparison, 'checkpoint_sha256': checkpoint_sha,
                 'execution_identity': identity, 'native_cpu_receipt_sha256': args.native_cpu_receipt_sha256,
                 'scope': 'exact 16-frame calibration parity; not an independent sample',
                 **({'shared_numerical_only': True, 'timing_evidence': False} if shared else {})})


def verify_scored(root: Path, identity: dict[str, str]) -> None:
    """Rebuild the output from all six saved points and require all arms ran."""
    import tempfile

    prepared, proof = root / 'prepared', root / 'scored'
    checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
    cp.verify(prepared, checkpoint_sha, identity)
    record = json.loads((proof / 'coalesced-validation.json').read_text())
    points = record['points']
    expected = [f'point-{codec}-{point}.json' for codec, point in enc.POINTS]
    if (record['checkpoint_sha256'] != checkpoint_sha or record['execution_identity'] != identity
            or [item['name'] for item in points] != expected):
        raise ValueError('six scored checkpoint proofs differ')
    files = []
    for item in points:
        path = proof / item['name']
        if oracle.sha(path) != item['sha256']:
            raise ValueError('point proof hash differs')
        files.append({'path': str(path), 'sha256': item['sha256']})
    with tempfile.TemporaryDirectory(prefix='k5d-parity-assemble-', dir=os.environ.get('TMPDIR')) as folder:
        assembled = Path(folder) / 'assembled'
        result = cp.assemble(prepared, checkpoint_sha, files, assembled, identity)
        if result != record['result'] or oracle.sha(assembled / 'pilot.json') != oracle.sha(proof / 'assembled' / 'pilot.json'):
            raise ValueError('saved assembly differs from scored points')
    report = json.loads((proof / 'assembled' / 'pilot.json').read_text())
    source = json.loads((proof / 'assembled' / 'source-qualification.json').read_text())
    if (len(source['rows']) != 16 or any(row['label_kind'] != 1 for row in source['rows'])
            or any(point['streak']['guard_off'] is not False for point in report['points'])):
        raise ValueError('calibration source or guard differs')
    events = json.loads((proof / 'timing.json').read_text())['events']
    required = {f'{kind}.{codec}.{point}.{arm}' for codec, point in enc.POINTS
                for kind in ('metrics', 'judges') for arm in ('baseline', 'streak')}
    if any(len(found := [e for e in events if e['name'] == name]) != 1 or found[0]['completed'] is not True
           for name in required):
        raise ValueError('metric or judge timing event missing or duplicated')


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
        published_sha = json.loads((stage / 'staging.json').read_text())['published']['sha256']
        with (cost.unpack(Path(entries['accepted16']['path']), cf.D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', published_sha) as output):
            identity = cp.execution_identity()
            saved = json.loads((output / 'parity.json').read_text())
            spec_for_receipts = json.loads((stage.parent / 'spec.json').read_text())
            low_memory = spec_for_receipts.get('k5d_shared_parity_profile') == LOW_MEMORY_PROFILE
            resource_records = (validate_resource_receipts(output / 'shared-resource-receipts', low_memory=low_memory)
                                if saved.get('shared_numerical_only') is True else [])
            verify_scored(output, identity)
            cf.prepare_parity(output / 'prepared', prior, identity)
            comparison = cp.compare_equivalence(prior, output / 'scored' / 'assembled', oracle.sha(prior / 'pilot.json'))
            if (saved['comparison'] != comparison or saved['execution_identity'] != identity
                    or saved['checkpoint_sha256'] != oracle.sha(output / 'prepared' / 'checkpoint.json')
                    or saved['native_cpu_receipt_sha256'] != entries['native_cpu_receipt']['sha256']):
                raise ValueError('published parity and execution identity differ')
        prefix = 'K5D_NAMESPACE_CANDIDATE '
        lines = [json.loads(line.removeprefix(prefix)) for line in (stage / 'command.log').read_text().splitlines()
                 if line.startswith(prefix)]
        if (len(lines) != 1 or lines[0].get('reason') != 'complete' or lines[0].get('status') != 0
                or lines[0].get('complete_descendant_wait') is not True
                or lines[0]['namespace_init']['namespace_pid'] != 1):
            raise ValueError('actual namespace execution proof missing')
        values = [lines[0]['children_cpu_seconds'], lines[0]['init_cpu_seconds']]
        if resource_records:
            worker = lines[0]['worker_host_identity']
            group = json.loads((stage.parent / 'run' / 'status.json').read_text())['process_group_id']
            dispatch = json.loads((stage / 'dispatch.json').read_text())
            gpu_uuid = dispatch['environment']['gpu']['uuid']
            parent = next(item for item in resource_records if item['role'] == 'parent')
            if (any(parent[key] != worker[key] for key in ('host_pid', 'start_ticks', 'host_pgid', 'campaign_pgid'))
                    or any(item['host_pgid'] != group or item['gpu_uuid'] != gpu_uuid for item in resource_records)):
                raise ValueError('shared allocator host ownership differs')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
            raise ValueError('actual namespace CPU accounting invalid')
        specification = json.loads((stage.parent / 'spec.json').read_text())
        profile = specification.get('k5d_shared_parity_profile')
        if profile not in (None, SHARED_PROFILE, LOW_MEMORY_PROFILE):
            raise ValueError('unknown shared profile')
        is_shared = profile in (SHARED_PROFILE, LOW_MEMORY_PROFILE)
        if is_shared and (saved.get('shared_numerical_only') is not True or saved.get('timing_evidence') is not False):
            raise ValueError('shared parity must disclaim timing evidence')
        target.write_text(json.dumps({'passed': True, 'checks': {'actual_preparation_reference': True,
                          'all_six_native_and_pixel_parity': True, 'all_metric_and_judge_arrays': True,
                          'new_strict_checkpoint_identity': True},
                          **({'shared_numerical_only': True, 'timing_evidence': False,
                              'scale_evidence': False, 'cost_evidence': False} if is_shared else {})}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'actual_parity': False}, 'error': str(exc)}) + '\n')
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'validate', 'noop'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'fidelity-gate',
                 'fidelity-gate-sha256', 'accepted16', 'native-cpu-receipt', 'native-cpu-receipt-sha256',
                 'unshare-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--frames', type=int)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument('--shared-numerical-only', action='store_true')
    profiles.add_argument('--shared-low-memory', action='store_true')
    args = parser.parse_args()
    if args.action == 'validate':
        validate()
    elif args.action == 'noop':
        if os.environ.get('PS_STAGE') != 'full':
            raise ValueError('noop is only permitted in full stage')
        output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        output.mkdir()
        cp.write_new(output / 'no-op.json', {'scope': 'no repeated parity work', 'stage': 'full'})
    else:
        if args.frames != 16 or any(getattr(args, name) is None for name in (
                'archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
                'image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone', 'fidelity_gate',
                'fidelity_gate_sha256', 'accepted16', 'native_cpu_receipt', 'native_cpu_receipt_sha256', 'unshare_sha256')):
            parser.error('all pinned 16-frame source, model and CPU-proof inputs required')
        run(args)


if __name__ == '__main__':
    main()
