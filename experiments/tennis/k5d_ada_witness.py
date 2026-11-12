"""Prospective CPU denial oracle and first isolated Ada witness fleet workers."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_ada_scale_adapter as adapter, k5d_ada_scale_worker as scale
from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_cost_pilot as cost, k5d_encoder as enc
from experiments.tennis import k5d_fleet as fleet, k5d_real_parity as parity
from experiments.tennis import k5d_source_oracle as oracle


def cpu_denial_approval(args: argparse.Namespace, identity: dict[str, str]) -> None:
    record = scale.checked(Path(args.cpu_denial), args.cpu_denial_sha256)
    context = record.get('producer_context')
    if (record.get('schema') != 1 or record.get('approved') is not True
            or record.get('actual_filesystem_denial_passed') is not True
            or record.get('consumer_execution_identity') != identity
            or record.get('config_sha256') != args.receiver_isolation_config_sha256
            or not scale.digest(record.get('publication_sha256')) or record.get('codec') != 'svt-av1-ivf'
            or not isinstance(context, dict) or context.get('execution_identity') != identity
            or context.get('gpu') != {'kind': 'cpu'} or context.get('stage') != 'smoke'
            or not str(context.get('job_id', '')).startswith('2026') or record.get('host') != 'gpu6'):
        raise ValueError('root-approved actual SVT CPU denial oracle required')
    isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
    proof = isolated.verify_receipt(record.get('receipt'), expected_context=context,
                                    expected_config_sha256=args.receiver_isolation_config_sha256)
    config = scale.checked(Path(args.receiver_isolation_config), args.receiver_isolation_config_sha256)
    profile = config['profiles']['gpu6']
    if (not isinstance(proof, dict) or proof.get('profile_sha256') != scale.canonical_sha(profile)
            or proof.get('runtime_inventory_sha256') != scale.canonical_sha(profile['runtime_files'])
            or proof.get('actual_policy_sha256') != record.get('actual_policy_sha256')
            or not scale.digest(record.get('actual_policy_sha256'))):
        raise ValueError('CPU denial receipt or approved inventory differs')


def context(stage: Path, identity: dict[str, str], gpu: dict[str, Any]) -> dict[str, Any]:
    return {'job_id': stage.parent.name, 'stage': stage.name, 'execution_identity': identity, 'gpu': gpu}


def cpu(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if os.environ.get('PS_STAGE') != 'smoke' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only smoke required')
    identity = cp.execution_identity()
    scale.base_admission(args, identity)
    protected_raw = scale.checked(Path(args.protected), args.protected_sha256)
    categories = ('source_archive', 'source_frame', 'ground_truth', 'encoder_state', 'private_detector')
    if any(not isinstance(protected_raw.get(key), list) or not protected_raw[key] for key in categories):
        raise ValueError('five real protected categories required')
    protected = {key: [Path(value) for value in values] for key, values in protected_raw.items()}
    if any(not path.is_absolute() or not path.is_file() or not os.access(path, os.R_OK)
           for values in protected.values() for path in values):
        raise ValueError('protected paths must be real readable files')
    isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
    gpu = {'kind': 'cpu'}
    actual_context = context(stage, identity, gpu)
    output = scratch / 'publish'
    receipt = isolated.cpu_oracle(Path(args.wire), Path(args.resources), output,
        wire_sha256=args.wire_sha256, map_sha256=args.resources_sha256,
        expected_yuv_sha256=args.expected_yuv_sha256, expected_rgb_sha256=args.expected_rgb_sha256,
        config_path=Path(args.receiver_isolation_config), config_sha256=args.receiver_isolation_config_sha256,
        protected_paths=protected, producer_context=actual_context)
    proof = isolated.verify_receipt(receipt, expected_context=actual_context,
                                    expected_config_sha256=args.receiver_isolation_config_sha256)
    if not isinstance(proof, dict) or not scale.digest(proof.get('actual_policy_sha256')):
        raise ValueError('actual CPU receiver denial proof missing')


def witness(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.environ.get('PS_STAGE') != 'smoke' or args.frames != 16:
        raise ValueError('first witness admits only the six-point 16-frame smoke')
    identity = cp.execution_identity()
    scale.base_admission(args, identity)
    cpu_denial_approval(args, identity)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    specification = json.loads((stage.parent / 'spec.json').read_text())
    import torch
    device = parity.check_gpu_profile(parity.ADA_PROFILE, specification, dispatch, torch, os.environ)
    if (oracle.sha(cf.AUDIT_PATH) != cf.AUDIT_SHA or oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA
            or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA):
        raise ValueError('source or exposure identity differs')
    data = scale.data_args(args, stage, scratch, dispatch)
    data.receiver_isolation_config = Path(args.receiver_isolation_config)
    data.receiver_producer_context = context(stage, identity, device)
    data.receiver_protected_paths = {
        'source_archive': [Path(args.archive)], 'ground_truth': [fleet.SOURCE, fleet.EXPOSURE],
        'private_detector': [Path(args.balltrack_weight), Path(args.wasb_config),
                             Path(args.wasb_weights) / 'wasb_tennis_best.pth.tar'],
    }
    enc.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED['archive'],
                      purpose='48-frame-cost-pilot', clip=cost.CLIP, frames=16)
    cp.prepare(data, identity)
    prepared = Path(data.output)
    adapter.producer(prepared, identity, job=stage.parent.name, stage='smoke', frames=16,
                     config_sha256=args.receiver_isolation_config_sha256,
                     config_path=Path(args.receiver_isolation_config))
    adapter.prepare16(Path(args.ada_reference), prepared, identity)


def noop() -> None:
    if os.environ.get('PS_STAGE') != 'full':
        raise ValueError('no-op is permitted only in full stage')
    output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
    output.mkdir()
    cp.write_new(output / 'no-op.json', {'scope': 'no repeated witness work'})


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        if os.environ['PS_STAGE'] != 'smoke':
            raise ValueError('smoke validator required')
        staging = json.loads((stage / 'staging.json').read_text())
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        modes = [(i, x) for i, x in enumerate(dispatch['command'])
                 if x in ('ada-denial-cpu', 'ada-first-witness')]
        if len(modes) != 1:
            raise ValueError('fixed denial worker mode required')
        index, mode = modes[0]
        args = parser().parse_args(['cpu' if mode == 'ada-denial-cpu' else 'witness', *dispatch['command'][index + 1:]])
        identity = cp.execution_identity()
        scale.base_admission(args, identity)
        with cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output:
            if args.action == 'cpu':
                isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
                receipt = json.loads((output / 'receiver.json').read_text())
                actual_context = context(stage, identity, {'kind': 'cpu'})
                proof = isolated.verify_receipt(receipt, expected_context=actual_context,
                                                expected_config_sha256=args.receiver_isolation_config_sha256)
                if (not isinstance(proof, dict) or not scale.digest(proof.get('actual_policy_sha256'))
                        or receipt.get('wire_sha256') != args.wire_sha256
                        or receipt.get('resource_map_sha256') != args.resources_sha256
                        or receipt.get('decoded_yuv_sha256') != args.expected_yuv_sha256
                        or receipt.get('rendered_rgb_sha256') != args.expected_rgb_sha256
                        or oracle.sha(output / 'native.yuv') != args.expected_yuv_sha256
                        or oracle.sha(output / 'rendered.rgb') != args.expected_rgb_sha256):
                    raise ValueError('actual CPU denial or decoded pixels differ')
            else:
                cpu_denial_approval(args, identity)
                marker = adapter.producer(output, identity, job=stage.parent.name, stage='smoke', frames=16,
                                          config_sha256=args.receiver_isolation_config_sha256,
                                          config_path=Path(args.receiver_isolation_config))
                spec = json.loads((stage.parent / 'spec.json').read_text())
                device = adapter.persisted_device(output, marker, dispatch_gpu=dispatch['environment']['gpu'])
                parity.validate_gpu_evidence(parity.ADA_PROFILE, spec, dispatch, device)
                adapter.prepare16(Path(args.ada_reference), output, identity)
        from experiments.tennis import k5d_ada_fresh as fresh
        wall, cpu_seconds = fresh.stage_receipt(stage)
        target.write_text(json.dumps({'passed': True, 'checks': {'actual_denial': True, 'native_pixels': True},
                                      'wall_seconds': wall, 'cpu_seconds': cpu_seconds}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'actual_denial': False}, 'error': str(exc)}) + '\n')
        raise


def spec(base: dict, *, action: str, assets: dict[str, dict[str, str]], deadline: str,
         unshare_sha256: str, identity: dict[str, str], expected_yuv_sha256: str = '',
         expected_rgb_sha256: str = '') -> dict:
    """Build a bounded CPU oracle or first GPU witness, not an authorization."""
    from experiments.jobs.inbox import validate_spec

    required = {'bridge', 'receiver_isolation_config', 'native_cpu_receipt'}
    if action == 'cpu':
        required |= {'wire', 'resources', 'protected'}
        if not scale.digest(expected_yuv_sha256) or not scale.digest(expected_rgb_sha256):
            raise ValueError('authenticated expected CPU oracle pixels required')
    elif action == 'witness':
        required |= {'cpu_denial', 'fidelity_gate', 'ada_reference'}
        if assets.get('ada_reference', {}).get('sha256') != adapter.REFERENCE_SHA256:
            raise ValueError('immutable Ada calibration reference required')
    else:
        raise ValueError('only CPU oracle or first GPU witness is supported')
    if set(assets) != required or not scale.digest(unshare_sha256):
        raise ValueError('exact source, proof and isolation asset inventory required')
    kwargs = {name: assets[name]['local'] for name in ('bridge', 'receiver_isolation_config', 'native_cpu_receipt')}
    kwargs.update({name + '_sha256': assets[name]['sha256'] for name in kwargs.copy()})
    kwargs['unshare_sha256'] = unshare_sha256
    scale.base_admission(argparse.Namespace(**kwargs), identity)
    if action == 'witness':
        kwargs.update(cpu_denial=assets['cpu_denial']['local'], cpu_denial_sha256=assets['cpu_denial']['sha256'])
        cpu_denial_approval(argparse.Namespace(**kwargs), identity)
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items()]
    source_base = json.loads(json.dumps(base))
    if action == 'cpu':
        source_base['staged_inputs'] = []
        source_base['inputs'] = []
    result = cf.common_spec(source_base, extra, deadline)
    result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
    result['scale']['frames'] = {'smoke': 16, 'full': 16}
    result['scale']['proof_mode'] = ({'smoke': 'ada-denial-cpu', 'full': 'ada-denial-cpu-noop'} if action == 'cpu'
                                      else {'smoke': 'ada-first-witness', 'full': 'ada-first-noop'})
    result['scale']['namespace_timeout'] = ({'smoke': 160, 'full': 4} if action == 'cpu'
                                              else {'smoke': 400, 'full': 4})
    result['arguments'] = ['outer', '--unshare-sha', unshare_sha256, '--timeout', '{namespace_timeout}', '--',
                           '{proof_mode}', *([] if action == 'cpu' else base['arguments']), '--unshare-sha256', unshare_sha256]
    for name, item in assets.items():
        flag = name.replace('_', '-')
        result['arguments'] += ['--' + flag, '{staged:' + name + '}', '--' + flag + '-sha256', item['sha256']]
    if action == 'cpu':
        result['scale'].pop('frames', None)
        result['arguments'] += ['--expected-yuv-sha256', expected_yuv_sha256, '--expected-rgb-sha256', expected_rgb_sha256]
        result.pop('gpu_sharing', None)
        result.update(device='cpu', gpu_models=[], gpu_memory_mib=0, contention={'policy': 'stop', 'resume_attempts': 0},
                      budget_seconds=220, smoke={'seconds': 180, 'representative_basis':
                      'Actual isolated CPU decoding of a pinned conventional packet with five live denial categories.'})
    else:
        result.update(device='gpu', gpu_models=['RTX 6000 Ada'], gpu_sharing={'mode': 'exclusive'}, budget_seconds=500,
                      smoke={'seconds': 420, 'representative_basis':
                      'First same-job Ada denial witness, all six fresh native paths and immutable 16-frame comparison.'})
    result.update(hosts=['gpu6'], single_cpu_campaign=True, validator_seconds=30, full={'seconds': 5},
                  validator=['{python}', '-m', 'experiments.tennis.k5d_ada_witness', 'validate'])
    validate_spec(result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('cpu', 'witness', 'noop', 'validate'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone'):
        p.add_argument('--' + name)
    for name in ('bridge', 'receiver-isolation-config', 'native-cpu-receipt', 'cpu-denial', 'fidelity-gate',
                 'ada-reference', 'wire', 'resources', 'protected'):
        p.add_argument('--' + name)
        p.add_argument('--' + name + '-sha256')
    p.add_argument('--expected-yuv-sha256')
    p.add_argument('--expected-rgb-sha256')
    p.add_argument('--unshare-sha256')
    p.add_argument('--frames', type=int)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == 'validate':
        validate()
    elif args.action == 'noop':
        noop()
    elif args.action == 'cpu':
        cpu(args)
    else:
        witness(args)


if __name__ == '__main__':
    main()
