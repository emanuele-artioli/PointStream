"""Separate prospective Ada qualification; not a dispatch or scientific admission."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_coalesced as co, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_fleet as fleet, k5d_source_oracle as oracle
from experiments.tennis.k5d_real_parity import ADA_PROFILE, check_gpu_profile


def validate_source(root: Path, prior: Path, identity: dict[str, str], *, prepared: bool = True) -> None:
    if prepared:
        cp.verify(root, oracle.sha(root / 'checkpoint.json'), identity)
    original, labels = cf.reference16(prior)
    report = json.loads((root / 'pilot.json').read_text())
    source = json.loads((root / 'source-qualification.json').read_text())
    cost.compare_contract(original, labels, report, source, exact=False, capability=(8, 9))
    if report['frames'] != 16 or report['conversion']['yuv_sha256'] != original['conversion']['yuv_sha256']:
        raise ValueError('immutable source YUV or frame count differs')


def run(args: argparse.Namespace) -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    if os.environ['PS_STAGE'] not in ('smoke', 'full') or args.frames != 16:
        raise ValueError('only the 16-frame calibration clip is admitted')
    identity = cp.execution_identity()
    source_args = tuple(getattr(args, name, None) for name in
                        ('first_pair_bridge', 'first_pair_bridge_sha256', 'calibration_source', 'consumer_source'))
    if (getattr(args, 'gpu_profile', None) != ADA_PROFILE or not all(source_args)
            or any(getattr(args, name, False) for name in
                   ('shared_numerical_only', 'shared_low_memory', 'shared_parent8_child2'))):
        raise ValueError('fresh qualification requires exclusive Ada and complete source bridge')
    if os.environ['PS_STAGE'] == 'full':
        gate = json.loads((stage.parent / 'gate.json').read_text())
        if gate.get('passed') is not True or gate.get('validation', {}).get('checks', {}).get('conditional_cost_gate') is not True:
            raise ValueError('validated A cost gate required before B')
        cost_gate(stage.parent / 'smoke')
    args.same_job_receiver = True
    from experiments.tennis import k5d_ada_fresh_source_bridge as bridge
    bridge.verify_signed(Path(args.first_pair_bridge), args.first_pair_bridge_sha256,
                         Path(args.calibration_source), Path(args.consumer_source), identity)
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
    import torch
    device = check_gpu_profile(ADA_PROFILE, specification, dispatch, torch, os.environ)
    profile = ADA_PROFILE
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
        validate_source(prepared, prior, identity)
        checkpoint_sha = oracle.sha(prepared / 'checkpoint.json')
        data.update(checkpoint=str(prepared), output=str(scratch / 'scored'))
        co.score_all(argparse.Namespace(**data), checkpoint_sha, identity)
        scored = Path(data['output'])
        validate_source(scored / 'assembled', prior, identity, prepared=False)
    publish = scratch / 'publish'
    publish.mkdir()
    shutil.move(str(prepared), publish / 'prepared')
    shutil.move(str(scored), publish / 'scored')
    cp.write_new(publish / 'fresh.json', {'checkpoint_sha256': checkpoint_sha,
                 'execution_identity': identity, 'native_cpu_receipt_sha256': args.native_cpu_receipt_sha256,
                 'gpu_profile': profile, 'device': device,
                 'scope': 'fresh Ada 16-frame calibration; not an independent sample'})
    if os.environ['PS_STAGE'] == 'full':
        from experiments.tennis import k5d_ada_fresh_compare as fresh_compare
        smoke = stage.parent / 'smoke'
        smoke_staging = json.loads((smoke / 'staging.json').read_text())
        with (cost.unpack(Path(args.accepted16), cf.D8_SMOKE_SHA) as prior,
              cost.unpack(smoke / 'published.tar', smoke_staging['published']['sha256']) as first):
            validate_one(first, prior, smoke, identity)
            validate_one(publish, prior, stage, identity)
            a_device = json.loads((first / 'fresh.json').read_text())['device']
            b_device = json.loads((publish / 'fresh.json').read_text())['device']
            if a_device != b_device or not a_device.get('uuid'):
                raise ValueError('fresh executions must use the same physical GPU')
            comparison = fresh_compare.compare(first / 'scored' / 'assembled',
                                               publish / 'scored' / 'assembled', prior)
            comparison.update(a_published_sha256=smoke_staging['published']['sha256'],
                              job_id=stage.parent.name, device=a_device)
        cp.write_new(publish / 'repeat.json', comparison)



def validate_one(output: Path, prior: Path, stage: Path, identity: dict[str, str]) -> None:
    from experiments.tennis import k5d_real_parity as parity

    saved = json.loads((output / 'fresh.json').read_text())
    specification = json.loads((stage.parent / 'spec.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    parity.validate_gpu_evidence(ADA_PROFILE, specification, dispatch, saved['device'])
    entries = {item['name']: item for item in json.loads((stage / 'staging.json').read_text())['inputs']}
    if (saved['native_cpu_receipt_sha256'] != entries['native_cpu_receipt']['sha256']
            or saved['execution_identity'] != identity or saved['gpu_profile'] != ADA_PROFILE
            or saved['checkpoint_sha256'] != oracle.sha(output / 'prepared' / 'checkpoint.json')
            or not (output / 'prepared' / 'same-job-receiver.json').exists()):
        raise ValueError('fresh execution identity or receiver evidence differs')
    validate_source(output / 'prepared', prior, identity)
    parity.verify_scored(output, identity)
    validate_source(output / 'scored' / 'assembled', prior, identity, prepared=False)


def stage_receipt(stage: Path) -> tuple[float, float]:
    import math

    execution = json.loads((stage / 'execution.json').read_text())
    prefix = 'K5D_NAMESPACE_CANDIDATE '
    lines = [json.loads(line.removeprefix(prefix)) for line in (stage / 'command.log').read_text().splitlines()
             if line.startswith(prefix)]
    if (execution['exit_code'] != 0 or len(lines) != 1 or lines[0].get('reason') != 'complete'
            or lines[0].get('status') != 0 or lines[0].get('complete_descendant_wait') is not True
            or lines[0]['namespace_init']['namespace_pid'] != 1):
        raise ValueError('complete native namespace receipt required')
    wall = execution['seconds']
    cpu = lines[0]['children_cpu_seconds'] + lines[0]['init_cpu_seconds']
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (wall, cpu)):
        raise ValueError('invalid complete-stage cost')
    return float(wall), float(cpu)


def cost_gate(stage: Path) -> dict:
    import time

    wall, cpu = stage_receipt(stage)
    clock = json.loads((stage / 'dispatch.json').read_text())['campaign_clock']
    now = time.monotonic()
    if not 0 < clock['started'] <= clock['phase_started'] <= now < clock['budget_end']:
        raise ValueError('invalid campaign monotonic clock')
    estimate = 1.5 * wall + 50
    remaining = clock['budget_end'] - now
    if wall > 300 or cpu > 300 or estimate > 500 or remaining < estimate + 130:
        raise ValueError('complete A cost or remaining campaign cannot support B')
    return {'a_wall_seconds': wall, 'a_cpu_seconds': cpu, 'b_forecast_seconds': estimate,
            'remaining_seconds': remaining, 'validator_and_reserve_seconds': 130,
            'setup_before_a_seconds': clock['phase_started'] - clock['started']}


def validate() -> None:
    target = Path(os.environ['PS_VALIDATION_PATH'])
    try:
        stage = Path(os.environ['PS_STAGE_DIR'])
        staging = json.loads((stage / 'staging.json').read_text())
        entries = {item['name']: item for item in staging['inputs']}
        with (cost.unpack(Path(entries['accepted16']['path']), cf.D8_SMOKE_SHA) as prior,
              cost.unpack(stage / 'published.tar', staging['published']['sha256']) as output):
            validate_one(output, prior, stage, cp.execution_identity())
        gate = cost_gate(stage)
        target.write_text(json.dumps({'passed': True, 'checks': {'fresh_a_complete': True,
                          'conditional_cost_gate': True}, 'forecast': gate}) + '\n')
    except Exception as exc:
        target.write_text(json.dumps({'passed': False, 'checks': {'fresh_a_complete': False}, 'error': str(exc)}) + '\n')
        raise


def spec(base: dict, *, assets: dict[str, dict[str, str]], deadline: str,
         unshare_sha256: str, identity: dict[str, str]) -> dict:
    from experiments.tennis import k5d_real_parity as parity
    from experiments.jobs.inbox import validate_spec

    from experiments.tennis import k5d_ada_fresh_source_bridge as bridge
    from experiments.tennis import k5d_scoped_namespace as namespace

    required = {'fidelity_gate', 'accepted16', 'native_cpu_receipt', 'first_pair_bridge',
                'calibration_source', 'consumer_source'}
    if set(assets) != required or assets['accepted16']['sha256'] != cf.D8_SMOKE_SHA or not cf.digest_string(unshare_sha256):
        raise ValueError('exact source, reference, gate and CPU proof assets required')
    bridge.verify_signed(Path(assets['first_pair_bridge']['local']), assets['first_pair_bridge']['sha256'],
                         Path(assets['calibration_source']['local']), Path(assets['consumer_source']['local']), identity)
    cpu = json.loads(Path(assets['native_cpu_receipt']['local']).read_text())
    if (cpu.get('audited') is not True or cpu.get('passed') is not True
            or cpu.get('scope') != 'native CPU PID containment and accounting' or cpu.get('gpu_used') is not False
            or cpu.get('consumer_execution_identity') != identity or cpu.get('unshare_sha256') != unshare_sha256
            or cpu.get('namespace_wrapper_sha256') != oracle.sha(Path(namespace.__file__))):
        raise ValueError('audited actual CPU proof does not match namespace')
    extra = [cf.asset(name, item['remote'], item['sha256'], Path(item['local']), extract=False)
             for name, item in assets.items()]
    result = cf.common_spec(base, extra, deadline)
    result['entrypoint'] = ['-m', 'experiments.tennis.k5d_scoped_namespace']
    result['arguments'] = ['outer', '--unshare-sha', unshare_sha256, '--timeout', '{namespace_timeout}', '--', '{proof_mode}',
                           *base['arguments'], '--fidelity-gate', '{staged:fidelity_gate}',
                           '--fidelity-gate-sha256', assets['fidelity_gate']['sha256'], '--accepted16', '{staged:accepted16}',
                           '--native-cpu-receipt', '{staged:native_cpu_receipt}',
                           '--native-cpu-receipt-sha256', assets['native_cpu_receipt']['sha256'], '--unshare-sha256', unshare_sha256,
                           '--gpu-profile', ADA_PROFILE, '--first-pair-bridge', '{staged:first_pair_bridge}',
                           '--first-pair-bridge-sha256', assets['first_pair_bridge']['sha256'],
                           '--calibration-source', '{staged:calibration_source}', '--consumer-source', '{staged:consumer_source}']
    result['scale']['frames'] = {'smoke': 16, 'full': 16}
    result.update(hosts=['gpu5', 'gpu6'], gpu_models=[parity.GPU_PROFILES[ADA_PROFILE][0]],
                  gpu_sharing={'mode': 'exclusive'})
    result['scale']['proof_mode'] = {'smoke': 'ada-fresh16', 'full': 'ada-fresh16'}
    result['scale']['namespace_timeout'] = {'smoke': 580, 'full': 580}
    result.update(budget_seconds=1200, single_cpu_campaign=True, validator_seconds=30,
                  smoke={'seconds': 600, 'representative_basis':
                         'Independent full 16-frame six-point Ada preparation, receiving and scoring; conditional cost gate.'},
                  full={'seconds': 600}, validator=['{python}', '-m', 'experiments.tennis.k5d_ada_fresh', 'validate'])
    validate_spec(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'validate'))
    for name in ('archive', 'balltrack-code', 'balltrack-weight', 'wasb-code', 'wasb-weights', 'wasb-config',
                 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone', 'fidelity-gate',
                 'fidelity-gate-sha256', 'accepted16', 'native-cpu-receipt', 'native-cpu-receipt-sha256',
                 'unshare-sha256', 'first-pair-bridge', 'first-pair-bridge-sha256', 'calibration-source', 'consumer-source'):
        parser.add_argument('--' + name)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--gpu-profile', choices=(ADA_PROFILE,))
    args = parser.parse_args()
    if args.action == 'validate':
        validate()
        return
    for name in ('shared_numerical_only', 'shared_low_memory', 'shared_parent8_child2'):
        setattr(args, name, False)
    required = ('archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
                'image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone', 'fidelity_gate',
                'fidelity_gate_sha256', 'accepted16', 'native_cpu_receipt', 'native_cpu_receipt_sha256',
                'unshare_sha256', 'first_pair_bridge', 'first_pair_bridge_sha256', 'calibration_source', 'consumer_source')
    if args.frames != 16 or any(getattr(args, name) is None for name in required):
        parser.error('all pinned 16-frame source, bridge, model and CPU-proof inputs required')
    run(args)


if __name__ == '__main__':
    main()
