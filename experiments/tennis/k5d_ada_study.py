"""Explicit gpu6 Ada registered-study admission; never an execution release.

Historical A6000 bridges and the Ada 16/48 worker remain unchanged. Ada study
calibration must use the exact consumer source/archive/environment. All drafts
still require independent root approval of their immutable hashes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any

from experiments.jobs import inbox
from experiments.tennis import k5d_ada_fresh as fresh
from experiments.tennis import k5d_ada_scale_adapter as adapter, k5d_ada_scale_worker as worker
from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_control_bridge as bridge, k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as enc, k5d_real_parity as parity
from experiments.tennis import k5d_same_job_receiver as receiver, k5d_source_oracle as oracle

A6000 = 'a6000'
ADA = 'rtx6000ada-gpu6-registered'
ASSETS = ('receiver_isolation_config', 'ada_reference_bridge', 'denial_witness')


def profile(record: dict[str, Any]) -> str:
    value = record.get('registered_gpu_profile', A6000)
    if value not in (A6000, ADA):
        raise ValueError('unknown registered-study GPU profile')
    return str(value)


def hardware(record: dict[str, Any]) -> tuple[str, list[int]]:
    return ('RTX 6000 Ada', [8, 9]) if profile(record) == ADA else ('RTX A6000', [8, 6])


def same_source(calibration: dict[str, str], consumer: dict[str, str]) -> None:
    if (set(consumer) != {'source_sha256', 'snapshot_sha256', 'environment_sha256'}
            or any(not worker.digest(v) for v in consumer.values()) or calibration != consumer):
        raise ValueError('Ada calibration requires identical frozen source, archive and environment')


def source_bridge(old: Path, calibration: dict[str, str], new: Path, consumer: dict[str, str]) -> dict[str, Any]:
    same_source(calibration, consumer)
    old_sha, old_files = bridge.manifest(old, calibration['snapshot_sha256'])
    new_sha, new_files = bridge.manifest(new, consumer['snapshot_sha256'])
    if old_sha != consumer['source_sha256'] or new_sha != old_sha or old_files != new_files:
        raise ValueError('Ada archived source manifests differ')
    return {'schema': 1, 'registered_gpu_profile': ADA,
            'calibration_execution_identity': calibration, 'consumer_execution_identity': consumer,
            'calibration_snapshot_sha256': calibration['snapshot_sha256'],
            'consumer_snapshot_sha256': consumer['snapshot_sha256'],
            'changed_paths': [], 'unchanged_files': len(new_files)}


def device(spec: dict[str, Any], dispatch: dict[str, Any]) -> dict[str, Any]:
    environment = dispatch.get('environment', {})
    gpu = environment.get('gpu', {})
    actual = {'name': gpu.get('name'), 'capability': [8, 9], 'uuid': gpu.get('uuid')}
    if (spec.get('hosts') != ['gpu6'] or environment.get('alias') != 'gpu6'
            or environment.get('host', '').split('.')[0] != 'gpu6'
            or spec.get('cpu_threads') != 1 or spec.get('single_cpu_campaign') is not True
            or ('capability' in gpu and gpu['capability'] != [8, 9])):
        raise ValueError('Ada registered study requires actual gpu6 and one CPU')
    adapter._uuid(gpu.get('uuid'), prefixed=True)
    parity.validate_gpu_evidence(parity.ADA_PROFILE, spec, dispatch, actual)
    return actual


def bindings(gate: dict[str, Any], args: argparse.Namespace, identity: dict[str, str]) -> None:
    if profile(gate) != ADA or getattr(args, 'registered_gpu_profile', A6000) != ADA:
        raise ValueError('explicit Ada registered-study profile required')
    for name in ASSETS:
        expected = gate.get(name + '_sha256')
        if not worker.digest(expected) or getattr(args, name + '_sha256', None) != expected:
            raise ValueError('Ada gate asset binding differs: ' + name)
    # The calibration-to-study bridge is separate from the historical own-Ada
    # reference bridge. Never substitute one identity or bridge for the other.
    values = vars(args).copy()
    values.update(bridge=args.ada_reference_bridge, bridge_sha256=args.ada_reference_bridge_sha256,
                  native_cpu_receipt_sha256=gate['confirmation_qualification']['native_cpu_receipt_sha256'],
                  unshare_sha256=gate['confirmation_qualification']['unshare_sha256'])
    worker.admission(argparse.Namespace(**values), identity)


def registered_producer(root: Path, identity: dict[str, str], *, job: str, clip: str, frames: int,
                        config: Path, config_sha: str, dispatch_gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    from experiments.tennis import k5d_full_study as study

    selected = study.selection()
    if (clip not in selected or frames != selected[clip]['label_rows']
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', job)):
        raise ValueError('Ada producer must be a complete registered timeline')
    sealed = cp.verify(root, oracle.sha(root / 'checkpoint.json'), identity, require_same_job_receiver=True)
    report = json.loads((root / 'pilot.json').read_text())
    marker = receiver.verify(root, report, identity)
    if (sealed['clip'] != clip or sealed['frames'] != frames or marker['producer_job'] != job
            or marker['producer_stage'] != 'full' or marker['frames'] != frames):
        raise ValueError('Ada registered producer job, stage or timeline differs')
    gpu = adapter.persisted_device(root, marker, dispatch_gpu=dispatch_gpu)
    configuration = worker.checked(config, config_sha)
    if configuration.get('execution_identity') != identity:
        raise ValueError('Ada receiver configuration source differs')
    runtime = configuration['profiles']['gpu6']
    isolated = importlib.import_module('experiments.tennis.k5d_isolated_receiver')
    context = {'job_id': job, 'stage': 'full', 'execution_identity': identity, 'gpu': gpu}
    if [(p['codec'], p['point']) for p in report['points']] != list(enc.POINTS):
        raise ValueError('all six fresh receiver points required')
    for point in report['points']:
        receipt = json.loads((root / f"{point['codec']}-{point['point']}" / 'receiver-evidence/receiver.json').read_text())
        verified = isolated.verify_receipt(receipt, expected_context=context, expected_config_sha256=config_sha)
        if (verified.get('profile_sha256') != worker.canonical_sha(runtime)
                or verified.get('runtime_inventory_sha256') != worker.canonical_sha(runtime['runtime_files'])
                or not worker.digest(verified.get('actual_policy_sha256'))):
            raise ValueError('Ada actual filesystem receiver policy differs')
    return marker


def calibration(inputs: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    """Independently read actual paired fleet metadata/publications; no timings supplied by caller."""
    from experiments.tennis import k5d_full_cost_builder as builder

    same_source(inputs['execution_identity'], identity)
    if set(inputs['stages']) != {'prepare16', 'prepare48', 'score16', 'score48'}:
        raise ValueError('four actual paired Ada stages required')
    jobs = inputs['calibration_jobs']
    if (set(jobs) != {'prepare', 'score'} or jobs['prepare'] == jobs['score']
            or any(not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', j) for j in jobs.values())):
        raise ValueError('distinct actual paired Ada producer and scorer jobs required')
    result: dict[str, Any] = {'schema': 1, 'registered_gpu_profile': ADA, 'execution_identity': identity,
                              'calibration_jobs': jobs, 'stages': {}, 'config_sha256': None}
    for frames in (16, 48):
        records: dict[str, Any] = {}
        for action in ('prepare', 'score'):
            key = action + str(frames)
            entry = inputs['stages'][key]
            meta, published_sha = builder.metadata(entry)
            spec, dispatch, staging, execution = (meta['records'][x] for x in ('spec', 'dispatch', 'staging', 'execution'))
            stage = 'smoke' if frames == 16 else 'full'
            command = dispatch['command']
            mode = 'ada-scale-' + action
            if (entry['job'] != jobs[action] or entry['stage'] != stage or command.count(mode) != 1
                    or spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace']
                    or spec['arguments'][:1] != ['outer'] or '--accounted-child' in command
                    or dispatch['code']['spec_sha256'] != inbox.digest(spec)
                    or dispatch['code']['source_sha256'] != identity['source_sha256']
                    or dispatch['code']['snapshot_sha256'] != identity['snapshot_sha256']
                    or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
                    or staging['environment']['sha256'] != identity['environment_sha256']
                    or dispatch['input_identity'] != spec['inputs'] or dispatch['staged_inputs'] != staging['inputs']
                    or command != inbox.phase_command(spec, stage, staging['inputs'], command[0])
                    or spec['scale']['frames'] != {'smoke': 16, 'full': 48}
                    or execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                    or execution.get('contention') != []):
                raise ValueError('Ada measured fleet source, command, stage or completion differs')
            gpu = device(spec, dispatch)
            args = worker.parser().parse_args([action, *command[command.index(mode) + 1:]])
            if args.frames != frames:
                raise ValueError('Ada measured frame scope differs')
            worker.admission(args, identity)
            if result['config_sha256'] not in (None, args.receiver_isolation_config_sha256):
                raise ValueError('Ada calibration configurations differ')
            result['config_sha256'] = args.receiver_isolation_config_sha256
            seconds, cpu_seconds, command_log_sha = stage_cost(Path(entry['directory']) / stage)
            if seconds != execution['seconds']:
                raise ValueError('Ada execution changed during measurement audit')
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('invalid observed Ada timing')
            archive = Path(inputs['archives'][key])
            if oracle.sha(archive) != published_sha:
                raise ValueError('Ada measured publication checksum differs')
            records[action] = (archive, published_sha, args, gpu)
            result['stages'][key] = {'job': entry['job'], 'stage': stage, 'seconds': seconds,
                                    'sha256': meta['entry']['sha256'], 'published_sha256': published_sha, 'gpu': gpu,
                                    'cpu_seconds': cpu_seconds, 'command_log_sha256': command_log_sha}
        prep, prep_sha, prep_args, prep_gpu = records['prepare']
        score, score_sha, score_args, score_gpu = records['score']
        if prep_gpu != score_gpu:
            raise ValueError('Ada measured producer/scorer UUID differs')
        if (getattr(score_args, 'producer' + str(frames) + '_job') != jobs['prepare']
                or getattr(score_args, 'prepare' + str(frames) + '_sha256') != prep_sha):
            raise ValueError('Ada measured score differs from actual producer')
        reference = inputs['reference16']
        if reference['sha256'] != adapter.REFERENCE_SHA256:
            raise ValueError('Ada original reference differs')
        with cost.unpack(prep, prep_sha) as prepared, cost.unpack(score, score_sha) as scored:
            marker = adapter.producer(prepared, identity, job=jobs['prepare'], stage=stage, frames=frames,
                                      config_sha256=prep_args.receiver_isolation_config_sha256,
                                      config_path=Path(prep_args.receiver_isolation_config))
            adapter.persisted_device(prepared, marker, dispatch_gpu=prep_gpu)
            worker.producer_attestation(score_args, identity, frames, marker, prepared)
            if frames == 16:
                adapter.prepare16(Path(reference['local']), prepared, identity)
            else:
                adapter.prepare48(Path(reference['local']), prepared, identity)
            worker.validate_score(scored, prepared, oracle.sha(prepared / 'checkpoint.json'), identity)
            full_metrics(scored / 'assembled', frames)
            if frames == 16:
                accepted = inputs['accepted16']
                if accepted['sha256'] != cf.D8_SMOKE_SHA:
                    raise ValueError('Ada metric-control reference differs')
                with cost.unpack(Path(accepted['local']), accepted['sha256']) as prior:
                    adapter.score16(Path(reference['local']), scored / 'assembled', prior)
    if len({inbox.digest(v['gpu']) for v in result['stages'].values()}) != 1:
        raise ValueError('Ada paired calibration GPU UUID changed')
    return result


def validate_calibration(receipt: dict[str, Any]) -> None:
    """Cheap checks on a root-pinned measured audit; raw audit is performed before approval."""
    same_source(receipt['calibration_execution_identity'], receipt['consumer_execution_identity'])
    audit = receipt['ada_calibration']
    if (receipt.get('gpu_model') != 'RTX 6000 Ada' or receipt.get('gpu_capability') != [8, 9]
            or audit.get('schema') != 1 or audit.get('registered_gpu_profile') != ADA or audit.get('execution_identity') != receipt['consumer_execution_identity']
            or audit.get('calibration_jobs') != receipt['calibration_jobs']
            or receipt.get('calibration_audit_sha256') != inbox.digest(audit)
            or receipt.get('budget_reconciled') is not True or not worker.digest(audit.get('config_sha256'))
            or set(audit.get('stages', {})) != {'prepare16', 'prepare48', 'score16', 'score48'}):
        raise ValueError('Ada measured calibration audit differs')
    if (receipt.get('calibration_archives') != {k: v['published_sha256'] for k, v in audit['stages'].items()}
            or receipt.get('calibration_execution_sha256') != {k: v['sha256']['execution'] for k, v in audit['stages'].items()}
            or set(receipt.get('score_timing_sha256', {})) != {'16', '48'}
            or any(not worker.digest(v) for v in receipt['score_timing_sha256'].values())):
        raise ValueError('Ada archive, execution or score-timing bindings differ')
    jobs = receipt['calibration_jobs']
    if (set(jobs) != {'prepare', 'score'} or jobs['prepare'] == jobs['score']
            or len({inbox.digest(v['gpu']) for v in audit['stages'].values()}) != 1):
        raise ValueError('Ada paired calibration jobs or GPU UUID differs')
    for action in ('prepare', 'score'):
        c = receipt['cost'][action]
        measured = []
        for frames in (16, 48):
            key = action + str(frames)
            record = audit['stages'][key]
            gpu = record['gpu']
            adapter._uuid(gpu.get('uuid'), prefixed=True)
            seconds = record['seconds']
            if (record['job'] != receipt['calibration_jobs'][action]
                    or record['stage'] != ('smoke' if frames == 16 else 'full')
                    or gpu['name'] != adapter.GPU_NAME or gpu['capability'] != [8, 9]
                    or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', record['job'])
                    or set(record['sha256']) != {'spec', 'dispatch', 'staging', 'execution'}
                    or any(not worker.digest(v) for v in record['sha256'].values())
                    or not worker.digest(record['published_sha256']) or not worker.digest(record.get('command_log_sha256'))
                    or isinstance(record.get('cpu_seconds'), bool) or not isinstance(record.get('cpu_seconds'), (int, float))
                    or not math.isfinite(record['cpu_seconds']) or record['cpu_seconds'] <= 0
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds <= 0 or c['seconds' + str(frames)] != seconds):
                raise ValueError('Ada observed stage evidence or cost differs')
            measured.append(seconds)
        from experiments.tennis import k5d_full_cost_builder as builder
        if builder.model(measured[0], measured[1], c['margin'], c['reserve_seconds']) != c:
            raise ValueError('Ada affine model differs from actual measurements')


def qualification(gate: dict[str, Any], proof: dict[str, Any], identity: dict[str, str]) -> None:
    if (profile(gate) != ADA or gate.get('gpu_models') != ['RTX 6000 Ada']
            or gate.get('gpu_capability') != [8, 9] or gate.get('hosts') != ['gpu6']
            or any(not worker.digest(gate.get(name + '_sha256')) for name in ASSETS)
            or proof.get('consumer_execution_identity') != identity or proof.get('exclusive') is not True
            or proof.get('timing_contaminated') is not False or proof.get('all_six_exact') is not True
            or proof.get('actual_filesystem_denial_passed') is not True
            or proof.get('config_sha256') != gate['receiver_isolation_config_sha256']
            or proof.get('ada_reference_bridge_sha256') != gate['ada_reference_bridge_sha256']
            or proof.get('denial_witness_sha256') != gate['denial_witness_sha256']):
        raise ValueError('source-bound Ada native/score/denial qualification required')


def score_producer(gate: dict[str, Any], clip: str, certificate: dict[str, Any] | None = None) -> dict[str, Any]:
    if gate.get('release') == 'full-study':
        if certificate is None or profile(certificate) != ADA:
            raise ValueError('Ada score requires matching root certificate')
        producer = certificate['targets'][clip]['score']['producer']
    else:
        producer = gate.get('ada_producers', {}).get(clip)
    if not isinstance(producer, dict):
        raise ValueError('root-approved exact Ada producer required')
    job = producer.get('job_id', '')
    expected = cf.DATA_ROOT / 'jobs/fleet/inbox' / job / 'full/published.tar'
    if (not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', job)
            or producer.get('remote') != str(expected)
            or any(not worker.digest(producer.get(k)) for k in ('archive_sha256', 'checkpoint_sha256'))
            or producer.get('config_sha256') != gate['receiver_isolation_config_sha256']):
        raise ValueError('Ada producer job, archive, checkpoint or config differs')
    return producer


def score_binding(gate: dict[str, Any], args: argparse.Namespace, identity: dict[str, str]) -> dict[str, Any]:
    import os
    from experiments.tennis import k5d_full_admission as admission

    cert = admission.read_certificate(Path(args.certificate), gate['certificate_sha256']) if args.certificate else None
    producer = score_producer(gate, args.clip, cert)
    stage = Path(os.environ['PS_STAGE_DIR'])
    entries = [x for x in json.loads((stage / 'staging.json').read_text())['inputs'] if x['name'] == 'checkpoint']
    if (len(entries) != 1 or entries[0]['source'] != producer['remote']
            or entries[0]['sha256'] != producer['archive_sha256']
            or args.checkpoint_sha256 != producer['checkpoint_sha256']):
        raise ValueError('staged Ada score input differs from approved producer')
    args.receiver_producer_job = producer['job_id']
    return producer


def configure(args: argparse.Namespace, gate: dict[str, Any], identity: dict[str, str], stage: Path) -> None:
    import os

    spec = json.loads((stage.parent / 'spec.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    gpu = device(spec, dispatch)
    if args.mode == 'full':
        import torch
        actual = parity.check_gpu_profile(parity.ADA_PROFILE, spec, dispatch, torch, os.environ)
        if actual != gpu:
            raise ValueError('Ada actual GPU differs from dispatch')
    args.same_job_receiver = True
    args.isolated_receiver = True
    args.receiver_producer_context = {'job_id': stage.parent.name, 'stage': 'full',
                                      'execution_identity': identity, 'gpu': gpu}
    args.receiver_protected_paths = {
        'source_archive': [Path(args.archive)],
        'ground_truth': [worker.fleet.SOURCE, worker.fleet.EXPOSURE],
        'private_detector': [Path(args.balltrack_weight), Path(args.wasb_config),
                             Path(args.wasb_weights) / 'wasb_tennis_best.pth.tar'],
    }


def full_metrics(root: Path, frames: int) -> None:
    result = enc.validate(root)
    if result.get('full_metrics') is not True or result.get('points') != 6 or result.get('frames') != frames:
        raise ValueError('Ada study requires all six full metrics and judges on every original frame')


def known_calibration_job(job: str, state: str, directory: Path, receipt: dict[str, Any]) -> bool:
    """Recognize only completed jobs already physically bound by the root-pinned measured audit."""
    if job not in receipt['calibration_jobs'].values():
        return False
    validate_calibration(receipt)
    if state != 'complete':
        raise ValueError('known Ada calibration is not canonically complete')
    records = [v for v in receipt['ada_calibration']['stages'].values() if v['job'] == job]
    if len(records) != 2 or {r['stage'] for r in records} != {'smoke', 'full'}:
        raise ValueError('known Ada calibration needs both actual stages')
    for record in records:
        if oracle.sha(directory / record['stage'] / 'command.log') != record['command_log_sha256']:
            raise ValueError('known Ada calibration command log changed')
        for name, path in (('spec', directory / 'spec.json'),
                           ('dispatch', directory / record['stage'] / 'dispatch.json'),
                           ('staging', directory / record['stage'] / 'staging.json'),
                           ('execution', directory / record['stage'] / 'execution.json')):
            if oracle.sha(path) != record['sha256'][name]:
                raise ValueError('known Ada calibration physical records changed')
    return True


def stage_cost(stage: Path) -> tuple[float, float, str]:
    """Full namespace lifetime, not a nested numerical timer or a successful exit alone."""
    wall, cpu = fresh.stage_receipt(stage)
    return wall, cpu, oracle.sha(stage / 'command.log')


def known_qualification_job(job: str, state: str, directory: Path, proof: dict[str, Any], identity: dict[str, str]) -> bool:
    """Only the two root-pinned successful qualification jobs may precede the study."""
    modes = {'actual-cpu-denial': {'smoke': 'ada-pgid-cpu', 'full': 'ada-pgid-cpu-noop'},
             'ada-first-witness': {'smoke': 'ada-first-witness', 'full': 'ada-first-noop'}}
    jobs = proof.get('qualification_jobs')
    if (proof.get('approved') is not True or proof.get('consumer_execution_identity') != identity
            or proof.get('all_six_exact') is not True or proof.get('actual_filesystem_denial_passed') is not True
            or not isinstance(jobs, dict) or set(jobs) != set(modes)
            or any(not isinstance(r, dict) or r.get('state') != 'complete' or r.get('execution_identity') != identity
                   or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(r.get('job_id')))
                   for r in jobs.values()) or len({r['job_id'] for r in jobs.values()}) != 2):
        raise ValueError('explicit root-approved Ada qualification job map required')
    matches = [(kind, record) for kind, record in jobs.items() if record['job_id'] == job]
    if not matches:
        return False
    kind, record = matches[0]
    if state != 'complete' or record.get('state') != 'complete' or record.get('execution_identity') != identity:
        raise ValueError('Ada qualification job status or identity differs')
    hashes = record['sha256']
    if any(oracle.sha(directory / (name + '.json')) != hashes[name] for name in ('spec', 'ready')):
        raise ValueError('Ada qualification spec or ready changed')
    spec, ready = (json.loads((directory / (name + '.json')).read_text()) for name in ('spec', 'ready'))
    if (spec.get('entrypoint') != ['-m', 'experiments.tennis.k5d_scoped_namespace']
            or spec.get('arguments', [])[:1] != ['outer'] or spec.get('scale', {}).get('proof_mode') != modes[kind]
            or ready.get('source_sha256') != identity['source_sha256']
            or ready.get('snapshot_sha256') != identity['snapshot_sha256'] or ready.get('spec_sha256') != inbox.digest(spec)):
        raise ValueError('Ada qualification literal mode or ready identity differs')
    for stage in ('smoke', 'full'):
        root = directory / stage
        for name in ('dispatch', 'staging', 'execution', 'command.log'):
            filename = name if name == 'command.log' else name + '.json'
            if oracle.sha(root / filename) != hashes[stage][name]:
                raise ValueError('Ada qualification physical stage record changed')
        dispatch, staging, execution = (json.loads((root / (name + '.json')).read_text())
                                        for name in ('dispatch', 'staging', 'execution'))
        command, code = dispatch['command'], dispatch['code']
        if (code.get('source_sha256') != identity['source_sha256'] or code.get('snapshot_sha256') != identity['snapshot_sha256']
                or code.get('spec_sha256') != inbox.digest(spec)
                or dispatch['staged_environment']['sha256'] != identity['environment_sha256']
                or staging['environment']['sha256'] != identity['environment_sha256']
                or dispatch['input_identity'] != spec['inputs'] or dispatch['staged_inputs'] != staging['inputs']
                or command != inbox.phase_command(spec, stage, staging['inputs'], command[0])
                or execution.get('exit_code') != 0 or execution.get('timing_contaminated') is not False
                or execution.get('contention') != [] or execution.get('paused_seconds') != 0):
            raise ValueError('Ada qualification canonical execution or identity differs')
    return True
