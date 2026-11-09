"""Post-seal W8 source, scoring and offline checks. No job release is implied.

The caller supplies independently reviewed producer and asset manifests by hash.
The scorer is never permitted to produce or modify receiver candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any

_CPU_LIMIT_BEFORE_IMPORTS = resource.getrlimit(resource.RLIMIT_CPU)
from experiments.tennis import w8_input_job as fixed  # noqa: E402
from experiments.tennis import w8_receiver_finish as finish  # noqa: E402
_CPU_LIMIT_AFTER_IMPORTS = resource.getrlimit(resource.RLIMIT_CPU)

SOURCE_KEYS = frozenset({'source_clip', 'receiver', 'ffmpeg', 'native_inventory'})
SCORE_KEYS = frozenset({'source_rgb', 'source_assessment', 'source_config', 'source_loader', 'source_validation',
                        'receiver', 'region_archive', 'judge_archive', 'lpips', 'dists'})
GOLDEN_KEYS = frozenset({'golden_base', 'golden_composite', 'golden_scores',
    *(f'golden_crop:{unit}' for unit in fixed.UNITS), *(f'golden_mask:{unit}' for unit in fixed.UNITS)})
OFFLINE_KEYS = GOLDEN_KEYS | {'scores', 'score_receipt', 'score_assessment', 'score_config', 'score_loader', 'score_validation'}


def cost_plan(*, phase: str, observed_wall: float, observed_cpu: float, staging: float,
              validation: float, cleanup: float, campaign_wall_cap: float,
              cpu_cap: float, gpu_cap: float) -> dict[str, Any]:
    """Prospective whole-stage admission; inputs are measured or explicit reserves."""
    values = (observed_wall, observed_cpu, staging, validation, cleanup,
              campaign_wall_cap, cpu_cap, gpu_cap)
    if (phase not in ('source', 'score', 'offline')
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
                   for x in values) or observed_wall <= 0 or observed_cpu <= 0):
        raise ValueError('measured assessment timing and bounded reserves required')
    # observed wall is the whole command: staging/validator/cleanup are disjoint.
    command = 1.5 * observed_wall
    total_wall = command + staging + validation + cleanup
    # One-core wall also bounds unmeasured same-affinity descendants.
    total_cpu = max(total_wall, 1.5 * observed_cpu + staging + validation + cleanup)
    total_gpu = total_wall if phase in ('source', 'score') else 0.0
    if (campaign_wall_cap <= 0 or cpu_cap <= 0 or total_wall > campaign_wall_cap
            or total_cpu > cpu_cap or total_gpu > gpu_cap):
        raise ValueError('assessment cost does not fit prospective stage caps')
    return {'phase': phase, 'scope': 'prospective assessment cost only', 'command_upper_seconds': command,
            'wall_upper_seconds': total_wall, 'cpu_upper_seconds': total_cpu, 'gpu_upper_seconds': total_gpu,
            'staging_seconds': staging, 'validation_seconds': validation, 'cleanup_seconds': cleanup,
            'safety_factor': 1.5, 'automatic_release': False}


def make_fleet_spec(*, template_local: Path, template_remote: str, deadline: str, host_name: str,
                    approved_producer_sha256: str, approved_asset_manifest_sha256: str,
                    approved_promotion_sha256: str, approved_ledger_sha256: str,
                    plan: dict[str, Any], command_wall: int, validator_wall: int,
                    campaign_wall: int, approved_previous_sha256: str | None = None) -> dict[str, Any]:
    """Build one separately reviewed post-seal phase; never promote another phase."""
    from experiments.jobs.inbox import validate_spec
    from experiments.tennis import w8_strict_packet_job as packet

    template = json.loads(template_local.read_text())
    # The independent root-pinned producer audit and actual full seals must pass
    # before even constructing a spec that can trigger privileged input hashing.
    if (not _sha(approved_producer_sha256)
            or template.get('producer_manifest', {}).get('sha256') != approved_producer_sha256):
        raise ValueError('root-pinned producer audit SHA required')
    candidate, producer = authenticate(template)
    phase = template.get('phase')
    check_assessment_promotion(template, producer=producer, host_name=host_name, plan=plan,
                               promotion_sha=approved_promotion_sha256, ledger_sha=approved_ledger_sha256,
                               previous_sha=approved_previous_sha256)
    if (not _sha(approved_asset_manifest_sha256)
            or template.get('asset_manifest', {}).get('sha256') != approved_asset_manifest_sha256):
        raise ValueError('root-pinned asset manifest SHA required')
    # Metadata only; never hash or open a source/golden artifact here.
    check_asset_manifest(template, phase)
    if phase in ('score', 'offline'):
        name = 'source_assessment' if phase == 'score' else 'score_assessment'
        output_name = 'source-rgb.npy' if phase == 'score' else 'scores.npz'
        output_key = 'source_rgb' if phase == 'score' else 'scores'
        prior = template.get('role_files', {}).get(name, {})
        if not _sha(approved_previous_sha256) or prior.get('sha256') != approved_previous_sha256:
            raise ValueError('root-pinned previous assessment required')
        prior_path = Path(prior['path'])
        if prior_path.is_symlink() or fixed.file_sha(prior_path) != approved_previous_sha256:
            raise ValueError('previous assessment identity differs')
        prior_config_key = 'source_config' if phase == 'score' else 'score_config'
        prior_config_row = template['role_files'][prior_config_key]
        prior_config = _manifest(Path(prior_config_row['path']), prior_config_row['sha256'])
        loader_key = 'source_loader' if phase == 'score' else 'score_loader'
        loader_record = prior_loader(template['role_files'][loader_key], prior_config)
        prior_gate_key = 'source_validation' if phase == 'score' else 'score_validation'
        prior_gate = template['role_files'][prior_gate_key]
        _manifest(Path(prior_gate['path']), prior_gate['sha256'])
        validate_prior_fleet_gate(Path(prior_gate['path']), phase='source' if phase == 'score' else 'score',
                                  assessment_sha=approved_previous_sha256, producer_sha=approved_producer_sha256)
        validate_prior_assessment(prior_path, phase='source' if phase == 'score' else 'score',
            producer_sha=approved_producer_sha256, candidate=candidate, output_name=output_name,
            output_sha=template['role_files'][output_key]['sha256'], prior_config=prior_config,
            prior_loader=loader_record)
    if (phase not in ('source', 'score', 'offline') or plan.get('phase') != phase
            or plan.get('automatic_release') is not False or host_name not in ('gpu5', 'gpu6')
            or any(type(x) is not int or x <= 0 for x in (command_wall, validator_wall, campaign_wall))
            or command_wall > 590 or campaign_wall > 2700
            or command_wall + validator_wall + 5 > campaign_wall
            or plan.get('wall_upper_seconds', float('inf')) > campaign_wall
            or plan.get('command_upper_seconds', float('inf')) + 10 + plan.get('cleanup_seconds', float('inf')) > command_wall
            or plan.get('validation_seconds', float('inf')) > validator_wall):
        raise ValueError('assessment phase or prospective fleet budget differs')
    caps = template['hard_ceilings']
    if (caps['cpu_seconds'] < campaign_wall + caps['external_cpu_seconds'] + caps['post_campaign_cpu_seconds']
            or (phase != 'offline' and caps['gpu_seconds'] < campaign_wall + caps['post_campaign_gpu_seconds'])):
        raise ValueError('inclusive hard ceilings do not cover campaign and external lifecycle')
    root = Path('/home/itec/emanuele/Datasets/pointstream-data')
    remote = Path(template_remote)
    if not remote.is_absolute() or not remote.is_relative_to(root) or '..' in remote.parts:
        raise ValueError('assessment template must be a registered data path')
    if (template.get('output') != '{assessment_output}' or template.get('phase') != phase
            or set(template.get('candidate_files', {})) != set(finish.CANDIDATE_KEYS)
            or set(template.get('role_files', {})) != set({'source': SOURCE_KEYS, 'score': SCORE_KEYS,
                                                           'offline': OFFLINE_KEYS}[phase])
            or type(template.get('cpu_seconds')) is not int or not 1 <= template['cpu_seconds'] <= 3600
            or (phase == 'source' and (type(template.get('child_cpu_seconds')) is not int
                or not 1 <= template['child_cpu_seconds'] <= 3600
                or type(template.get('wall_seconds')) is not int
                or not 1 <= template['wall_seconds'] <= math.ceil(plan['command_upper_seconds'])))):
        raise ValueError('exact candidate and role template required')
    rows = [template['producer_manifest'], template['asset_manifest'], template['promotion_manifest'], template['ledger'],
            *template['candidate_files'].values(), *template['role_files'].values()]
    identities = {(str(remote), fixed.file_sha(template_local))}
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute() or not path.is_relative_to(root) or '..' in path.parts or not _sha(row['sha256']):
            raise ValueError('assessment inputs must be registered and SHA-identified')
        identities.add((str(path), row['sha256']))
    spec = packet.make_spec(deadline)
    spec.update(hosts=[host_name], device='cpu' if phase == 'offline' else 'gpu',
                gpu_models=[] if phase == 'offline' else ['RTX 6000 Ada'],
                gpu_memory_mib=0 if phase == 'offline' else 24000,
                entrypoint=['-m', 'experiments.tennis.w8_witness_assessment'],
                arguments=['{phase}', '--template', '{staged:assessment_config}', '--template-sha256',
                           fixed.file_sha(template_local)], scale={'phase': {'smoke': 'fleet-run', 'full': 'fleet-noop'}},
                staged_inputs=[{'name': 'assessment_config', 'path': str(remote),
                                'sha256': fixed.file_sha(template_local), 'extract': False}],
                inputs=[{'path': path, 'sha256': sha} for path, sha in sorted(identities)],
                smoke={'seconds': command_wall, 'representative_basis': 'One separately gated post-seal assessment phase.'},
                full={'seconds': 5}, budget_seconds=campaign_wall, validator_seconds=validator_wall,
                validator=['{python}', '-m', 'experiments.tennis.w8_witness_assessment', 'fleet-validate'],
                w8_assessment_profile='w8-postseal-' + phase + '-v1', w8_assessment_plan=plan,
                w8_assessment_producer_manifest_sha256=approved_producer_sha256,
                w8_assessment_asset_manifest_sha256=approved_asset_manifest_sha256,
                w8_assessment_promotion_sha256=approved_promotion_sha256,
                w8_assessment_ledger_sha256=approved_ledger_sha256,
                w8_assessment_previous_sha256=approved_previous_sha256,
                w8_assessment_hard_ceilings=caps,
                w8_assessment_worker_wall=math.ceil(plan['command_upper_seconds']) + 10,
                stall_seconds=command_wall)
    if phase != 'offline':
        spec['gpu_sharing'] = {'mode': 'exclusive'}
    else:
        spec.pop('gpu_sharing', None)
    validate_spec(spec)
    return spec


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def check_assessment_promotion(config: dict[str, Any], *, producer: dict[str, Any], host_name: str,
                               plan: dict[str, Any], promotion_sha: str, ledger_sha: str,
                               previous_sha: str | None, source: dict[str, Any] | None = None) -> None:
    """Join the fresh root forecast to the unspent ledger and sealed producers."""
    from experiments.tennis import w8_witness_job as job

    promotion_item, ledger_item = config['promotion_manifest'], config['ledger']
    if (not _sha(promotion_sha) or not _sha(ledger_sha) or promotion_item.get('sha256') != promotion_sha
            or ledger_item.get('sha256') != ledger_sha):
        raise ValueError('root-pinned forecast and ledger identities required')
    gate = _manifest(Path(promotion_item['path']), promotion_sha)
    ledger = _manifest(Path(ledger_item['path']), ledger_sha)
    gpu, cpu = job.promotion_ledger_remaining(ledger)
    source = source or {'source_sha256': producer.get('source_sha256'),
                        'snapshot_sha256': producer.get('snapshot_sha256')}
    if (source.get('source_sha256') != producer.get('source_sha256')
            or source.get('snapshot_sha256') != producer.get('snapshot_sha256') or gate.get('host') != host_name):
        raise ValueError('assessment promotion source or host differs')
    producer_sha = config['producer_manifest']['sha256']
    job.check_promotion(gate, gate_sha=promotion_sha, expected_sha=promotion_sha, source=source,
                        cpu_gate_sha=config['cpu_proof_sha256'], prior_artifact_sha=producer_sha,
                        ledger_sha=ledger_sha, gpu_remaining=gpu, cpu_remaining=cpu)
    completed = gate['completed_artifact_sha256']
    producer_artifacts = producer.get('completed_artifact_sha256')
    required = {'unit0_full', 'unit1_full', 'native_sam_compose'}
    if (not isinstance(producer_artifacts, dict) or set(producer_artifacts) != required
            or any(completed.get(name) != producer_artifacts[name] or not _sha(producer_artifacts[name])
                   for name in required)):
        raise ValueError('assessment forecast lacks exact completed producer artifacts')
    phase = config.get('phase')
    if phase not in ('source', 'score', 'offline'):
        raise ValueError('invalid assessment promotion phase')
    if phase == 'score' and (not _sha(previous_sha) or gate.get('source_assessment_sha256') != previous_sha):
        raise ValueError('forecast does not bind the accepted source assessment')
    if phase == 'offline' and (not _sha(previous_sha) or completed.get('postseal_score') != previous_sha):
        raise ValueError('forecast does not bind the completed score assessment')
    component = gate['remaining_components']['offline_validation' if phase == 'offline' else 'postseal_score']
    caps = config.get('hard_ceilings')
    cap_names = {'gpu_seconds', 'cpu_seconds', 'external_cpu_seconds',
                 'post_campaign_cpu_seconds', 'post_campaign_gpu_seconds'}
    if (not isinstance(caps, dict) or set(caps) != cap_names
            or any(type(caps[name]) not in (int, float) or not math.isfinite(caps[name]) or caps[name] < 0
                   for name in cap_names) or caps['cpu_seconds'] <= 0
            or (phase == 'offline' and (caps['gpu_seconds'] != 0 or caps['post_campaign_gpu_seconds'] != 0))
            or (phase != 'offline' and caps['gpu_seconds'] <= 0)
            or caps['gpu_seconds'] > component['gpu_seconds'] or caps['cpu_seconds'] > component['cpu_seconds']):
        raise ValueError('inclusive assessment ceilings do not fit remaining-work forecast')
    if (plan.get('phase') != phase or plan.get('automatic_release') is not False
            or any(not isinstance(plan.get(key), (int, float)) or isinstance(plan[key], bool)
                   or not math.isfinite(plan[key]) or plan[key] < 0 for key in ('gpu_upper_seconds', 'cpu_upper_seconds'))
            or plan['gpu_upper_seconds'] > caps['gpu_seconds'] or plan['cpu_upper_seconds'] > caps['cpu_seconds']):
        raise ValueError('assessment phase does not fit remaining-work forecast')


def _manifest(path: Path, digest: str) -> dict[str, Any]:
    if not path.is_absolute() or not _sha(digest) or path.is_symlink() or fixed.file_sha(path) != digest:
        raise ValueError('reviewed manifest identity differs')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('reviewed manifest must be an object')
    return value


def authenticate(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Any]]:
    """Authenticate candidate and producer before opening any source/golden asset."""
    item = config['producer_manifest']
    manifest = _manifest(Path(item['path']), item['sha256'])
    if (manifest.get('schema') != 1 or manifest.get('scope') != 'independently reviewed W8 full candidate and finish'
            or manifest.get('passed') is not True or manifest.get('quality_claim') is not False
            or not _sha(manifest.get('source_sha256')) or not _sha(manifest.get('snapshot_sha256'))
            or not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('revision')))
            or not _sha(manifest.get('independent_audit_sha256'))
            or not isinstance(manifest.get('jobs'), dict)
            or set(manifest['jobs']) != {'unit0', 'unit1', 'finish'}
            or any(not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(x)) for x in manifest['jobs'].values())
            or len(set(manifest['jobs'].values())) != 3):
        raise ValueError('reviewed full producer provenance incomplete')
    files = config['candidate_files']
    expected = manifest.get('candidate_sha256')
    if (not isinstance(files, dict) or set(files) != set(finish.CANDIDATE_KEYS)
            or not isinstance(expected, dict) or set(expected) != set(files)
            or any(not _sha(expected[name]) or files[name].get('sha256') != expected[name] for name in files)):
        raise ValueError('candidate artifacts differ from reviewed producer')
    paths = finish._inputs({'files': files}, set(finish.CANDIDATE_KEYS))
    finish.validate_candidate_seals(paths)
    return paths, manifest


def role_files(config: dict[str, Any], role: str) -> dict[str, dict[str, str]]:
    """Only called after candidate authentication; bind review-pinned role assets."""
    check_asset_manifest(config, role)
    files = config['role_files']
    finish._inputs({'files': files}, set(files))
    return files


def check_asset_manifest(config: dict[str, Any], role: str) -> None:
    """Validate root-pinned archive/member provenance without opening source pixels."""
    from experiments.tennis import w8_receiver_score as scorer

    role_keys = {'source': SOURCE_KEYS, 'score': SCORE_KEYS, 'offline': OFFLINE_KEYS}
    if role not in role_keys:
        raise ValueError('unknown assessment role')
    item = config['asset_manifest']
    manifest = _manifest(Path(item['path']), item['sha256'])
    files = config['role_files']
    expected = manifest.get('role_sha256', {}).get(role)
    if (manifest.get('schema') != 1 or manifest.get('scope') != 'independently reviewed W8 post-seal assets'
            or manifest.get('historical_source_sha256') != scorer.RECEIVER_SHA
            or manifest.get('judge_archive_sha256') != scorer.JUDGE_ARCHIVE
            or manifest.get('region_archive_sha256') != scorer.REGION_ARCHIVE
            or not _sha(manifest.get('independent_audit_sha256'))
            or not isinstance(files, dict) or set(files) != set(role_keys[role])
            or not isinstance(expected, dict) or set(expected) != set(files)
            or any(not _sha(expected[name]) or files[name].get('sha256') != expected[name] for name in files)):
        raise ValueError('post-seal asset manifest differs')
    if role == 'offline':
        origins = manifest.get('golden_origins')
        if (not isinstance(origins, dict) or set(origins) != set(GOLDEN_KEYS)
                or any(not isinstance(origins[name], dict)
                       or origins[name].get('artifact_sha256') != expected[name]
                       or not _sha(origins[name].get('archive_sha256'))
                       or not _sha(origins[name].get('member_sha256')) for name in GOLDEN_KEYS)):
            raise ValueError('independently audited golden archive/member provenance differs')


def _gpu(expected_uuid: str) -> dict[str, Any]:
    import torch
    from experiments.tennis.w8_cuda_system_leaves import canonical_gpu_uuid

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one CUDA scoring device required')
    props = torch.cuda.get_device_properties(0)
    actual = canonical_gpu_uuid(str(getattr(props, 'uuid', '')))
    visible = canonical_gpu_uuid(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    claim = canonical_gpu_uuid(expected_uuid)
    if actual != visible or actual != claim or (props.major, props.minor) != (8, 9):
        raise ValueError('physical scoring GPU differs from fleet claim')
    return {'uuid': str(actual), 'visible_uuid': str(visible), 'name': props.name,
            'major': props.major, 'minor': props.minor, 'provider': 'torch CUDA'}


def validate_role_isolation(record: dict[str, Any], config: dict[str, Any], phase: str, loader: dict[str, Any]) -> None:
    from experiments.tennis import w8_clock_profile as clock, w8_cuda_system_leaves as policy

    isolation = record.get('role_isolation') or {}
    inputs = {str(Path(row['path']).resolve()) for row in
              [*config['candidate_files'].values(), *config['role_files'].values()]}
    denials, ns = isolation.get('read_only_inputs'), isolation.get('namespaces', {})
    abi = isolation.get('landlock_abi', 0)
    if (not isinstance(denials, dict) or set(denials) != inputs
            or any(value not in (1, 13) for value in denials.values())
            or type(abi) is not int or abi < 3 or isolation.get('handled_mask') != policy.handled_rights(abi)
            or isolation.get('output') != config['output']
            or not ns.get('user') or not ns.get('net') or ns['user'] == config['parent_user_namespace']
            or ns['net'] == config['parent_network_namespace']):
        raise ValueError('assessment role input grants or namespaces differ')
    expected_exec = []
    if phase == 'source':
        expected_exec = sorted([str(Path(config['role_files']['ffmpeg']['path']).resolve()),
                                str(Path(config['native_loader_path']).resolve())])
    if isolation.get('execution_grants') != expected_exec:
        raise ValueError('assessment executable grants differ')
    leaves = isolation.get('approved_leaves')
    if not isinstance(leaves, list):
        raise ValueError('assessment approved system leaves missing')
    policy.validate_leaf_receipt(leaves, loader)
    clock.validate_receipt(isolation.get('clock_profile'))


def validate_prior_assessment(path: Path, *, phase: str, producer_sha: str,
                              candidate: dict[str, Path], output_name: str, output_sha: str,
                              prior_config: dict[str, Any], prior_loader: dict[str, Any]) -> dict[str, Any]:
    record = json.loads(path.read_text())
    device = record.get('device') or {}
    if (record.get('scope') != 'post-seal W8 ' + phase or record.get('producer_manifest_sha256') != producer_sha
            or record.get('candidate_sha256') != {name: fixed.file_sha(item) for name, item in candidate.items()}
            or record.get('outputs', {}).get(output_name) != output_sha
            or device.get('provider') != 'torch CUDA' or not device.get('uuid')
            or device.get('uuid') != device.get('visible_uuid')
            or (device.get('major'), device.get('minor')) != (8, 9)):
        raise ValueError('prior assessment provenance or isolation differs')
    if (prior_config.get('phase') != phase
            or {name: row['sha256'] for name, row in prior_config['candidate_files'].items()}
            != record['candidate_sha256'] or prior_config['producer_manifest']['sha256'] != producer_sha
            or {name: row['sha256'] for name, row in prior_config['role_files'].items()} != record['role_sha256']):
        raise ValueError('prior assessment retained config differs')
    validate_role_isolation(record, prior_config, phase, prior_loader)
    return record


def validate_prior_fleet_gate(path: Path, *, phase: str, assessment_sha: str, producer_sha: str) -> None:
    gate = json.loads(path.read_text())
    if (gate.get('passed') is not True or gate.get('scope') != 'post-seal ' + phase + ' only'
            or gate.get('automatic_promotion') is not False or gate.get('checks', {}).get('independent_reopen') is not True
            or gate.get('assessment_sha256') != assessment_sha or gate.get('producer_manifest_sha256') != producer_sha
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(gate.get('job_id')))):
        raise ValueError('prior fleet assessment validation differs')


def prior_loader(row: dict[str, str], prior_config: dict[str, Any]) -> dict[str, Any]:
    if row.get('sha256') != prior_config.get('loader_receipt', {}).get('sha256'):
        raise ValueError('prior loader identity differs from retained config')
    return _manifest(Path(row['path']), row['sha256'])


def isolate(config: dict[str, Any], role: str, candidate: dict[str, Path],
            assets: dict[str, dict[str, str]], output: Path) -> dict[str, Any]:
    """Apply reviewed system leaves and exact read-only assessment inputs."""
    from experiments.tennis import w8_clock_profile as clock, w8_cuda_system_leaves as policy
    from experiments.tennis import w8_strict_host_cost_job as host

    ns = {name: os.readlink('/proc/self/ns/' + name) for name in ('user', 'net')}
    if (ns['user'] == config['parent_user_namespace'] or ns['net'] == config['parent_network_namespace']
            or sorted(name for _, name in socket.if_nameindex()) != ['lo']):
        raise ValueError('assessment requires fresh user and network namespaces')
    runtime = Path(config['runtime_root']).resolve(strict=True)
    expected_runtime = Path('/local/users/emanuele/pointstream/cache') / host.MAIN_ENV / 'tree'
    if runtime != expected_runtime.resolve(strict=True):
        raise ValueError('assessment MAIN runtime differs')
    source = Path(__file__).resolve().parents[2]
    files = [*candidate.values(), *(Path(row['path']) for row in assets.values())]
    resolved = [path.resolve(strict=True) for path in files]
    if any(path == output or output.is_relative_to(path) or path.is_relative_to(output) for path in resolved):
        raise ValueError('assessment inputs and writable output overlap')
    loader_item = config['loader_receipt']
    loader = _manifest(Path(loader_item['path']), loader_item['sha256'])
    leaf_paths, leaf_receipt = policy.approved_leaves(resolved, loader)
    clock_item = config['clock_receipt']
    clock_paths, clock_receipt = clock.prepare_child(Path(clock_item['path']), clock_item['sha256'], resolved)
    policy.validate_leaf_receipt(leaf_receipt, loader)
    clock.validate_receipt(clock_receipt)
    roots = [source, runtime, Path('/usr/lib'), Path('/lib'), Path('/lib64'), output]
    ffmpeg = Path(assets['ffmpeg']['path']).resolve(strict=True) if role == 'source' else None
    if any(path != ffmpeg and any(path == root or path.is_relative_to(root) for root in roots if root != output)
           for path in resolved):
        raise ValueError('assessment input lies inside a broad runtime grant')
    executable: set[Path] = set()
    if role == 'source':
        from experiments.tennis import w8_receiver_native_bootstrap as native
        inventory = Path(assets['native_inventory']['path'])
        if fixed.file_sha(inventory) != 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b':
            raise ValueError('source native inventory differs')
        record = json.loads(inventory.read_text())
        binary = Path(assets['ffmpeg']['path']).resolve(strict=True)
        loader_path = Path(record['elf_interpreter']).resolve(strict=True)
        if str(loader_path) != config.get('native_loader_path'):
            raise ValueError('source native loader differs from retained config')
        mapped = record['mapped_files']
        libraries = {Path(name): item['sha256'] for name, item in mapped.items()
                     if name not in (str(binary), str(loader_path))}
        native.validate_inventory(inventory, fixed.file_sha(inventory), binary, loader_path,
                                  mapped[str(loader_path)]['sha256'], libraries)
        executable = {binary, loader_path}
        roots.extend([loader_path, *libraries])
    from experiments.tennis.w8_strict_receiver_worker import clean_process
    clean_process()
    original = policy.allowed_rights
    def rights(path: Path, output: Path, abi: int) -> int:
        if path == clock.ROOT:
            return policy.READ_DIR
        base = original(path, output, abi)
        return base | policy.EXECUTE if path in executable else base
    policy.allowed_rights = rights
    try:
        abi, mask = policy.restrict([*roots, *resolved, *leaf_paths, *clock_paths], output)
    finally:
        policy.allowed_rights = original
    denied: dict[str, int] = {}
    for path in resolved:
        try:
            fd = os.open(path, os.O_WRONLY)
        except PermissionError as exc:
            denied[str(path)] = int(exc.errno or 0)
        else:
            os.close(fd)
            raise ValueError('assessment input write unexpectedly allowed')
    if any(code not in (1, 13) for code in denied.values()):
        raise ValueError('assessment read-only denial differs')
    return {'landlock_abi': abi, 'handled_mask': mask, 'namespaces': ns,
            'read_only_inputs': denied, 'approved_leaves': leaf_receipt, 'clock_profile': clock_receipt,
            'execution_grants': sorted(str(path) for path in executable), 'output': str(output)}


def policy_probe(config: dict[str, Any], *, cpu_limit: dict[str, Any]) -> dict[str, Any]:
    """Exercise the real role policy on synthetic files, without any candidate claim."""
    if (sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('PS_CLAIMED_GPU_UUID') or len(os.sched_getaffinity(0)) != 1):
        raise ValueError('assessment policy probe requires one CPU and hidden CUDA')
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    work = Path(config['probe_work']).resolve(strict=True)
    output = Path(config['output']).resolve(strict=True)
    if (work == scratch or not work.is_relative_to(scratch) or output.parent != work
            or not output.is_dir() or any(output.iterdir())):
        raise ValueError('fresh task-owned policy probe paths required')
    fixture, forbidden = work / 'fixture', work / 'forbidden'
    with fixture.open('xb') as stream:
        stream.write(b'fixed assessment policy fixture')
    with forbidden.open('xb') as stream:
        stream.write(b'not an assessment input')
    assets = config['probe_assets']
    if set(assets) != {'ffmpeg', 'native_inventory'}:
        raise ValueError('fixed native policy probe inputs required')
    for name, expected in (('ffmpeg', finish.FFMPEG_SHA),
                           ('native_inventory', 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b')):
        if assets[name].get('sha256') != expected or fixed.file_sha(Path(assets[name]['path'])) != expected:
            raise ValueError('native policy probe input differs')
    isolation = isolate(config, 'source', {'fixture': fixture}, assets, output)
    with (output / 'write-probe').open('xb') as stream:
        stream.write(b'permitted')
    try:
        forbidden.read_bytes()
    except PermissionError as exc:
        read_denial = int(exc.errno or 0)
    else:
        raise ValueError('undeclared assessment input unexpectedly readable')
    try:
        subprocess.run(['/usr/bin/true'], check=False, timeout=5, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except PermissionError as exc:
        exec_denial = int(exc.errno or 0)
    else:
        raise ValueError('undeclared assessment executable unexpectedly allowed')
    if read_denial not in (1, 13) or exec_denial not in (1, 13):
        raise ValueError('assessment denial code differs')
    result = subprocess.run([assets['ffmpeg']['path'], '-version'], check=True, timeout=12,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not result.stdout.startswith(b'ffmpeg version '):
        raise ValueError('pinned native version probe differs')
    receipt = {'cpu_limit_telemetry': {'before_imports': list(_CPU_LIMIT_BEFORE_IMPORTS),
               'after_imports': list(_CPU_LIMIT_AFTER_IMPORTS), 'setup': cpu_limit},
               'schema': 1, 'scope': 'assessment policy component only; no candidate or quality claim',
               'gpu_claim': False, 'quality_claim': False, 'isolation': isolation,
               'undeclared_read_denial': read_denial, 'undeclared_execute_denial': exec_denial,
               'output_write': True, 'ffmpeg_version_sha256': fixed.digest(result.stdout)}
    with (output / 'assessment-policy.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
    return receipt


def run_assessment_policy_proof(prepared: dict[str, Any]) -> dict[str, Any]:
    """Run the component probe in its own bounded, owned PID namespace."""
    import time
    from experiments.tennis import w8_clock_profile as clock, w8_witness_namespace as namespace
    from experiments.tennis import w8_witness_profile as witness
    from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('CPU-only assessment policy proof required')
    scratch = Path(prepared['scratch']).resolve(strict=True)
    work = scratch / 'assessment-policy-proof'
    work.mkdir()
    output = work / 'output'
    output.mkdir()
    inventory = Path(prepared['native_inventory'])
    native = json.loads(inventory.read_text())
    binary = Path(native['ffmpeg_path']).resolve(strict=True)
    clock_path = work / 'clock-identity.json'
    clock_sha = clock.write_parent_receipt(clock_path)
    loader = Path(prepared['loader_receipt'])
    config = {'phase': 'policy-probe', 'probe_work': str(work), 'output': str(output),
              'native_loader_path': str(Path(native['elf_interpreter']).resolve(strict=True)),
              'runtime_root': str(Path(prepared['policies']['main']['python']).resolve(strict=True).parent.parent),
              'parent_user_namespace': os.readlink('/proc/self/ns/user'),
              'parent_network_namespace': os.readlink('/proc/self/ns/net'),
              'loader_receipt': {'path': str(loader), 'sha256': fixed.file_sha(loader)},
              'clock_receipt': {'path': str(clock_path), 'sha256': clock_sha},
              'probe_assets': {'ffmpeg': {'path': str(binary), 'sha256': finish.FFMPEG_SHA},
                               'native_inventory': {'path': str(inventory),
                                 'sha256': 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b'}}}
    config_path = work / 'config.json'
    with config_path.open('x') as stream:
        json.dump(config, stream, sort_keys=True)
    config_sha = fixed.file_sha(config_path)
    root = Path(prepared['root']).resolve(strict=True)
    outside = os.readlink('/proc/self/ns/pid')
    receipt = work / 'namespace.json'
    argv = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', '--pid', '--fork', '--kill-child=SIGKILL',
            str(prepared['policies']['main']['python']), '-m', 'experiments.tennis.w8_witness_namespace',
            '--outside', outside, '--timeout', '35', '--receipt', str(receipt), '--', namespace.ASSESSMENT_TARGET,
            '--phase', 'policy-probe', '--config', str(config_path), '--config-sha256', config_sha]
    env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PS_STAGE_DIR': os.environ['PS_STAGE_DIR'],
           'PS_SCRATCH_DIR': str(scratch), 'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work)}
    started = time.monotonic()
    parent = os.getpid()
    with (work / 'stdout').open('xb') as stdout, (work / 'stderr').open('xb') as stderr:
        proc = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                preexec_fn=lambda: parent_death_signal(parent), close_fds=True)
        try:
            status = proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise RuntimeError('assessment policy proof timed out; preserve artifacts')
    if status:
        raise RuntimeError('assessment policy proof failed; preserve artifacts')
    witness.namespace_completion(receipt, expected_pgid=os.getpgrp(), expected_outside=outside,
                                 expected_target=namespace.ASSESSMENT_TARGET)
    proof_path = output / 'assessment-policy.json'
    proof = json.loads(proof_path.read_text())
    if (proof.get('scope') != 'assessment policy component only; no candidate or quality claim'
            or proof.get('gpu_claim') is not False or proof.get('quality_claim') is not False):
        raise ValueError('assessment policy proof scope differs')
    return {'scope': proof['scope'], 'wall_seconds': time.monotonic() - started,
            'receipt_paths': {'policy.json': proof_path, 'config.json': config_path,
                              'namespace.json': receipt,
                              'namespace.started': receipt.with_name(receipt.name + '.started')}}


def execute(config_path: Path, config_sha256: str) -> dict[str, Any]:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux assessment required')
    config = _manifest(config_path, config_sha256)
    role = config.get('phase')
    if role not in ('source', 'score', 'offline'):
        raise ValueError('invalid assessment phase')
    candidate, producer = authenticate(config)
    assets = role_files(config, role)
    if role in ('score', 'offline'):
        prefix = 'source' if role == 'score' else 'score'
        previous = assets[prefix + '_assessment']
        previous_config = json.loads(Path(assets[prefix + '_config']['path']).read_text())
        previous_loader = prior_loader(assets[prefix + '_loader'], previous_config)
        validate_prior_fleet_gate(Path(assets[prefix + '_validation']['path']), phase=prefix,
                                  assessment_sha=previous['sha256'], producer_sha=config['producer_manifest']['sha256'])
        previous_proof = validate_prior_assessment(Path(previous['path']), phase=prefix,
            producer_sha=config['producer_manifest']['sha256'], candidate=candidate,
            output_name='source-rgb.npy' if role == 'score' else 'scores.npz',
            output_sha=assets['source_rgb' if role == 'score' else 'scores']['sha256'],
            prior_config=previous_config, prior_loader=previous_loader)
        if role == 'offline' and (previous_proof['outputs'].get('sealed.json') != assets['score_receipt']['sha256']
                                  or previous_proof.get('quality_claim') is not False):
            raise ValueError('scoring execution identity differs')
    output = Path(config['output'])
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if (not output.is_absolute() or output.is_symlink() or not output.is_dir() or any(output.iterdir())
            or output.resolve(strict=True) != output or output == scratch or not output.is_relative_to(scratch)):
        raise ValueError('fresh assessment output required')
    cpu = config.get('cpu_seconds')
    if type(cpu) is not int or not 1 <= cpu <= 3600:
        raise ValueError('bounded assessment CPU required')
    child = config.get('child_cpu_seconds', cpu)
    if type(child) is not int or not 1 <= child <= 3600:
        raise ValueError('bounded source child CPU required')
    if _CPU_LIMIT_BEFORE_IMPORTS != _CPU_LIMIT_AFTER_IMPORTS:
        raise ValueError('CPU limit changed during assessment imports')
    cpu_limit = fixed.set_cpu_soft_limit(cpu, required_child=child if role == 'source' else cpu)
    os.environ.update({'TMPDIR': str(output), 'TMP': str(output), 'TEMP': str(output),
                       'XDG_CACHE_HOME': str(output / '.cache'), 'TORCH_HOME': str(output / '.cache' / 'torch'),
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    sys.dont_write_bytecode = True
    isolation = isolate(config, role, candidate, assets, output)
    device = None
    if role in ('source', 'score'):
        claim = config.get('claimed_gpu_uuid')
        if not isinstance(claim, str) or claim != os.environ.get('PS_CLAIMED_GPU_UUID'):
            raise ValueError('assessment must use the fleet-claimed GPU')
        device = _gpu(claim)
    if role == 'source':
        from experiments.tennis import w8_receiver_score as scorer
        scorer.prepare_source({'cpu_seconds': cpu, 'child_cpu_seconds': child,
                               'wall_seconds': config['wall_seconds'], 'output': str(output),
                               'files': {**config['candidate_files'],
                                         **{k: v for k, v in assets.items() if k != 'native_inventory'}}})
    elif role == 'score':
        from experiments.tennis import w8_receiver_score as scorer
        scorer.run({'cpu_seconds': cpu, 'output': str(output),
                    'files': {**config['candidate_files'],
                              **{k: v for k, v in assets.items()
                                 if k not in ('source_assessment', 'source_config', 'source_loader', 'source_validation')}}})
    else:
        offline_assets = {key: value for key, value in assets.items()
                          if key not in ('score_assessment', 'score_config', 'score_loader', 'score_validation')}
        offline_config = {'files': {**config['candidate_files'], **offline_assets},
                          'scorer_identity': config['scorer_identity']}
        report = finish.validate_offline(offline_config)
        with (output / 'offline.json').open('x') as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
    result_names = {'source': ('source-rgb.npy', 'sealed.json'),
                    'score': ('scores.npz', 'sealed.json'), 'offline': ('offline.json',)}[role]
    result = {'cpu_limit_telemetry': {'before_imports': list(_CPU_LIMIT_BEFORE_IMPORTS),
              'after_imports': list(_CPU_LIMIT_AFTER_IMPORTS), 'setup': cpu_limit},
              'schema': 1, 'scope': 'post-seal W8 ' + role, 'producer_manifest_sha256': config['producer_manifest']['sha256'],
              'asset_manifest_sha256': config['asset_manifest']['sha256'], 'producer_jobs': producer['jobs'],
              'candidate_sha256': {name: fixed.file_sha(path) for name, path in candidate.items()},
              'role_sha256': {name: row['sha256'] for name, row in assets.items()},
              'outputs': {name: fixed.file_sha(output / name) for name in result_names}, 'device': device,
              'role_isolation': isolation,
              'quality_claim': role == 'offline'}
    with (output / 'assessment.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    return result


def fleet_run(stage: Path, template_path: Path, template_sha: str) -> None:
    """Run exactly one independently promoted post-seal role under PID1."""
    import time
    from experiments.tennis import w8_clock_profile as clock, w8_witness_namespace as namespace
    from experiments.tennis import w8_witness_profile as witness
    from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal

    spec = json.loads((stage.parent / 'spec.json').read_text())
    template = _manifest(template_path, template_sha)
    phase = template.get('phase')
    if (phase not in ('source', 'score', 'offline') or stage.name != 'smoke' or os.environ.get('PS_STAGE') != 'smoke'
            or spec.get('w8_assessment_profile') != 'w8-postseal-' + str(phase) + '-v1'
            or spec.get('staged_inputs', [{}])[0].get('sha256') != template_sha
            or spec.get('w8_assessment_producer_manifest_sha256') != template.get('producer_manifest', {}).get('sha256')
            or spec.get('w8_assessment_asset_manifest_sha256') != template.get('asset_manifest', {}).get('sha256')
            or spec.get('w8_assessment_hard_ceilings') != template.get('hard_ceilings')
            or template.get('output') != '{assessment_output}' or len(os.sched_getaffinity(0)) != 1):
        raise ValueError('assessment fleet dispatch differs')
    producer_paths, producer = authenticate(template)
    del producer_paths
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    check_assessment_promotion(template, producer=producer, host_name=dispatch['environment']['alias'],
        plan=spec['w8_assessment_plan'], promotion_sha=spec['w8_assessment_promotion_sha256'],
        ledger_sha=spec['w8_assessment_ledger_sha256'], previous_sha=spec['w8_assessment_previous_sha256'],
        source=dispatch['code'])
    check_asset_manifest(template, phase)
    caps = template['hard_ceilings']
    if (caps['cpu_seconds'] < spec['budget_seconds'] + caps['external_cpu_seconds'] + caps['post_campaign_cpu_seconds']
            or (phase != 'offline' and caps['gpu_seconds'] < spec['budget_seconds'] + caps['post_campaign_gpu_seconds'])):
        raise ValueError('assessment hard ceilings do not cover declared lifecycle')
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    work = scratch / 'assessment-worker'
    work.mkdir()
    output = work / 'output'
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    loader = Path('/etc/ld.so.cache').resolve(strict=True)
    info = loader.stat()
    if str(loader) != '/etc/ld.so.cache' or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & stat.S_IWOTH:
        raise ValueError('unsafe assessment loader identity')
    loader_path = work / 'loader.json'
    with loader_path.open('x') as stream:
        json.dump({'device': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode, 'uid': info.st_uid}, stream)
    clock_path = work / 'clock.json'
    clock_sha = clock.write_parent_receipt(clock_path)
    config = dict(template)
    config.update(output=str(output), parent_user_namespace=os.readlink('/proc/self/ns/user'),
                  parent_network_namespace=os.readlink('/proc/self/ns/net'),
                  loader_receipt={'path': str(loader_path), 'sha256': fixed.file_sha(loader_path)},
                  clock_receipt={'path': str(clock_path), 'sha256': clock_sha})
    if phase in ('source', 'score'):
        claim = os.environ.get('PS_CLAIMED_GPU_UUID', '')
        if not claim or claim != os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('assessment fleet GPU claim missing')
        config['claimed_gpu_uuid'] = claim
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_CLAIMED_GPU_UUID'):
        raise ValueError('offline assessment must hide CUDA')
    config_path = work / 'config.json'
    with config_path.open('x') as stream:
        json.dump(config, stream, sort_keys=True)
    outside = os.readlink('/proc/self/ns/pid')
    namespace_path = work / 'namespace.json'
    wall = spec.get('w8_assessment_worker_wall')
    if type(wall) is not int or wall <= 15 or wall >= spec['smoke']['seconds']:
        raise ValueError('assessment worker needs bounded cleanup time')
    clock_record = dispatch['campaign_clock']
    now = time.monotonic()
    cleanup = spec['w8_assessment_plan']['cleanup_seconds']
    if (now + wall + cleanup > clock_record['phase_started'] + spec['smoke']['seconds']
            or now + wall + cleanup + spec['validator_seconds'] + 5 > clock_record['budget_end']):
        raise ValueError('remaining assessment budget cannot cover worker and cleanup')
    argv = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', '--pid', '--fork', '--kill-child=SIGKILL',
            sys.executable, '-m', 'experiments.tennis.w8_witness_namespace', '--outside', outside,
            '--timeout', str(wall - 10), '--receipt', str(namespace_path), '--', namespace.ASSESSMENT_TARGET,
            '--phase', phase, '--config', str(config_path), '--config-sha256', fixed.file_sha(config_path)]
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    parent = os.getpid()
    started = time.monotonic()
    with (work / 'stdout').open('xb') as stdout, (work / 'stderr').open('xb') as stderr:
        proc = subprocess.Popen(argv, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr, close_fds=True,
                                preexec_fn=lambda: parent_death_signal(parent))
        timed_out = False
        try:
            code = proc.wait(timeout=wall - 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait(timeout=5)
            timed_out = True
    if code or timed_out:
        deadline = time.monotonic() + 3
        detail = ''
        absent = False
        while time.monotonic() < deadline:
            try:
                witness.namespace_started_absence(namespace_path, expected_pgid=os.getpgrp(),
                    expected_outside=outside, expected_target=namespace.ASSESSMENT_TARGET)
                absent = True
                break
            except (OSError, ValueError, KeyError, IndexError) as exc:
                detail = str(exc)
                time.sleep(.02)
        with (work / 'failure-teardown.json').open('x') as stream:
            json.dump({'worker_status': code, 'timed_out': timed_out, 'owned_init_and_worker_absent': absent,
                       'detail': detail, 'scope': 'namespace kernel teardown; descendants not individually observed'}, stream)
        if not absent:
            raise RuntimeError('assessment teardown unresolved; preserve task-owned outputs')
        raise RuntimeError('assessment failed or timed out; preserved task-owned outputs')
    witness.namespace_completion(namespace_path, expected_pgid=os.getpgrp(), expected_outside=outside,
                                 expected_target=namespace.ASSESSMENT_TARGET)
    retained = stage / 'w8-assessment-retained'
    retained.mkdir()
    names = {'source': ('source-rgb.npy', 'source.yuv', 'sealed.json', 'assessment.json',
                         'source-handle.json', 'source-execution.json', 'source.stdout', 'source.stderr'),
             'score': ('scores.npz', 'sealed.json', 'assessment.json'),
             'offline': ('offline.json', 'assessment.json')}[phase]
    for name in names:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('assessment output member missing')
        with path.open('rb') as src, (retained / name).open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    for name, source_path in (('namespace.json', namespace_path),
                              ('namespace.json.started', namespace_path.with_name(namespace_path.name + '.started')),
                              ('config.json', config_path), ('loader.json', loader_path)):
        shutil.copyfile(source_path, retained / name)
    publication = scratch / 'publish'
    publication.mkdir()
    summary = {'scope': 'post-seal assessment phase only', 'phase': phase, 'automatic_promotion': False,
               'template_sha256': template_sha, 'retained': str(retained),
               'members': {path.name: fixed.file_sha(path) for path in retained.iterdir()},
               'outside_namespace': outside, 'campaign_pgid': os.getpgrp(),
               'elapsed_seconds': time.monotonic() - started}
    with (publication / 'summary.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)


def check_clean_execution(execution: dict[str, Any], dispatch: dict[str, Any],
                          phase: str, record: dict[str, Any]) -> None:
    from experiments.jobs.inbox import contention_policy
    from experiments.tennis.w8_cuda_system_leaves import canonical_gpu_uuid

    if (phase not in ('source', 'score', 'offline') or execution.get('exit_code') != 0
            or execution.get('timing_contaminated') is not False or execution.get('contention') != []
            or execution.get('paused_seconds') != 0
            or contention_policy(dispatch.get('contention_policy')) != {'policy': 'stop', 'resume_attempts': 0}):
        raise ValueError('assessment execution was failed or contaminated')
    gpu = (dispatch.get('environment') or {}).get('gpu')
    if phase == 'offline':
        if gpu is not None or record.get('device') is not None or execution.get('gpu_sharing') is not None:
            raise ValueError('offline assessment unexpectedly claimed a GPU')
    else:
        device = record.get('device') or {}
        if (not isinstance(gpu, dict) or execution.get('gpu_sharing') != {'mode': 'exclusive'}
                or dispatch.get('gpu_sharing') != {'mode': 'exclusive'}
                or canonical_gpu_uuid(device.get('uuid', '')) != canonical_gpu_uuid(gpu.get('uuid', ''))
                or device.get('name') != gpu.get('name') or 'RTX 6000 Ada' not in str(gpu.get('name', ''))
                or (device.get('major'), device.get('minor')) != (8, 9)):
            raise ValueError('assessment GPU claim or exclusive execution differs')


def fleet_validate(stage: Path, destination: Path) -> None:
    """Reopen retained outputs after scratch release; never promote another role."""
    import io
    import tarfile
    import numpy as np
    from experiments.tennis import w8_receiver_score as scorer, w8_witness_namespace as namespace
    from experiments.tennis import w8_witness_profile as witness

    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    publication = stage / 'published.tar'
    if fixed.file_sha(publication) != staging['published']['sha256']:
        raise ValueError('assessment publication differs')
    with tarfile.open(publication) as archive:
        files = [entry for entry in archive if entry.isfile()]
        if len(files) != 1 or files[0].name != 'publish/summary.json':
            raise ValueError('assessment publication membership differs')
        stream = archive.extractfile(files[0])
        if stream is None:
            raise ValueError('assessment summary missing')
        summary = json.load(stream)
    phase = summary['phase']
    retained = stage / 'w8-assessment-retained'
    if (phase not in ('source', 'score', 'offline') or summary.get('scope') != 'post-seal assessment phase only'
            or summary.get('automatic_promotion') is not False or summary.get('retained') != str(retained)
            or spec.get('w8_assessment_profile') != 'w8-postseal-' + phase + '-v1'
            or summary.get('template_sha256') != spec['staged_inputs'][0]['sha256']
            or retained.is_symlink() or not retained.is_dir()):
        raise ValueError('assessment retained identity differs')
    members = summary.get('members')
    if (not isinstance(members, dict) or {path.name for path in retained.iterdir()} != set(members)
            or any(not _sha(sha) or fixed.file_sha(retained / name) != sha for name, sha in members.items())):
        raise ValueError('assessment retained bytes differ')
    group = json.loads((stage.parent / 'run/status.json').read_text())['process_group_id']
    outside = os.readlink('/proc/self/ns/pid')
    if summary['outside_namespace'] != outside or summary['campaign_pgid'] != group:
        raise ValueError('assessment campaign namespace or group differs')
    # The helper requires sibling `.started`; retained name is intentionally exact.
    witness.namespace_completion(retained / 'namespace.json', expected_pgid=group,
                                 expected_outside=outside, expected_target=namespace.ASSESSMENT_TARGET)
    config = json.loads((retained / 'config.json').read_text())
    candidate, _ = authenticate(config)
    assets = role_files(config, phase)
    record = json.loads((retained / 'assessment.json').read_text())
    check_clean_execution(json.loads((stage / 'execution.json').read_text()), dispatch, phase, record)
    if (record.get('producer_manifest_sha256') != config['producer_manifest']['sha256']
            or record.get('asset_manifest_sha256') != config['asset_manifest']['sha256']
            or record.get('candidate_sha256') != {key: fixed.file_sha(path) for key, path in candidate.items()}
            or record.get('role_sha256') != {key: row['sha256'] for key, row in assets.items()}
            or record.get('role_isolation', {}).get('landlock_abi', 0) < 3
            or not record['role_isolation'].get('read_only_inputs')
            or any(value not in (1, 13) for value in record['role_isolation']['read_only_inputs'].values())):
        raise ValueError('assessment producer or role hashes differ')
    loader_record = prior_loader({'path': str(retained / 'loader.json'),
                                  'sha256': fixed.file_sha(retained / 'loader.json')}, config)
    validate_role_isolation(record, config, phase, loader_record)
    if phase in ('source', 'score'):
        name = 'source-rgb.npy' if phase == 'source' else 'scores.npz'
        validate_prior_assessment(retained / 'assessment.json', phase=phase,
            producer_sha=config['producer_manifest']['sha256'], candidate=candidate,
            output_name=name, output_sha=fixed.file_sha(retained / name), prior_config=config, prior_loader=loader_record)
        if record['outputs'].get('sealed.json') != fixed.file_sha(retained / 'sealed.json'):
            raise ValueError('assessment original execution seal differs')
        gpu = dispatch['environment']['gpu']
        from experiments.tennis.w8_cuda_system_leaves import canonical_gpu_uuid
        if canonical_gpu_uuid(record['device']['uuid']) != canonical_gpu_uuid(gpu['uuid']):
            raise ValueError('assessment device differs from dispatch')
        if phase == 'source':
            source = np.load(retained / name, mmap_mode='r', allow_pickle=False)
            hashes = scorer._source_hashes(Path(assets['receiver']['path']))
            if (source.shape != (80, 1080, 1920, 3) or source.dtype != np.uint8
                    or any(finish._array_sha(source[i]) != hashes[str(t)] for i, t in enumerate(finish.TICKS))
                    or fixed.file_sha(retained / 'source.yuv') !=
                    '5a26ceb4c08fdfac2726805bb37c5cefe29ff18b9a3c733690fd4e263d3d972d'):
                raise ValueError('source RGB or native decode identity differs')
        else:
            validate_prior_fleet_gate(Path(assets['source_validation']['path']), phase='source',
                                      assessment_sha=assets['source_assessment']['sha256'],
                                      producer_sha=config['producer_manifest']['sha256'])
            source_config = json.loads(Path(assets['source_config']['path']).read_text())
            validate_prior_assessment(Path(assets['source_assessment']['path']), phase='source',
                producer_sha=config['producer_manifest']['sha256'], candidate=candidate,
                output_name='source-rgb.npy', output_sha=assets['source_rgb']['sha256'],
                prior_config=source_config, prior_loader=prior_loader(assets['source_loader'], source_config))
            score_seal = json.loads((retained / 'sealed.json').read_text())
            if (score_seal.get('passed') is not True or score_seal.get('scores_sha256') != fixed.file_sha(retained / name)
                    or score_seal.get('composite_sha256') != fixed.file_sha(candidate['composite'])):
                raise ValueError('scorer output seal differs')
            with np.load(retained / 'scores.npz', allow_pickle=False) as scores:
                saved = scorer._member(Path(assets['judge_archive']['path']),
                    'publish/nadal_kyrgios__000@120/scores.npz', 2_000_000)
                with np.load(io.BytesIO(saved), allow_pickle=False) as original:
                    if not np.array_equal(scores['ticks'], finish.TICKS) or not np.array_equal(original['ticks'], finish.TICKS):
                        raise ValueError('scoring ticks differ')
                    finish._compare_scores({key: scores[key] for key in finish.METRICS},
                                           {key: original[key][3] for key in finish.METRICS})
    else:
        validate_prior_fleet_gate(Path(assets['score_validation']['path']), phase='score',
                                  assessment_sha=assets['score_assessment']['sha256'],
                                  producer_sha=config['producer_manifest']['sha256'])
        validate_prior_assessment(Path(assets['score_assessment']['path']), phase='score',
            producer_sha=config['producer_manifest']['sha256'], candidate=candidate,
            output_name='scores.npz', output_sha=assets['scores']['sha256'],
            prior_config=json.loads(Path(assets['score_config']['path']).read_text()),
            prior_loader=prior_loader(assets['score_loader'],
                json.loads(Path(assets['score_config']['path']).read_text())))
        report = finish.validate_offline({'files': {**config['candidate_files'],
            **{key: value for key, value in assets.items()
               if key not in ('score_assessment', 'score_config', 'score_loader', 'score_validation')}},
            'scorer_identity': config['scorer_identity']})
        if report != json.loads((retained / 'offline.json').read_text()):
            raise ValueError('offline comparison differs from retained result')
    with destination.open('x') as validation_output:
        json.dump({'passed': True, 'scope': 'post-seal ' + phase + ' only',
                   'automatic_promotion': False, 'checks': {'independent_reopen': True},
                   'assessment_sha256': fixed.file_sha(retained / 'assessment.json'),
                   'producer_manifest_sha256': config['producer_manifest']['sha256'],
                   'job_id': stage.parent.name}, validation_output, sort_keys=True)


def main() -> None:
    if sys.argv[1:2] == ['make-spec']:
        parser = argparse.ArgumentParser()
        parser.add_argument('action', choices=('make-spec',))
        parser.add_argument('--template-local', type=Path, required=True)
        parser.add_argument('--template-remote', required=True)
        parser.add_argument('--deadline', required=True)
        parser.add_argument('--host', required=True)
        parser.add_argument('--approved-producer-sha256', required=True)
        parser.add_argument('--approved-asset-manifest-sha256', required=True)
        parser.add_argument('--approved-promotion-sha256', required=True)
        parser.add_argument('--approved-ledger-sha256', required=True)
        parser.add_argument('--approved-previous-sha256')
        parser.add_argument('--plan', type=Path, required=True)
        parser.add_argument('--command-wall', type=int, required=True)
        parser.add_argument('--validator-wall', type=int, required=True)
        parser.add_argument('--campaign-wall', type=int, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        spec = make_fleet_spec(template_local=args.template_local, template_remote=args.template_remote,
            deadline=args.deadline, host_name=args.host, approved_producer_sha256=args.approved_producer_sha256,
            approved_asset_manifest_sha256=args.approved_asset_manifest_sha256,
            approved_promotion_sha256=args.approved_promotion_sha256, approved_ledger_sha256=args.approved_ledger_sha256,
            approved_previous_sha256=args.approved_previous_sha256, plan=json.loads(args.plan.read_text()),
            command_wall=args.command_wall, validator_wall=args.validator_wall, campaign_wall=args.campaign_wall)
        with args.output.open('x') as output:
            json.dump(spec, output, sort_keys=True, indent=2, allow_nan=False)
            output.write('\n')
        return
    if sys.argv[1:2] == ['fleet-run']:
        parser = argparse.ArgumentParser()
        parser.add_argument('action', choices=('fleet-run',))
        parser.add_argument('--template', type=Path, required=True)
        parser.add_argument('--template-sha256', required=True)
        args = parser.parse_args()
        fleet_run(Path(os.environ['PS_STAGE_DIR']), args.template, args.template_sha256)
        return
    if sys.argv[1:2] == ['fleet-validate']:
        fleet_validate(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['PS_VALIDATION_PATH']))
        return
    if sys.argv[1:2] == ['fleet-noop']:
        if os.environ.get('PS_STAGE') != 'full':
            raise ValueError('assessment completion marker requires full stage')
        publish = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        publish.mkdir()
        with (publish / 'completion.json').open('x') as stream:
            json.dump({'scope': 'assessment completion marker; no promotion'}, stream)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('source', 'score', 'offline', 'policy-probe'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    config = _manifest(args.config, args.config_sha256)
    if config.get('phase') != args.phase:
        raise ValueError('assessment CLI phase differs from sealed config')
    if args.phase == 'policy-probe':
        if _CPU_LIMIT_BEFORE_IMPORTS != _CPU_LIMIT_AFTER_IMPORTS:
            raise ValueError('CPU limit changed during assessment imports')
        cpu_limit = fixed.set_cpu_soft_limit(30, required_child=30)
        policy_probe(config, cpu_limit=cpu_limit)
    else:
        execute(args.config, args.config_sha256)


if __name__ == '__main__':
    main()
