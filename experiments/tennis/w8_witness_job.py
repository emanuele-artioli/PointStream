"""Proposed W8 context witness control. No compute release is implied.

The CPU gate is a root-pinned audit envelope, not an operator boolean. Its raw
proof hashes and exact source/runtime identities must be independently audited
before root supplies the envelope SHA to a prospective fleet spec.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.tennis import w8_strict_host_cost_job as host
from experiments.tennis import w8_input_job as fixed

CPU_SCOPE = 'root-reviewed W8 same-source CPU isolation and native lifecycle proof'
REQUIRED_PROOFS = ('namespace_lifecycle', 'native_normal', 'native_timeout_kill', 'strict_modes', 'independent_audit')
REQUIRED_CHECKS = ('same_source_snapshot_environment', 'cpu_hidden_one_core', 'namespace_normal_timeout_term_kill',
                   'native_real_yuv_and_timeout_kill', 'owned_pid_starttick_teardown', 'strict_mode_deny_paths',
                   'strict_source_closure_and_clock_leaves', 'counter_and_external_cost_coverage')


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def check_cpu_gate(gate: dict[str, Any], *, gate_sha: str, expected_sha: str,
                   source: dict[str, Any], environment_sha: str, host_name: str | None) -> None:
    """Check the exact root-pinned envelope after primary review of raw proofs."""
    if (not _sha(gate_sha) or gate_sha != expected_sha or gate.get('schema') != 1
            or gate.get('scope') != CPU_SCOPE or gate.get('passed') is not True
            or not _sha(source.get('source_sha256')) or not _sha(source.get('snapshot_sha256'))
            or not re.fullmatch(r'[0-9a-f]{40}', str(source.get('git_head')))
            or not _sha(environment_sha) or host_name not in ('gpu5', 'gpu6')
            or gate.get('source_sha256') != source.get('source_sha256')
            or gate.get('snapshot_sha256') != source.get('snapshot_sha256')
            or gate.get('revision') != source.get('git_head')
            or gate.get('environment_sha256') != environment_sha or gate.get('host') != host_name
            or not re.fullmatch(r'\d{8}T\d{6}Z-[0-9a-f]{8}', str(gate.get('job_id')))
            or not _sha(gate.get('spec_sha256'))):
        raise ValueError('same-source CPU proof envelope identity differs')
    raw, checks = gate.get('raw_receipt_sha256'), gate.get('checks')
    if (not isinstance(raw, dict) or set(raw) != set(REQUIRED_PROOFS)
            or any(not _sha(raw[name]) for name in REQUIRED_PROOFS)
            or not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS)
            or any(checks[name] is not True for name in REQUIRED_CHECKS)):
        raise ValueError('independent lifecycle, native and strict source proof references required')


def load_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Bind staged proof and every frozen receiver input to the fleet dispatch."""
    paths, spec = host.stage_inputs(stage, witness=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    proof = json.loads(paths['cpu_proof'].read_text())
    code = dispatch.get('code', {})
    environment = dispatch.get('staged_environment', {})
    host_name = (dispatch.get('environment') or {}).get('alias')
    check_cpu_gate(proof, gate_sha=fixed.file_sha(paths['cpu_proof']),
                   expected_sha=spec['witness_cpu_proof_sha256'], source=code,
                   environment_sha=environment.get('sha256'), host_name=host_name)
    return paths, spec, dispatch


def load_released_inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Use immutable staged-source records after fleet scratch has been released."""
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    declared = {x['name']: x for x in spec['staged_inputs']}
    staged = {x['name']: x for x in staging['inputs']}
    expected = {'packet': fixed.WINDOW_SHA, 'profile': fixed.PROFILE_SHA, 'oracle': host.ORACLE_SHA,
                'base': host.BASE_ARCHIVE_SHA, 'aa': host.receiver.AA_ARCHIVE_SHA,
                **{key: host.receiver.PINNED_WEIGHTS[name] for key, name in host.WEIGHT_NAMES.items()},
                'cpu_proof': spec['witness_cpu_proof_sha256']}
    if (len(declared) != len(spec['staged_inputs']) or len(staged) != len(staging['inputs'])
            or set(declared) != set(expected) | {'preflight', 'identity'} or set(staged) != set(declared)):
        raise ValueError('released staged asset membership differs')
    for name in ('preflight', 'identity'):
        expected[name] = declared[name]['sha256']
    for name, digest in expected.items():
        if (declared[name].get('sha256') != digest or staged[name].get('sha256') != digest
                or declared[name].get('extract', False) or staged[name].get('extract', False)
                or staged[name].get('source') != declared[name].get('path')):
            raise ValueError('released staged asset identity differs')
    paths = {name: Path(staged[name]['source']) for name in ('preflight', 'identity', 'oracle', 'cpu_proof')}
    if any(fixed.file_sha(path) != expected[name] for name, path in paths.items()):
        raise ValueError('released proof or preflight source differs')
    proof = json.loads(paths['cpu_proof'].read_text())
    check_cpu_gate(proof, gate_sha=fixed.file_sha(paths['cpu_proof']),
                   expected_sha=spec['witness_cpu_proof_sha256'], source=dispatch.get('code', {}),
                   environment_sha=dispatch.get('staged_environment', {}).get('sha256'),
                   host_name=(dispatch.get('environment') or {}).get('alias'))
    return paths, spec, dispatch


def _pilot_spec(spec: dict[str, Any]) -> None:
    """Reject every profile other than the separately reviewed unit-0 pilot."""
    if (spec.get('w8_witness_profile') != 'w8-context-pilot-v1'
            or spec.get('entrypoint') != ['-m', 'experiments.tennis.w8_witness_job']
            or spec.get('arguments') != ['{phase}']
            or spec.get('scale') != {'phase': {'smoke': 'pilot', 'full': 'noop'}}
            or spec.get('validator') != ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'validate']
            or spec.get('device') != 'gpu' or spec.get('gpu_models') != ['RTX 6000 Ada']
            or spec.get('gpu_sharing') != {'mode': 'exclusive'}
            or spec.get('single_cpu_campaign') is not True or spec.get('cpu_threads') != 1
            or spec.get('contention', {}).get('policy') != 'stop'
            or spec.get('contention', {}).get('resume_attempts') != 0
            or not 0 < spec.get('smoke', {}).get('seconds', 0) <= 600
            or spec.get('full', {}).get('seconds') != 5
            or not 0 < spec.get('budget_seconds', 0) <= 900
            or not 0 < spec.get('validator_seconds', 0) <= 120
            or not 0 < spec.get('witness_validator_cpu_seconds', 0) <= 120
            or not 0 < spec.get('witness_cpu_ceiling_seconds', 0) <= 1000):
        raise ValueError('unit-0 witness pilot spec differs')
    caps = spec.get('witness_caps')
    names = {'parent_cpu', 'packet_cpu', 'generator_cpu', 'generator_wall', 'external_cpu'}
    if (not isinstance(caps, dict) or set(caps) != names
            or any(type(caps[x]) is not int or caps[x] <= 0 for x in names)
            or caps['generator_wall'] > 500 or caps['packet_cpu'] > 100 or caps['parent_cpu'] > 60
            or caps['generator_cpu'] > 500 or caps['external_cpu'] > 150
            or caps['parent_cpu'] + caps['packet_cpu'] + caps['generator_cpu']
               + spec['witness_validator_cpu_seconds'] + caps['external_cpu'] > spec['witness_cpu_ceiling_seconds']):
        raise ValueError('pilot CPU or wall caps differ')


def run_pilot(stage: Path) -> None:
    """One unit's 30-step context only; no second unit is admitted here."""
    import os
    import resource
    import sys
    import time
    from experiments.tennis import w8_witness_profile as witness

    if stage.name != 'smoke' or os.environ.get('PS_STAGE') != 'smoke':
        raise ValueError('pilot is admitted only as the smoke stage')
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-core Linux campaign required')
    started = time.monotonic()
    paths, spec, dispatch = load_inputs(stage)
    _pilot_spec(spec)
    caps = spec['witness_caps']
    resource.setrlimit(resource.RLIMIT_CPU, (caps['parent_cpu'], caps['parent_cpu']))
    if (spec.get('clock_policy_profile') != host.w8_clock_profile.PROFILE
            or spec.get('clock_reference_audit_sha256') != host.CLOCK_AUDIT_SHA
            or spec.get('materialization_profile') != host.folder_profile.PROFILE
            or spec.get('folder_reference_audit_sha256') != host.FOLDER_AUDIT_SHA):
        raise ValueError('qualification requires frozen clock and materialization proofs')
    host.require_gpu_uuid_api(__import__('torch'))
    digests = host.StableDigests()
    prepared = host.prepare_receiver_inputs(paths, spec, packet_cpu=caps['packet_cpu'],
                                             materialize_profile=True, clock_profile=True,
                                             digests=digests, witness=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    result = witness.execute_generation_units(prepared, dispatch, visible=visible,
        worker_cpu=caps['generator_cpu'], worker_wall=caps['generator_wall'],
        mode='context_qualification', units=(fixed.UNITS[0],))
    unit = fixed.UNITS[0]
    work, output = prepared['model_work'], result['outputs'][unit]
    retained = host.retained_path(stage)
    retained.mkdir()
    named = {'source-manifest.json': None, 'loader-identity.json': prepared['loader_receipt'],
             'protected-identity.json': prepared['protected_path'],
             'packet-config.json': prepared['decoder_work'] / 'worker-config.json',
             'generator-config.json': work / 'worker-config.json',
             'packet-namespace.json': prepared['decoder_work'] / 'namespace-receipt.json',
             'generator-namespace.json': work / 'namespace-receipt.json',
             'packet-namespace.json.started': prepared['decoder_work'] / 'namespace-receipt.json.started',
             'generator-namespace.json.started': work / 'namespace-receipt.json.started',
             'decoded-manifest.json': prepared['decoder_manifest'], 'decoded.pt': prepared['decoder_out'] / 'decoded.pt',
             'packet-unit.npz': prepared['decoder_out'] / 'publish' / 'units' / f'{unit}.npz',
             'packet-unit-other.npz': prepared['decoder_out'] / 'publish' / 'units' / f'{fixed.UNITS[1]}.npz',
             'packet-cpu-only.json': prepared['decoder_out'] / 'cpu-only.json',
             'gpu-identity.json': output / 'gpu-identity.json', 'materialization.json': output / 'materialization.json',
             'sealed.json': output / 'sealed.json', f'{unit}@match5.npz': output / f'{unit}@match5.npz',
             f'{unit}@match5.json': output / f'{unit}@match5.json'}
    with (retained / 'source-manifest.json').open('x') as stream:
        json.dump(prepared['code_manifest'], stream, sort_keys=True)
    for name, path in named.items():
        if path is not None:
            host.copy_new(path, retained / name)
    import tarfile
    archive = retained / 'witness-context-artifacts.tar'
    if sum((retained / name).stat().st_size for name in named) > 512 * 1024 * 1024:
        raise ValueError('pilot retained data exceeds fixed bound')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w|') as tar:
            for name in sorted(named):
                tar.add(retained / name, arcname=name, recursive=False)
        raw.flush()
        os.fsync(raw.fileno())
    summary = {'scope': 'context_qualification_only', 'unit': unit, 'quality_claim': False,
               'automatic_promotion': False, 'expected': result['expected'][unit],
               'validation': result['validation'][unit], 'generation_usage': result['usage'][unit],
               'packet_usage': prepared['packet_usage'], 'cpu_gate_sha256': fixed.file_sha(paths['cpu_proof']),
               'source_manifest_sha256': fixed.file_sha(retained / 'source-manifest.json'),
               'retained': str(retained), 'archive_sha256': fixed.file_sha(archive),
               'archive_bytes': archive.stat().st_size,
               'members': {name: fixed.file_sha(retained / name) for name in named},
               'elapsed_seconds': time.monotonic() - started,
               'note': '24-frame context; no full-run prefix, quality or second-unit claim'}
    if set(summary['members']) != set(named) or any(not _sha(x) for x in summary['members'].values()):
        raise ValueError('pilot retained artifacts incomplete')
    publish = prepared['scratch'] / 'publish'
    publish.mkdir()
    with (publish / 'summary.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def validate_pilot(stage: Path, target: Path) -> None:
    """Independently reopen the sealed pilot and recompute its identity checks."""
    import os
    import resource
    import tarfile
    from experiments.tennis import w8_receiver_generate as generation
    from experiments.tennis import w8_witness_profile as witness

    try:
        paths, spec, dispatch = load_released_inputs(stage)
        _pilot_spec(spec)
        resource.setrlimit(resource.RLIMIT_CPU, (spec['witness_validator_cpu_seconds'],
                                                 spec['witness_validator_cpu_seconds']))
        publication = stage / 'published.tar'
        staging = json.loads((stage / 'staging.json').read_text())
        if fixed.file_sha(publication) != staging['published']['sha256']:
            raise ValueError('published pilot archive SHA differs')
        with tarfile.open(publication) as tar:
            files = [entry for entry in tar.getmembers() if entry.isfile()]
            if len(files) != 1 or files[0].name != 'publish/summary.json' or files[0].size > 2_000_000:
                raise ValueError('unexpected pilot publication')
            stream = tar.extractfile(files[0])
            if stream is None:
                raise ValueError('pilot summary missing')
            summary = json.load(stream)
        unit = fixed.UNITS[0]
        if (summary.get('scope') != 'context_qualification_only' or summary.get('unit') != unit
                or summary.get('quality_claim') is not False or summary.get('automatic_promotion') is not False
                or summary.get('cpu_gate_sha256') != spec['witness_cpu_proof_sha256']):
            raise ValueError('pilot scope or CPU proof differs')
        import math
        caps = spec['witness_caps']
        for key, cpu_cap, wall_cap in (('packet_usage', caps['packet_cpu'], 130),
                                       ('generation_usage', caps['generator_cpu'], caps['generator_wall'])):
            usage = summary.get(key, {})
            if (usage.get('status') != 0 or usage.get('timed_out') is not False
                    or usage.get('canaries_preserved') is not True
                    or any(isinstance(usage.get(name), bool) or not isinstance(usage.get(name), (int, float))
                           or not math.isfinite(usage[name]) or usage[name] < 0 for name in ('wall_seconds', 'cpu_seconds'))
                    or usage['wall_seconds'] > wall_cap or usage['cpu_seconds'] > cpu_cap):
                raise ValueError('pilot worker bound or canary differs')
        retained = host.retained_path(stage)
        if summary.get('retained') != str(retained) or retained.is_symlink():
            raise ValueError('pilot retained path differs')
        members = summary.get('members')
        required = {'source-manifest.json', 'loader-identity.json', 'protected-identity.json',
                    'packet-config.json', 'generator-config.json', 'packet-namespace.json', 'generator-namespace.json',
                    'packet-namespace.json.started', 'generator-namespace.json.started',
                    'decoded-manifest.json', 'decoded.pt', 'packet-unit.npz', 'packet-unit-other.npz',
                    'packet-cpu-only.json', 'gpu-identity.json', 'materialization.json', 'sealed.json',
                    f'{unit}@match5.npz', f'{unit}@match5.json'}
        if (not isinstance(members, dict) or set(members) != required
                or any(not _sha(v) or fixed.file_sha(retained / name) != v for name, v in members.items())):
            raise ValueError('pilot retained member identities differ')
        archive = retained / 'witness-context-artifacts.tar'
        if (archive.is_symlink() or archive.stat().st_size != summary['archive_bytes']
                or fixed.file_sha(archive) != summary['archive_sha256']):
            raise ValueError('pilot durable archive differs')
        with tarfile.open(archive) as tar:
            entries = tar.getmembers()
            if (len(entries) != len(required) or {entry.name for entry in entries} != required
                    or any(not entry.isfile() for entry in entries)):
                raise ValueError('pilot durable archive membership differs')
            import hashlib
            for entry in entries:
                member = tar.extractfile(entry)
                if member is None:
                    raise ValueError('pilot archive member unavailable')
                digest = hashlib.sha256()
                while chunk := member.read(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != members[entry.name]:
                    raise ValueError('pilot durable archive member differs')
        manifest = json.loads((retained / 'source-manifest.json').read_text())
        if fixed.file_sha(retained / 'source-manifest.json') != summary['source_manifest_sha256']:
            raise ValueError('pilot source manifest differs')
        config = json.loads((retained / 'generator-config.json').read_text())
        source_root = Path(config['allow_paths'][0]).resolve()
        scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve()
        if source_root != scratch / 'receiver-source':
            raise ValueError('pilot source staging path differs')
        files = manifest.get('files')
        if (not isinstance(files, dict) or not files
                or any(not name.endswith('.py') or not name.startswith(('experiments/', 'src/', 'tools/')) for name in files)):
            raise ValueError('pilot source manifest membership differs')
        import hashlib
        if hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != manifest.get('manifest_sha256'):
            raise ValueError('pilot source manifest digest differs')
        source_archive = stage.parent / 'source.tar'
        if fixed.file_sha(source_archive) != dispatch['code']['snapshot_sha256']:
            raise ValueError('pilot fleet source snapshot differs')
        with tarfile.open(source_archive) as tar:
            for name, digest in files.items():
                member = tar.getmember(name)
                stream = tar.extractfile(member) if member.isfile() else None
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError('pilot source member differs')
        host.check_preflight_bundle(json.loads(paths['preflight'].read_text()), paths['identity'])
        fingerprints = host.read_oracle(paths['oracle'])
        decoded = json.loads((retained / 'decoded-manifest.json').read_text())
        packet_config = json.loads((retained / 'packet-config.json').read_text())
        preflight = json.loads(paths['preflight'].read_text())
        loader_path = retained / 'loader-identity.json'
        host.packet_audit.check_grants(packet_config, preflight['runtimes']['main'])
        host.packet_audit.check_isolation(packet_config, decoded['isolation'], json.loads(loader_path.read_text()),
                                          manifest, fixed.file_sha(loader_path))
        host.packet_audit.reconcile_protected_identity(json.loads(paths['identity'].read_text()),
            json.loads((retained / 'protected-identity.json').read_text()))
        cpu_only = json.loads((retained / 'packet-cpu-only.json').read_text())
        if (cpu_only.get('cuda_visible_devices') != '' or cpu_only.get('cuda_initialized_after_decode') is not False
                or cpu_only.get('torch_version') != preflight['receipts']['main']['torch']
                or cpu_only.get('pre_serialization', {}).get('state_tensors') != 480
                or cpu_only.get('pre_serialization', {}).get('restored_tensors') != 480):
            raise ValueError('packet CPU-only qualification differs')
        if (decoded.get('packet_sha256') != fixed.WINDOW_SHA or decoded.get('profile_sha256') != fixed.PROFILE_SHA
                or decoded.get('match_tensors') != fingerprints or decoded.get('unit_sha256') != generation.ORACLE_UNIT_SHA
                or fixed.file_sha(retained / 'packet-unit.npz') != generation.ORACLE_UNIT_SHA[unit]
                or fixed.file_sha(retained / 'packet-unit-other.npz') != generation.ORACLE_UNIT_SHA[fixed.UNITS[1]]
                or generation.verify_serialized_match(retained / 'decoded.pt', fingerprints) != decoded['checkpoint_sha256']):
            raise ValueError('pilot packet and decoded checkpoint join differs')
        expected = {'unit': unit, 'packet_sha256': fixed.WINDOW_SHA, 'profile_sha256': fixed.PROFILE_SHA,
                    'tensor_fingerprints': fingerprints, 'files': {k: v['sha256'] for k, v in config['files'].items()}}
        if (expected != summary['expected'] or expected['files']['checkpoint'] != decoded['checkpoint_sha256']
                or expected['files']['decoder_manifest'] != fixed.file_sha(retained / 'decoded-manifest.json')):
            raise ValueError('pilot worker input identities differ')
        validation = generation.validate_context_qualification(retained, expected)
        if validation != summary['validation']:
            raise ValueError('pilot context validation differs')
        seal = json.loads((retained / 'sealed.json').read_text())
        witness.check_worker_isolation(config, seal['isolation'],
            runtime=preflight['runtimes']['jcv'], source_manifest=manifest,
            loader=json.loads(loader_path.read_text()), loader_sha=fixed.file_sha(loader_path))
        gpu = json.loads((retained / 'gpu-identity.json').read_text())
        if (gpu.get('sealed_sha256') != fixed.file_sha(retained / 'sealed.json')
                or gpu.get('config_sha256') != fixed.file_sha(retained / 'generator-config.json')
                or gpu.get('materialization_profile') != host.folder_profile.PROFILE
                or gpu.get('materialization_sha256') != fixed.file_sha(retained / 'materialization.json')):
            raise ValueError('pilot physical GPU identity receipt differs')
        claimed = dispatch['environment']['gpu']['uuid']
        host.check_gpu_binding(gpu, seal, claimed, dispatch, manifest['files'])
        group = json.loads((stage.parent / 'run/status.json').read_text())['process_group_id']
        outside = os.readlink('/proc/self/ns/pid')
        for name, module, usage_key in (('packet-namespace.json', 'experiments.tennis.w8_strict_receiver_worker', 'packet_usage'),
                                       ('generator-namespace.json', 'experiments.tennis.w8_strict_host_model_worker', 'generation_usage')):
            ns = witness.namespace_completion(retained / name, expected_pgid=group,
                                               expected_outside=outside, expected_target=module)
            if (summary[usage_key].get('namespace_receipt_sha256') != fixed.file_sha(retained / name)
                    or summary[usage_key]['cpu_seconds'] + 0.05 < ns['children_cpu_seconds']):
                raise ValueError('pilot namespace CPU or receipt join differs')
        if summary['validation']['scope'] != 'context_qualification_only':
            raise ValueError('pilot cannot be a candidate')
        with target.open('x') as validation_stream:
            json.dump({'passed': True, 'scope': 'unit-0 context qualification only', 'automatic_promotion': False,
                       'checks': {'source_packet_gpu_namespace_and_context': True}}, validation_stream, sort_keys=True)
    except Exception as exc:
        with target.open('x') as validation_stream:
            json.dump({'passed': False, 'scope': 'unit-0 context qualification only', 'error': str(exc)}, validation_stream)
        raise


def make_pilot_spec(*, deadline: str, proof_local: Path, proof_remote: str) -> dict[str, Any]:
    """Prepare a review-only spec after a real, independently reviewed CPU gate exists."""
    digest = fixed.file_sha(proof_local)
    gate = json.loads(proof_local.read_text())
    if (gate.get('schema') != 1 or gate.get('scope') != CPU_SCOPE or gate.get('passed') is not True
            or not _sha(digest) or not Path(proof_remote).is_absolute()
            or not Path(proof_remote).is_relative_to(Path('/home/itec/emanuele/Datasets/pointstream-data'))):
        raise ValueError('actual root-reviewed proof gate and shared path required')
    check_cpu_gate(gate, gate_sha=digest, expected_sha=digest,
                   source={'source_sha256': gate.get('source_sha256'), 'snapshot_sha256': gate.get('snapshot_sha256'),
                           'git_head': gate.get('revision')},
                   environment_sha=gate.get('environment_sha256'), host_name=gate.get('host'))
    spec = host.make_materialized_spec(deadline)
    spec.pop('w8_paired_cost_profile', None)
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_witness_job']
    spec['arguments'] = ['{phase}']
    spec['scale'] = {'phase': {'smoke': 'pilot', 'full': 'noop'}}
    spec['smoke'] = {'seconds': 580, 'representative_basis':
        'One actual original-recipe 30-step 24-frame context; no candidate or automatic promotion.'}
    spec['full'] = {'seconds': 5}
    spec['budget_seconds'] = 700
    spec['validator_seconds'] = 60
    spec['witness_validator_cpu_seconds'] = 60
    spec['witness_cpu_ceiling_seconds'] = 1000
    spec['witness_caps'] = {'parent_cpu': 30, 'packet_cpu': 100, 'generator_cpu': 300,
                            'generator_wall': 310, 'external_cpu': 150}
    spec['w8_witness_profile'] = 'w8-context-pilot-v1'
    spec['witness_cpu_proof_sha256'] = digest
    spec['clock_policy_profile'] = host.w8_clock_profile.PROFILE
    spec['clock_reference_audit_sha256'] = host.CLOCK_AUDIT_SHA
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_witness_job', 'validate']
    spec['gpu_models'] = ['RTX 6000 Ada']
    spec['staged_inputs'].append({'name': 'cpu_proof', 'path': proof_remote, 'sha256': digest, 'extract': False})
    spec['inputs'].append({'path': proof_remote, 'sha256': digest})
    _pilot_spec(spec)
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    return spec


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('make-spec')
    make.add_argument('--deadline', required=True)
    make.add_argument('--cpu-proof-local', type=Path, required=True)
    make.add_argument('--cpu-proof-remote', required=True)
    make.add_argument('--output', type=Path, required=True)
    for action in ('pilot', 'noop', 'validate'):
        sub.add_parser(action)
    args = parser.parse_args()
    if args.action == 'make-spec':
        spec = make_pilot_spec(deadline=args.deadline, proof_local=args.cpu_proof_local,
                               proof_remote=args.cpu_proof_remote)
        with args.output.open('x') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.action == 'pilot':
        run_pilot(stage)
    elif args.action == 'validate':
        validate_pilot(stage, Path(os.environ['PS_VALIDATION_PATH']))
    else:
        if os.environ.get('PS_STAGE') != 'full':
            raise ValueError('pilot no-op is full-stage only')
        output = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        output.mkdir()
        with (output / 'noop.json').open('x') as stream:
            json.dump({'scope': 'no additional scientific execution'}, stream)


if __name__ == '__main__':
    main()
