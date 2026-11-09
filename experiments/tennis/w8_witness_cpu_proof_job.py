"""Bounded W8 CPU proof campaign, separate from root's later audit envelope.

No GPU execution or quality promotion occurs here. The proof only produces raw
receipts for independent review; it cannot approve itself for a witness job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import resource
import sys
import time
from typing import Any

_CPU_LIMIT_BEFORE_FIXED_IMPORT = resource.getrlimit(resource.RLIMIT_CPU)
from experiments.jobs.inbox import validate_spec  # noqa: E402
from experiments.tennis import w8_input_job as fixed  # noqa: E402
_CPU_LIMIT_AFTER_FIXED_IMPORT = resource.getrlimit(resource.RLIMIT_CPU)
from experiments.tennis import w8_strict_packet_job as packet  # noqa: E402

PROFILE = 'w8-exact-source-cpu-proof-v1'
NATIVE_INVENTORY_SHA = 'cd6bb8cf2a239d94f595fd92f823cf6406c018e1fbda745e81cd0c0e91d1b59b'
DATA_ROOT = Path('/home/itec/emanuele/Datasets/pointstream-data')


def check_lifecycle(record: dict[str, Any], campaign_pgid: int) -> None:
    """Check actual synthetic records; they are not native or quality evidence."""
    from experiments.tennis.w8_witness_namespace_proof import check_waited_cpu

    cases = record.get('cases')
    if (record.get('scope') != 'synthetic CPU namespace lifecycle only'
            or record.get('campaign_pgid') != campaign_pgid or type(campaign_pgid) is not int
            or campaign_pgid <= 0 or not isinstance(cases, list)
            or [case.get('case') for case in cases] != ['normal', 'timeout', 'term', 'kill']):
        raise ValueError('namespace lifecycle proof scope or cases differ')
    for case in cases:
        worker, init, detached = case['worker'], case['init'], case['detached']
        if (case.get('all_absent') is not True or worker['pgid'] != campaign_pgid
                or init['pgid'] != campaign_pgid or detached['pgid'] != detached['pid']
                or worker['ppid'] != init['pid'] or detached['ppid'] != worker['pid']
                or any(type(row.get('pid')) is not int or row['pid'] <= 0
                       or type(row.get('start_ticks')) is not int or row['start_ticks'] <= 0
                       for row in (worker, init, detached))):
            raise ValueError('namespace lifecycle host identity differs')
        receipt = case.get('receipt')
        if case['case'] == 'normal':
            if (case.get('launcher_rc') != 0 or not isinstance(receipt, dict)
                    or receipt.get('reason') != 'complete' or receipt.get('status') != 0
                    or receipt.get('complete_descendant_wait') is not True
                    or receipt['worker_host_identity']['host_pid'] != worker['pid']
                    or receipt['worker_host_identity']['start_ticks'] != worker['start_ticks']
                    or receipt['namespace_init']['host_pid'] != init['pid']
                    or receipt['namespace_init']['start_ticks'] != init['start_ticks']):
                raise ValueError('normal namespace lifecycle differs')
            check_waited_cpu(case['outer_waited_cpu_seconds'], receipt['init_cpu_seconds'],
                             receipt['children_cpu_seconds'])
        elif case['case'] == 'timeout':
            if case.get('launcher_rc') == 0 or not isinstance(receipt, dict) or receipt.get('reason') != 'timeout':
                raise ValueError('timeout namespace lifecycle differs')
        elif case.get('launcher_rc') != -({'term': signal.SIGTERM, 'kill': signal.SIGKILL}[case['case']]):
            raise ValueError('interrupted namespace lifecycle signal differs')


def check_protected_metadata(receipt: dict[str, Any], historical: dict[str, Any]) -> None:
    """Verify typed denial-target metadata without reading protected contents."""
    if (receipt.get('schema') != 1 or receipt.get('scope') != 'witness protected metadata only'
            or receipt.get('content_verified_now') is not False
            or receipt.get('historical_receipt_sha256') != packet.IDENTITY_SHA):
        raise ValueError('CPU proof must use stat-only protected metadata')
    from experiments.tennis import w8_protected_metadata

    w8_protected_metadata.validate(receipt, historical=historical, check_current=True)


def remaining(dispatch: dict[str, Any], reserve: float) -> None:
    clock = dispatch['campaign_clock']
    if (not math.isfinite(reserve) or reserve <= 0
            or time.monotonic() + reserve > min(clock['budget_end'], clock['phase_started'] + 530)):
        raise ValueError('remaining campaign time cannot cover required proof and cleanup')


def retain_file(source: Path, scratch: Path, output: Path) -> dict[str, Any]:
    """Copy an actual task-owned file for audit; reject aliases and overwrite."""
    root = scratch.resolve(strict=True)
    if source.is_symlink() or not source.is_file() or not source.resolve(strict=True).is_relative_to(root):
        raise ValueError('proof receipt must be an owned regular scratch file')
    digest = hashlib.sha256()
    with source.open('rb') as reader, output.open('xb') as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if fixed.file_sha(source) != digest.hexdigest():
        raise ValueError('proof receipt changed during retention')
    return {'sha256': digest.hexdigest(), 'bytes': output.stat().st_size}


def make_spec(deadline: str, inventory: str) -> dict[str, Any]:
    path = Path(inventory)
    if not path.is_absolute() or not path.is_relative_to(DATA_ROOT) or '..' in path.parts:
        raise ValueError('native inventory must be an absolute registered data path')
    spec = packet.make_spec(deadline)
    native = {'name': 'native_inventory', 'path': str(path), 'sha256': NATIVE_INVENTORY_SHA, 'extract': False}
    spec['staged_inputs'].append(native)
    spec['inputs'].append({'path': str(path), 'sha256': NATIVE_INVENTORY_SHA})
    spec.update(w8_combined_cpu_proof_profile=PROFILE, budget_seconds=600, validator_seconds=50,
                smoke={'seconds': 530, 'representative_basis': 'Exact-source CPU namespace, packet, native and policy proof.'},
                full={'seconds': 5}, entrypoint=['-m', 'experiments.tennis.w8_witness_cpu_proof_job'],
                arguments=['{phase}'], scale={'phase': {'smoke': 'run', 'full': 'noop'}},
                validator=['{python}', '-m', 'experiments.tennis.w8_witness_cpu_proof_job', 'validate'])
    bootstrap = Path(__file__).resolve().parent / 'w8_receiver_bootstrap.py'
    spec['witness_bootstrap_sha256'] = fixed.file_sha(bootstrap)
    spec['witness_input_job_sha256'] = fixed.file_sha(Path(fixed.__file__).resolve())
    spec['witness_system_leaves_sha256'] = fixed.file_sha(bootstrap.parent / 'w8_cuda_system_leaves.py')
    validate_spec(spec)
    return spec


def run(stage: Path) -> None:
    """Run real, bounded CPU proof leaves; emit raw evidence, not an approval."""
    from experiments.tennis import w8_witness_namespace_proof as lifecycle
    from experiments.tennis import w8_witness_profile as witness

    if (sys.platform != 'linux' or stage.name != 'smoke' or os.environ.get('PS_STAGE') != 'smoke'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PS_CLAIMED_GPU_UUID')
            or len(os.sched_getaffinity(0)) != 1):
        raise ValueError('CPU-hidden single-core fleet proof required')
    spec = json.loads((stage.parent / 'spec.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('w8_combined_cpu_proof_profile') != PROFILE or spec.get('budget_seconds') != 600
            or spec.get('smoke', {}).get('seconds') != 530 or spec.get('validator_seconds') != 50
            or spec.get('full', {}).get('seconds') != 5):
        raise ValueError('exact combined CPU proof budget profile required')
    # Preserve the inherited hard ceiling for the bounded workers;
    # the one-core fleet campaign wall remains the aggregate 600-second bound.
    if _CPU_LIMIT_BEFORE_FIXED_IMPORT != _CPU_LIMIT_AFTER_FIXED_IMPORT:
        raise ValueError('CPU limit changed during fixed module import')
    cpu_limit = fixed.set_cpu_soft_limit(100, required_child=100)
    started = time.monotonic()
    remaining(dispatch, 520)  # packet 80 + lifecycle 90 + strict 120 + assessor 45 + native 135 + setup/publish 50
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    publish = scratch / 'publish'
    publish.mkdir(exist_ok=False)
    raw = publish / 'raw'
    raw.mkdir()
    prepared = packet.run_proof(stage)
    if (prepared.get('scope') != 'strict_packet_cpu_proof_only'
            or Path(prepared['scratch']).resolve(strict=True) != scratch):
        raise ValueError('packet proof scratch differs from fleet')
    retained: dict[str, Any] = {}

    def keep(name: str, source: str | Path) -> None:
        if not name or Path(name).name != name or name in retained:
            raise ValueError('invalid or duplicate proof receipt name')
        retained[name] = retain_file(Path(source), scratch, raw / name)

    for name, path in prepared['proof_receipts'].items():
        keep('packet-' + name, path)
    for unit in fixed.UNITS:
        keep('packet-unit-' + unit + '.npz', prepared['decoder_out'] / 'publish' / 'units' / f'{unit}.npz')
    keep('packet-decoded.pt', prepared['decoder_out'] / 'decoded.pt')
    remaining(dispatch, 420)  # lifecycle 90 + strict 120 + assessor 45 + native 135 + publication 30
    life_dir = scratch / 'namespace-lifecycle'
    life_dir.mkdir()
    unshare = Path('/usr/bin/unshare').resolve(strict=True)
    unshare_sha = hashlib.sha256(unshare.read_bytes()).hexdigest()
    lifecycle.run(life_dir, unshare, unshare_sha)
    lifecycle_path = life_dir / 'proof.json'
    lifecycle_record = json.loads(lifecycle_path.read_text())
    check_lifecycle(lifecycle_record, os.getpgrp())
    keep('namespace-lifecycle.json', lifecycle_path)
    for name in ('normal', 'timeout', 'term', 'kill'):
        case_dir = life_dir / name
        for basename in ('worker.json', 'detached.json', 'receipt.json', 'receipt.json.started'):
            path = case_dir / basename
            if path.is_file():
                keep(f'namespace-{name}-{basename}', path)
    remaining(dispatch, 330)  # strict 120 + assessor 45 + native 135 + publication 30
    isolation = witness.run_isolation_proof(prepared, worker_cpu=40, worker_wall=60)
    if (isolation.get('scope') != 'cpu_isolation_proof_only'
            or [row['runtime'] for row in isolation.get('runs', [])] != ['main', 'jcv']):
        raise ValueError('both real strict runtime proofs required')
    for row in isolation['runs']:
        for key in ('config_path', 'receipt_path', 'namespace_path', 'namespace_started_path', 'self_leaves_path'):
            keep(f"strict-{row['runtime']}-{key}.json", row[key])
    remaining(dispatch, 210)  # assessor 45 + native 135 + publication 30
    from experiments.tennis import w8_witness_assessment as assessment
    assessed = assessment.run_assessment_policy_proof(prepared)
    if assessed.get('scope') != 'assessment policy component only; no candidate or quality claim':
        raise ValueError('actual assessment policy proof required')
    for name, path in assessed['receipt_paths'].items():
        keep('assessor-' + name, path)
    for index, case in enumerate(('normal', 'timeout', 'kill')):
        remaining(dispatch, (3 - index) * 45 + 30)
        result = witness.run_native_proof(prepared, inventory_path=prepared['native_inventory'],
                                          case=case)
        if (result.get('scope') != 'native_cpu_lifecycle_proof_only' or result.get('case') != case
                or not isinstance(result.get('receipt_paths'), dict)):
            raise ValueError('actual native proof case and raw receipts required')
        for name, path in result['receipt_paths'].items():
            keep(f'native-{case}-{name}', path)
        if case == 'normal':
            keep('native-normal-decoded.yuv', Path(result['output']) / 'native.yuv')
    remaining(dispatch, 10)
    summary = {'schema': 1, 'scope': 'raw CPU proof receipts only; independent root audit required',
               'quality_claim': False, 'gpu_claim': False, 'source': dispatch['code'],
               'environment_sha256': dispatch['staged_environment']['sha256'],
               'host': dispatch['environment']['alias'], 'spec_sha256': dispatch['code']['spec_sha256'],
               'unshare_sha256': unshare_sha, 'affinity': sorted(os.sched_getaffinity(0)),
               'cpu_limit_telemetry': {'before_fixed_import': list(_CPU_LIMIT_BEFORE_FIXED_IMPORT),
                   'after_fixed_import': list(_CPU_LIMIT_AFTER_FIXED_IMPORT), 'setup': cpu_limit},
               'elapsed_seconds': time.monotonic() - started, 'receipts': retained}
    with (publish / 'raw-summary.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())



def validation_receipt() -> dict[str, Any]:
    """Describe only the raw-publication checks completed by this validator."""
    return {'passed': True, 'scope': 'raw publication completeness only; root audit required',
            'quality_claim': False, 'gpu_claim': False,
            'checks': {'publication_and_source_binding': True, 'raw_receipt_hashes': True,
                       'namespace_lifecycle': True, 'cpu_policy_receipts': True,
                       'assessment_policy_component': True, 'native_yuv_and_teardown': True}}

def validate(stage: Path, destination: Path) -> None:
    """Check raw publication completeness; independent audit remains mandatory."""
    import tarfile

    dispatch = json.loads((stage / 'dispatch.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    archive = stage / 'published.tar'
    if fixed.file_sha(archive) != staging['published']['sha256']:
        raise ValueError('CPU proof publication hash differs')
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        files = {entry.name: entry for entry in entries if entry.isfile()}
        if len(files) != sum(entry.isfile() for entry in entries):
            raise ValueError('duplicate CPU proof publication member')
        summary_file = tar.extractfile(files['publish/raw-summary.json'])
        if summary_file is None:
            raise ValueError('CPU proof summary missing')
        summary = json.load(summary_file)
        receipts = summary.get('receipts')
        if (summary.get('scope') != 'raw CPU proof receipts only; independent root audit required'
                or summary.get('gpu_claim') is not False or summary.get('quality_claim') is not False
                or summary.get('source') != dispatch['code']
                or summary.get('environment_sha256') != dispatch['staged_environment']['sha256']
                or summary.get('host') != dispatch['environment']['alias']
                or summary.get('spec_sha256') != dispatch['code']['spec_sha256']
                or not isinstance(receipts, dict) or not receipts
                or set(files) != {'publish/raw-summary.json'} | {f'publish/raw/{name}' for name in receipts}
                or any(not e.isfile() and not e.isdir() for e in entries)):
            raise ValueError('CPU proof source or publication membership differs')
        for name, info in receipts.items():
            if Path(name).name != name:
                raise ValueError('invalid raw receipt name')
            member = tar.extractfile(files[f'publish/raw/{name}'])
            if member is None:
                raise ValueError('raw receipt missing')
            digest = hashlib.sha256()
            for chunk in iter(lambda: member.read(1024 * 1024), b''):
                digest.update(chunk)
            if digest.hexdigest() != info['sha256'] or files[f'publish/raw/{name}'].size != info['bytes']:
                raise ValueError('raw receipt identity differs')
        lifecycle_stream = tar.extractfile(files['publish/raw/namespace-lifecycle.json'])
        if lifecycle_stream is None:
            raise ValueError('namespace lifecycle receipt missing')
        lifecycle = json.load(lifecycle_stream)
        check_lifecycle(lifecycle, lifecycle['campaign_pgid'])
        required = {'packet-packet-config.json', 'packet-packet-decoded-manifest.json',
                    'packet-packet-self-leaves.json', 'packet-packet-namespace.json',
                    'packet-packet-cpu-only.json', 'packet-decoded.pt',
                    'packet-protected-identity.json',
                    *(f'namespace-{case}-receipt.json.started' for case in ('normal', 'timeout', 'term', 'kill')),
                    *(f'packet-unit-{unit}.npz' for unit in fixed.UNITS),
                    *(f'strict-{runtime}-{name}.json' for runtime in ('main', 'jcv')
                      for name in ('config_path', 'receipt_path', 'namespace_path', 'namespace_started_path', 'self_leaves_path')),
                    'assessor-policy.json', 'assessor-config.json', 'assessor-namespace.json',
                    'assessor-namespace.started',
                    *(f'native-{case}-{name}' for case in ('normal', 'timeout', 'kill')
                      for name in ('namespace_started', 'teardown', 'launcher_execution',
                                   'native_handle', 'native_handle_ready', 'native_handle_committed')),
                    'native-normal-namespace_final', 'native-normal-native_seal', 'native-normal-decoded.yuv',
                    'native-normal-native_execution',
                    'native-normal-control_execution', 'native-timeout-interruption',
                    'native-timeout-namespace_final', 'native-kill-interruption'}
        if not required <= receipts.keys():
            raise ValueError('required raw proof receipt absent')
        def raw_json(name: str) -> Any:
            stream = tar.extractfile(files['publish/raw/' + name])
            if stream is None:
                raise ValueError('required raw JSON missing')
            return json.load(stream)
        spec = json.loads((stage.parent / 'spec.json').read_text())
        identity_rows = [item for item in spec['staged_inputs'] if item.get('name') == 'identity']
        if len(identity_rows) != 1 or identity_rows[0].get('sha256') != packet.IDENTITY_SHA:
            raise ValueError('historical identity metadata input differs')
        identity_path = Path(identity_rows[0]['path'])
        from experiments.tennis import w8_protected_metadata
        historical = w8_protected_metadata.read_historical(identity_path, packet.IDENTITY_SHA)
        check_protected_metadata(raw_json('packet-protected-identity.json'), historical)
        from experiments.tennis import w8_cuda_system_leaves as policy
        packet_leaves = raw_json('packet-packet-self-leaves.json')
        policy.validate_self_leaf_identity(packet_leaves, raw_json('packet-packet-namespace.json')['worker_host_identity'])
        if packet_leaves != raw_json('packet-packet-decoded-manifest.json')['isolation']['approved_leaves']:
            raise ValueError('packet self leaves differ from decoded isolation')
        for runtime in ('main', 'jcv'):
            observed = raw_json(f'strict-{runtime}-receipt_path.json')
            leaves = raw_json(f'strict-{runtime}-self_leaves_path.json')
            policy.validate_self_leaf_identity(leaves, raw_json(f'strict-{runtime}-namespace_path.json')['worker_host_identity'])
            if leaves != observed['isolation']['approved_leaves']:
                raise ValueError('strict self leaves differ from isolation')
            if (observed.get('scope') != 'cpu_isolation_proof_only'
                    or observed.get('cuda_initialized') is not False or observed.get('visible_devices') != ''):
                raise ValueError('strict CPU proof receipt differs')
        assessor = raw_json('assessor-policy.json')
        policy.validate_self_leaf_identity(assessor['isolation']['approved_leaves'],
            raw_json('assessor-namespace.json')['worker_host_identity'])
        if (assessor.get('scope') != 'assessment policy component only; no candidate or quality claim'
                or assessor.get('gpu_claim') is not False or assessor.get('quality_claim') is not False
                or assessor.get('output_write') is not True
                or assessor.get('undeclared_read_denial') not in (1, 13)
                or assessor.get('undeclared_execute_denial') not in (1, 13)):
            raise ValueError('assessment role policy proof differs')
        for case in ('normal', 'timeout', 'kill'):
            if raw_json(f'native-{case}-teardown').get('owned_init_and_worker_absent') is not True:
                raise ValueError('native lifecycle teardown missing')
            if (raw_json(f'native-{case}-native_handle_ready').get('sha256')
                    != receipts[f'native-{case}-native_handle']['sha256']):
                raise ValueError('native observed handle identity differs')
        native = raw_json('native-normal-native_seal')
        if (native.get('yuv_sha256') != '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff'
                or native.get('yuv_sha256') != receipts['native-normal-decoded.yuv']['sha256']
                or native.get('inventory_sha256') != NATIVE_INVENTORY_SHA or native.get('exit_code') != 0):
            raise ValueError('actual native YUV proof differs')
    with destination.open('x') as stream:
        json.dump(validation_receipt(), stream, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    spec = sub.add_parser('make-spec')
    spec.add_argument('--deadline', required=True)
    spec.add_argument('--native-inventory', required=True)
    spec.add_argument('--output', required=True, type=Path)
    for action in ('run', 'noop', 'validate'):
        sub.add_parser(action)
    args = parser.parse_args()
    if args.action == 'make-spec':
        result = make_spec(args.deadline, args.native_inventory)
        with args.output.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    elif args.action == 'run':
        run(Path(os.environ['PS_STAGE_DIR']))
    elif args.action == 'validate':
        validate(Path(os.environ['PS_STAGE_DIR']), Path(os.environ['PS_VALIDATION_PATH']))
    elif args.action == 'noop':
        if os.environ.get('PS_STAGE') != 'full' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU-only full completion marker required')
        publish = Path(os.environ['PS_SCRATCH_DIR']) / 'publish'
        publish.mkdir()
        with (publish / 'completion.json').open('x') as stream:
            json.dump({'scope': 'CPU proof completion marker; no root approval'}, stream)


if __name__ == '__main__':
    main()
