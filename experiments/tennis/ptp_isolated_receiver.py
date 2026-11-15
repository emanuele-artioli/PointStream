"""Bounded PTP1 filesystem-denial adapter using the existing Linux mechanism.

No network/MCP isolation is established. Native invocation receipts describe
ordered calls under inherited Landlock, not independently observed child births.
Expected reconstructed pixels are parent-only inputs, never decoder hints.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from experiments.tennis import k5d_isolated_receiver as kernel

SCHEMA = 'ptp-filesystem-denial-v1'
SCOPE = 'fresh-exec PTP1 filesystem-denied receiver; inherited native restrictions'
RUNTIME_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
ENV_KEYS = {'PATH', 'PYTHONPATH', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'PS_SCRATCH_DIR', 'TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'LC_CTYPE'}
ROLES = ('source_archive', 'staged_source_archive', 'source_frame', 'source_yuv',
         'ground_truth', 'encoder_state', 'private_detector')
DECODER_SOURCE = (
    'src/__init__.py', 'src/codecs/__init__.py', 'src/codecs/svtav1.py', 'src/codecs/quality.py',
    'src/pointstream/__init__.py', 'src/pointstream/codec.py', 'src/pointstream/tracked_patch.py',
    'experiments/__init__.py', 'experiments/tennis/__init__.py',
    'experiments/tennis/ptp_isolated_receiver.py', 'experiments/tennis/k5d_isolated_receiver.py',
    'experiments/tennis/w8_cuda_system_leaves.py',
)


def write_new(path: Path, value: Any) -> None:
    with path.open('x') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')


def amend_policy(config: dict[str, Any], source_root: Path, identity: dict[str, str]) -> dict[str, Any]:
    """Amend an existing reviewed inventory; never discover generic runtime files.

    Caller must review/freeze the new policy and source hashes before admission.
    Existing runtime/executable/system inventories remain byte-for-byte values.
    """
    result = json.loads(json.dumps(config))
    if identity.get('environment_sha256') != RUNTIME_SHA:
        raise ValueError('only the authenticated packed32e runtime is admitted')
    result['execution_identity'] = identity
    for profile in result['profiles'].values():
        profile['runtime_archive_sha256'] = RUNTIME_SHA
        profile['source_files'] = {}
        for name in DECODER_SOURCE:
            path = source_root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f'required reviewed decoder source missing: {name}')
            profile['source_files'][name] = {'sha256': kernel.sha(path), 'size': path.stat().st_size}
    return result


def protected_evidence(spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if set(spec) != set(ROLES) or any(not isinstance(spec[k], list) or not spec[k] for k in ROLES):
        raise ValueError('all actual source/cache/model roles are required')
    paths: dict[str, list[Path]] = {key: [] for key in kernel.CATEGORIES}
    category = {'staged_source_archive': 'source_archive', 'source_yuv': 'encoder_state'}
    for role, rows in spec.items():
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'path', 'sha256', 'size'}:
                raise ValueError('protected input needs path/hash/size')
            path = Path(row['path'])
            if (not path.is_absolute() or not path.is_file() or type(row['size']) is not int
                    or path.stat().st_size != row['size'] or kernel.sha(path) != row['sha256']):
                raise ValueError('actual protected input identity mismatch')
            paths[category.get(role, role)].append(path)
    evidence, _ = kernel._protected(paths)
    return evidence


def reject_protected_grants(evidence: dict[str, list[dict[str, Any]]], granted: list[Path], work: Path) -> None:
    denied = {(r['device'], r['inode']) for rows in evidence.values() for r in rows}
    if any((p.stat().st_dev, p.stat().st_ino) in denied for p in granted):
        raise ValueError('protected inode aliases a granted file')
    if any(Path(r['resolved']).is_relative_to(work.resolve()) for rows in evidence.values() for r in rows):
        raise ValueError('protected target lies inside recursively writable receiver directory')


def _no_extra_fds() -> bool:
    return not any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd'))


def recorded_decode(packet: Path, output: Path, scratch: Path, decoder: str,
                    decode: Callable[..., dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Record every subprocess.run issued by this trusted decoder, sequentially.

    No monkeypatch changes its pixels or arguments. No claim of independently
    observed native PID/birth identity is made; Landlock propagates to children.
    """
    original = subprocess.run
    calls: list[dict[str, Any]] = []

    def invoke(command: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(command, list) or not command or Path(command[0]).resolve() != Path(decoder).resolve():
            raise ValueError('PTP decoder attempted an undeclared executable')
        started = time.monotonic()
        done = original(command, *args, **kwargs)
        calls.append({'index': len(calls), 'argv': command, 'argv_sha256': kernel._object_sha(command),
                      'returncode': done.returncode, 'seconds': time.monotonic() - started,
                      'restriction_scope': 'inherits restricted worker Landlock and namespaces'})
        return done

    setattr(subprocess, 'run', invoke)
    try:
        result = decode(packet, output, scratch=scratch, decoder=decoder)
    finally:
        setattr(subprocess, 'run', original)
    return result, calls


def verify_receipt(receipt: dict[str, Any], *, context: dict[str, Any], policy_sha: str,
                   packet_sha: str, observer: dict[str, Any]) -> None:
    if receipt.get('schema') != SCHEMA or receipt.get('scope') != SCOPE:
        raise ValueError('PTP proof schema/scope mismatch')
    proof = receipt['filesystem_proof']
    if (proof.get('context') != context or proof.get('policy_sha256') != policy_sha
            or proof.get('worker_observer') != observer or proof.get('pid') != 1
            or proof.get('birth') != observer.get('birth') or observer.get('namespace_pids', [])[-1:] != [1]
            or proof.get('namespace', {}).get('namespaces') != observer.get('namespaces')
            or proof.get('threads_before_lock') != 1 or proof.get('extra_fds_before_lock') is not False
            or proof.get('cuda_visible_devices') != '' or proof.get('landlock_abi', 0) < 3
            or proof.get('handled_mask') != kernel.leaves.handled_rights(proof['landlock_abi'])
            or proof.get('runtime_archive_sha256') != RUNTIME_SHA
            or proof.get('network_isolation') is not False or proof.get('mcp_isolation') is not False
            or proof.get('native_scope') != 'inherited restrictions; no independent native birth observation'):
        raise ValueError('PTP worker identity/restriction evidence incomplete')
    environment = proof.get('environment', {})
    if (not isinstance(environment, dict) or set(environment) - ENV_KEYS or environment.get('PATH') != '/nonexistent'
            or environment.get('CUDA_VISIBLE_DEVICES') != '' or environment.get('PYTHONNOUSERSITE') != '1'
            or environment.get('PYTHONDONTWRITEBYTECODE') != '1'
            or any(environment.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))):
        raise ValueError('restricted environment proof incomplete')
    policy = proof['actual_policy']
    if kernel._object_sha(policy) != proof.get('actual_policy_sha256'):
        raise ValueError('actual policy digest mismatch')
    denied = proof.get('denials')
    if not isinstance(denied, dict) or set(denied) != set(kernel.CATEGORIES):
        raise ValueError('incomplete denied categories')
    for rows in denied.values():
        if not rows or any(r.get('errno') not in (errno.EACCES, errno.EPERM) or not r.get('inode')
                           or not r.get('sha256') for r in rows):
            raise ValueError('actual read denial missing/inconclusive')
    if policy.get('denials') != {k: [{a: b for a, b in r.items() if a != 'errno'} for r in rows] for k, rows in denied.items()}:
        raise ValueError('denial targets differ from frozen policy')
    if set(proof.get('protected_roles', {})) != set(ROLES) or any(not proof['protected_roles'][r] for r in ROLES):
        raise ValueError('protected roles incomplete')
    by_role = {'staged_source_archive': 'source_archive', 'source_yuv': 'encoder_state'}
    for role, rows in proof['protected_roles'].items():
        for row in rows:
            matches = [r for r in denied[by_role.get(role, role)] if r['path'] == row['path'] and r['sha256'] == row['sha256']]
            if len(matches) != 1:
                raise ValueError('protected role is not linked to actual denial')
    namespace = proof['namespace']
    if (namespace.get('private_proc') is not True or namespace.get('initial_pids') != [1]
            or any(namespace['namespaces'][k] == namespace['parent_namespaces'][k] for k in ('user', 'pid', 'mnt', 'ipc'))):
        raise ValueError('private namespace proof incomplete')
    decoded = receipt['decode']
    for key in ('profile_sha256', 'runtime_inventory_sha256', 'source_inventory_sha256'):
        if not isinstance(proof.get(key), str) or len(proof[key]) != 64:
            raise ValueError('generic runtime/source inventory identity missing')
    if decoded.get('packet_sha256') != packet_sha or decoded.get('decoded_frames') != len(decoded.get('rgb_frame_sha256', [])):
        raise ValueError('decoded packet/frame binding differs')
    calls = proof.get('native_calls', [])
    expected = 3 if decoded.get('patch_bytes', 0) else 2
    if len(calls) != expected:
        raise ValueError('missing native invocation facts')
    for index, call in enumerate(calls):
        argv = call.get('argv')
        if (call.get('index') != index or call.get('returncode') != 0 or not isinstance(argv, list) or not argv
                or kernel._object_sha(argv) != call.get('argv_sha256')
                or str(Path(argv[0]).resolve()) != proof['decoder_path']
                or call.get('restriction_scope') != 'inherits restricted worker Landlock and namespaces'):
            raise ValueError('invalid ordered native invocation')
    if calls[0]['argv'] != [proof['decoder_path'], '--version'] or any('-i' not in call['argv'] for call in calls[1:]):
        raise ValueError('native version/base/patch call matrix differs')


def _worker(work: Path) -> None:
    launch = json.loads((work / 'launch.json').read_text())
    if len(os.listdir('/proc/self/task')) != 1 or not _no_extra_fds() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU worker threads/descriptors/environment invalid')
    namespace = kernel._private_proc(launch['parent_namespaces'], launch['outer_pid'])
    print('READY', flush=True)
    observed_path = work / 'worker-observer.json'
    deadline = time.monotonic() + 30
    while not observed_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('parent observation missing')
        time.sleep(.01)
    observer = json.loads(observed_path.read_text())
    if observer['birth'] != kernel._birth() or observer['namespaces'] != namespace['namespaces'] or observer['namespace_pids'][-1:] != [1]:
        raise ValueError('parent worker observation differs')
    config = launch['config']
    files, execute, system = kernel.inventory(config)
    packet = work / 'packet.ptp'
    reject_protected_grants(launch['denials'], files + execute + system + [packet], work)
    files += [packet]
    profile = config['profiles'][os.uname().nodename.split('.')[0]]
    directories = kernel._import_directories(profile)
    policy = {'read_files': [str(p) for p in files], 'execute_files': [str(p) for p in execute],
              'system_leaves': [str(p) for p in system], 'writable_output': str(work),
              'read_directories': sorted(str(p) for p in kernel._read_directories(files, execute, system, directories)),
              'denials': launch['denials']}
    policy = json.loads(json.dumps(policy))
    threads, extra = len(os.listdir('/proc/self/task')), not _no_extra_fds()
    if threads != 1 or extra:
        raise ValueError('threads/descriptors appeared before lock')
    abi, mask = kernel._restrict(files, execute, system, work, directories)
    kernel._denials(launch['denials'])
    if kernel.sha(packet) != launch['packet_sha256']:
        raise ValueError('positive packet read failed')
    sys.prefix = profile['runtime_prefix']
    # Decode code and NumPy are imported only after the kernel restriction.
    from src.pointstream import tracked_patch
    decoded, calls = recorded_decode(packet, work / 'decoded.rgb', work, launch['decoder'], tracked_patch.decode)
    proof = {'context': launch['context'], 'policy_sha256': launch['policy_sha256'],
             'runtime_archive_sha256': profile['runtime_archive_sha256'],
             'profile_sha256': kernel._object_sha(profile),
             'runtime_inventory_sha256': kernel._object_sha(profile['runtime_files']),
             'source_inventory_sha256': kernel._object_sha(profile['source_files']), 'landlock_abi': abi, 'handled_mask': mask,
             'pid': os.getpid(), 'birth': kernel._birth(), 'worker_observer': observer, 'namespace': namespace,
             'threads_before_lock': threads, 'extra_fds_before_lock': extra, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
             'environment': dict(os.environ),
             'protected_roles': launch['protected_roles'], 'denials': launch['denials'],
             'actual_policy': policy, 'actual_policy_sha256': kernel._object_sha(policy),
             'native_calls': calls, 'decoder_path': launch['decoder'],
             'native_scope': 'inherited restrictions; no independent native birth observation',
             'network_isolation': False, 'mcp_isolation': False}
    receipt = {'schema': SCHEMA, 'scope': SCOPE, 'decode': decoded, 'filesystem_proof': proof}
    verify_receipt(receipt, context=launch['context'], policy_sha=launch['policy_sha256'], packet_sha=launch['packet_sha256'], observer=observer)
    write_new(work / 'receiver.json', receipt)



def expected_worker_policy(*, files: list[Path], execute: list[Path], system: list[Path],
                           import_directories: list[Path], work: Path, evidence: dict[str, Any],
                           profile: dict[str, Any], parent_pid: int) -> dict[str, Any]:
    """Construct the private PID-1 policy after receiver namespace verification.

    Only the two reviewed literal self-proc leaves differ between namespaces;
    no other PID or resource path is rewritten.
    """
    rows = profile['system_leaves']
    if len(rows) != len(system) or type(parent_pid) is not int or parent_pid <= 1:
        raise ValueError('invalid parent system-leaf inventory')
    special = {'/proc/self/maps': 'maps', '/proc/self/status': 'status'}
    seen: set[str] = set()
    worker_system = []
    for declared, resolved in zip(rows, system):
        name = declared['path']
        if name in special:
            if name in seen or resolved != Path('/proc') / str(parent_pid) / special[name]:
                raise ValueError('unexpected parent self-proc resolution')
            seen.add(name)
            worker_system.append(Path('/proc/1') / special[name])
        else:
            worker_system.append(resolved)
    if seen != set(special):
        raise ValueError('reviewed self-proc leaves missing')
    packet_files = files + [work / 'packet.ptp']
    return {'read_files': [str(p) for p in packet_files], 'execute_files': [str(p) for p in execute],
            'system_leaves': [str(p) for p in worker_system], 'writable_output': str(work),
            'read_directories': sorted(str(p) for p in kernel._read_directories(
                packet_files, execute, worker_system, import_directories)), 'denials': evidence}


def equal_policy(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare grant lists as multisets; retain exact non-grant fields and duplicates."""
    grants = {'read_files', 'execute_files', 'system_leaves', 'read_directories'}
    if set(actual) != set(expected):
        return False
    for key in actual:
        if key in grants:
            a, b = actual[key], expected[key]
            if (not isinstance(a, list) or not isinstance(b, list)
                    or any(not isinstance(v, str) for v in a + b) or sorted(a) != sorted(b)):
                return False
        elif actual[key] != expected[key]:
            return False
    return True

def replay(packet: Path, work: Path, *, policy_path: Path, policy_sha: str, packet_sha: str,
           protected: dict[str, Any], context: dict[str, Any], expected: dict[str, Any],
           decoder: Path, timeout: int = 900) -> dict[str, Any]:
    """Parent-owned orchestration. Expected hash arrays never enter worker inputs."""
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1 or not 1 <= timeout <= 900:
        raise ValueError('Linux, one CPU and bounded timeout required')
    kernel._context(context)
    work = work.resolve()
    kernel._canonical(packet)
    kernel._canonical(decoder)
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True)
    if not work.resolve().is_relative_to(scratch) or work.resolve() == scratch or work.exists():
        raise ValueError('new private receiver directory below scratch required')
    if kernel.sha(packet) != packet_sha or kernel.sha(policy_path) != policy_sha:
        raise ValueError('packet/policy hash mismatch')
    config = json.loads(policy_path.read_text())
    if config['execution_identity'] != context['execution_identity'] or context['execution_identity']['environment_sha256'] != RUNTIME_SHA:
        raise ValueError('packed runtime/source execution identity differs')
    files, execute, system = kernel.inventory(config)
    profile = config['profiles'][os.uname().nodename.split('.')[0]]
    if profile.get('runtime_archive_sha256') != RUNTIME_SHA or decoder not in execute:
        raise ValueError('runtime/decoder absent from declared generic inventory')
    evidence = protected_evidence(protected)
    reject_protected_grants(evidence, files + execute + system + [packet], work)
    work.mkdir(mode=0o700, parents=False, exist_ok=False)
    shutil.copyfile(packet, work / 'packet.ptp')
    launch = {'config': config, 'policy_sha256': policy_sha, 'context': context, 'denials': evidence,
              'protected_roles': protected, 'packet_sha256': packet_sha, 'decoder': str(decoder),
              'parent_namespaces': kernel._namespaces('self'), 'outer_pid': int(os.readlink('/proc/self'))}
    write_new(work / 'launch.json', launch)
    python = kernel._canonical(Path(profile['python']))
    if python not in execute:
        raise ValueError('Python missing from executable inventory')
    unshare_row = profile['unshare']
    unshare = kernel._file(Path(unshare_row['path']), unshare_row['sha256'], unshare_row['size'])
    env = {'PATH': '/nonexistent', 'PYTHONPATH': os.pathsep.join(str(p) for p in kernel._import_directories(profile)),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
           'PS_SCRATCH_DIR': str(work), 'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work), 'XDG_CACHE_HOME': str(work)}
    command = [str(unshare), '--user', '--map-root-user', '--pid', '--mount', '--ipc', '--fork',
               '--kill-child=SIGKILL', '--mount-proc', str(python), '-S', '-m', 'experiments.tennis.ptp_isolated_receiver', 'worker', str(work)]
    started = time.monotonic()
    with kernel._launch_receiver(command, env) as process:
        try:
            if process.stdout is None or not select.select([process.stdout], [], [], min(120, timeout))[0] or process.stdout.readline() != b'READY\n':
                raise RuntimeError('PTP receiver startup unavailable; no denial qualification')
            launcher = kernel._translate_child(process.pid, launch['outer_pid'], launch['parent_namespaces']['pid'])
            if launcher['host_ppid'] != launch['outer_pid'] or launcher['host_pgid'] != os.getpgrp():
                raise ValueError('launcher left the parent process group')
            children = Path(f"/proc/{launcher['pid']}/task/{launcher['pid']}/children").read_text().split()
            if len(children) != 1:
                raise ValueError('ambiguous private worker')
            observer = kernel._proc_identity(int(children[0]))
            if observer['host_ppid'] != launcher['pid'] or observer['host_pgid'] != launcher['host_pgid'] or observer['namespace_pids'][-1:] != [1]:
                raise ValueError('worker lineage differs')
            write_new(work / 'worker-observer.json', observer)
            _, stderr = process.communicate(timeout=max(1, timeout - (time.monotonic() - started)))
            if process.returncode:
                raise RuntimeError(f'PTP restricted receiver failed: {process.returncode}: {stderr[-2000:]!r}')
        except BaseException:
            kernel._stop_launcher(process)
            raise
    receipt = json.loads((work / 'receiver.json').read_text())
    verify_receipt(receipt, context=context, policy_sha=policy_sha, packet_sha=packet_sha, observer=observer)
    proof = receipt['filesystem_proof']
    expected_policy = expected_worker_policy(files=files, execute=execute, system=system,
        import_directories=kernel._import_directories(profile), work=work, evidence=evidence,
        profile=profile, parent_pid=launch['outer_pid'])
    if (not equal_policy(proof['actual_policy'], expected_policy) or proof['protected_roles'] != protected
            or proof['profile_sha256'] != kernel._object_sha(profile)
            or proof['runtime_inventory_sha256'] != kernel._object_sha(profile['runtime_files'])
            or proof['source_inventory_sha256'] != kernel._object_sha(profile['source_files'])):
        raise ValueError('actual policy/runtime differs from parent-approved inventory')
    if any(proof['environment'].get(k) != v for k, v in env.items()):
        raise ValueError('actual worker environment differs from launch')
    output = work / 'decoded.rgb' 
    if (kernel.sha(output) != receipt['decode']['decoded_sha256'] or receipt['decode']['decoded_sha256'] != expected['decoded_sha256']
            or receipt['decode']['rgb_frame_sha256'] != expected['rgb_frame_sha256']):
        raise ValueError('restricted receiver pixels differ from authenticated expected reconstruction')
    receipt['parent_validation'] = {'exact_rgb_and_frame_hashes': True, 'wall_seconds': time.monotonic() - started,
                                    'launcher_observer': launcher, 'expected_supplied_to_decoder': False}
    write_new(work / 'qualification.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('worker', 'replay', 'amend-policy'))
    parser.add_argument('work', type=Path)
    for name in ('packet', 'policy', 'protected', 'context', 'expected', 'decoder', 'source-root', 'identity'):
        parser.add_argument('--' + name, type=Path)
    for name in ('packet-sha256', 'policy-sha256'):
        parser.add_argument('--' + name)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.command == 'worker':
        _worker(args.work)
        return
    if args.command == 'amend-policy':
        if any(getattr(args, k) is None for k in ('policy', 'source_root', 'identity')):
            parser.error('amend-policy requires existing policy, source-root and identity JSON')
        write_new(args.work, amend_policy(json.loads(args.policy.read_text()), args.source_root, json.loads(args.identity.read_text())))
        return
    required = ('packet', 'policy', 'protected', 'context', 'expected', 'decoder', 'packet_sha256', 'policy_sha256')
    if any(getattr(args, k) is None for k in required):
        parser.error('replay requires packet/policy hashes, protected/context/expected JSON and decoder')
    result = replay(args.packet, args.work, policy_path=args.policy, policy_sha=args.policy_sha256,
                    packet_sha=args.packet_sha256, protected=json.loads(args.protected.read_text()),
                    context=json.loads(args.context.read_text()), expected=json.loads(args.expected.read_text()),
                    decoder=args.decoder, timeout=args.timeout)
    print(json.dumps({'schema': result['schema'], 'qualified': True, 'output': str(args.work / 'qualification.json')}))


if __name__ == '__main__':
    main()
