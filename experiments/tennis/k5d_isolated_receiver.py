"""Fail-closed fresh-exec Landlock receiver for authenticated Ball packets.

The profile is a reviewed, host-specific exact-file inventory. This module does
not collect or approve one, dispatch jobs, or claim that any witness has run.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import select
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from experiments.tennis import w8_cuda_system_leaves as leaves

CATEGORIES = ('source_archive', 'source_frame', 'ground_truth', 'encoder_state', 'private_detector')
SCOPE = 'actual isolated packet receiver with post-exec filesystem denial'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _canonical(path: Path) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f'noncanonical path: {path}')
    return path


def _file(path: Path, digest: str, size: int) -> Path:
    _canonical(path)
    if not path.is_file() or type(size) is not int or path.stat().st_size != size or sha(path) != digest:
        raise ValueError(f'file identity differs: {path}')
    return path


def _manifest(root: Path, entries: dict[str, Any]) -> list[Path]:
    _canonical(root)
    if not root.is_dir() or not isinstance(entries, dict) or not entries:
        raise ValueError('nonempty rooted file manifest required')
    files = []
    for name, row in entries.items():
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or any(p in ('.', '..') for p in relative.parts):
            raise ValueError('invalid relative inventory path')
        if not isinstance(row, dict) or set(row) != {'sha256', 'size'}:
            raise ValueError('invalid file inventory entry')
        files.append(_file(root / relative, row['sha256'], row['size']))
    return files


def _source_root(profile: dict[str, Any]) -> Path:
    if profile['source_root'] != '$FROZEN_SOURCE':
        raise ValueError('source root must use frozen source token')
    return _canonical(Path(__file__).resolve().parents[2])


def _import_directories(profile: dict[str, Any]) -> list[Path]:
    """Resolve only the frozen source root and configured packed-runtime import roots."""
    source = _source_root(profile)
    runtime_root = _canonical(Path(profile['runtime_root']))
    prefix = _canonical(Path(profile['runtime_prefix']))
    if not source.is_dir() or not runtime_root.is_dir() or not prefix.is_dir() or not prefix.is_relative_to(runtime_root):
        raise ValueError('frozen source and packed runtime directories required')
    paths = [_canonical(Path(p)) for p in profile['runtime_import_paths']]
    if not paths or any(not p.is_dir() or not p.is_relative_to(runtime_root) for p in paths):
        raise ValueError('runtime import paths must stay in packed runtime')
    return [source, *paths]


def _read_directories(files: list[Path], execute: list[Path], system: list[Path],
                      import_directories: list[Path]) -> set[Path]:
    """READ_DIR grants only; file content grants remain separately enumerated."""
    return {p.parent for p in files + execute + system} | set(import_directories)


def inventory(config: dict[str, Any]) -> tuple[list[Path], list[Path], list[Path]]:
    """Verify the exact reviewed read and execute file inventory."""
    if config.get('schema') != 1:
        raise ValueError('unknown policy schema')
    profile = config['profiles'][os.uname().nodename.split('.')[0]]
    runtime = _manifest(Path(profile['runtime_root']), profile['runtime_files'])
    source = _manifest(_source_root(profile), profile['source_files'])
    for name in profile['source_files']:
        path = Path(name)
        if path.suffix != '.py' or any(part in {'docs', 'tests', 'test', 'data', 'fixtures'} for part in path.parts):
            raise ValueError('source grant is not reviewed decoder code')
    files = runtime + source
    execute = []
    for row in profile['executables']:
        execute.append(_file(Path(row['path']), row['sha256'], row['size']))
    for row in profile['system_files']:
        files.append(_file(Path(row['path']), row['sha256'], row['size']))
    if not execute or not profile['system_files']:
        raise ValueError('executable and system library inventories required')
    # System leaves are exact typed nodes, never recursive directory grants.
    system = []
    for row in profile['system_leaves']:
        raw_path = Path(row['path'])
        if str(raw_path) in ('/proc/self/maps', '/proc/self/status'):
            path = raw_path.resolve(strict=True)
        else:
            path = _canonical(raw_path)
        info = path.stat()
        kind = row['kind']
        if kind not in ('regular', 'character') or not (stat.S_ISREG(info.st_mode) if kind == 'regular' else stat.S_ISCHR(info.st_mode)):
            raise ValueError('invalid system leaf type')
        approved = {str(Path(name).resolve()) for name in leaves.APPROVED_LEAVES + leaves.SYSTEM_GRANTS if Path(name).exists()}
        if str(path) not in approved or path.is_dir():
            raise ValueError('unapproved system leaf')
        system.append(path)
    return files, execute, system


def _protected(paths: dict[str, list[Path]]) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    if set(paths) != set(CATEGORIES):
        raise ValueError('all five protected categories required')
    evidence: dict[str, list[dict[str, Any]]] = {}
    flat = []
    for category in CATEGORIES:
        if not paths[category]:
            raise ValueError('empty protected category')
        evidence[category] = []
        for path in paths[category]:
            # Preserve alias spellings for the post-restriction open, but bind the inode.
            if not path.is_absolute() or not path.is_file():
                raise ValueError('protected target must be an existing regular file')
            info = path.stat()
            digest = sha(path)  # positive outside read of the actual target
            evidence[category].append({'path': str(path), 'resolved': str(path.resolve(strict=True)),
                                       'device': info.st_dev, 'inode': info.st_ino, 'sha256': digest})
            flat.append(path)
    return evidence, flat


def _context(context: dict[str, Any]) -> None:
    if set(context) != {'job_id', 'stage', 'execution_identity', 'gpu'}:
        raise ValueError('producer context incomplete')
    stage = Path(os.environ['PS_STAGE_DIR']).resolve(strict=True)
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    identity = context['execution_identity']
    if (context['job_id'] != stage.parent.name or context['stage'] != stage.name
            or not isinstance(identity, dict)
            or identity.get('source_sha256') != dispatch['code']['source_sha256']
            or identity.get('snapshot_sha256') != dispatch['code']['snapshot_sha256']
            or identity.get('environment_sha256') != dispatch['staged_environment']['sha256']):
        raise ValueError('producer dispatch differs')
    gpu = context['gpu']
    if gpu != {'kind': 'cpu'}:
        actual = dispatch['environment']['gpu']
        if (not isinstance(gpu, dict) or gpu.get('uuid') != actual.get('uuid')
                or gpu.get('name') != actual.get('name')
                or gpu.get('uuid') != os.environ.get('CUDA_VISIBLE_DEVICES')):
            raise ValueError('producer GPU differs')


def _restrict(files: list[Path], execute: list[Path], system: list[Path], output: Path,
              import_directories: list[Path]) -> tuple[int, int]:
    lib = ctypes.CDLL(None, use_errno=True)
    abi = int(lib.syscall(444, 0, 0, 1))
    mask = leaves.handled_rights(abi)
    fd = int(lib.syscall(444, ctypes.byref(ctypes.c_uint64(mask)), ctypes.sizeof(ctypes.c_uint64), 0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    # READ_DIR is nonrecursive with respect to file reads; do not grant parent file reads.
    dirs = _read_directories(files, execute, system, import_directories)
    grants = [(p, 4) for p in files] + [(p, 5) for p in execute] + [(p, 8) for p in dirs]
    for path in system:
        bits = 4
        if stat.S_ISCHR(path.stat().st_mode) and (str(path).startswith('/dev/nvidia') or path == Path('/dev/null')):
            bits |= 2 | (leaves.IOCTL_DEV if abi >= 5 else 0)
        grants.append((path, bits))
    grants.append((Path('/proc'), 12))  # Verified private procfs only.
    grants.append((output, leaves.allowed_rights(output, output, abi)))
    try:
        for path, rights in grants:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(rights, descriptor)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), f'landlock rule {path}')
            finally:
                os.close(descriptor)
        if lib.prctl(38, 1, 0, 0, 0) or lib.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return abi, mask


def _namespaces(pid: str) -> dict[str, str]:
    return {name: os.readlink(f'/proc/{pid}/ns/{name}') for name in ('user', 'pid', 'mnt', 'ipc')}


def _proc_identity(pid: int) -> dict[str, Any]:
    base = Path('/proc') / str(pid)
    fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
    birth, host_pgid, host_ppid = fields[19], int(fields[2]), int(fields[1])
    status = (base / 'status').read_text()
    ns_pid = next(line for line in status.splitlines() if line.startswith('NSpid:')).split()[1:]
    return {'pid': pid, 'birth': birth, 'host_pgid': host_pgid, 'host_ppid': host_ppid, 'namespace_pids': [int(x) for x in ns_pid],
            'namespaces': _namespaces(str(pid))}


def _effective_proc_mount(lines: list[str], mount_id: int, device: tuple[int, int]) -> str:
    matches = []
    for line in lines:
        left, sep, right = line.partition(' - ')
        fields, tail = left.split(), right.split()
        if (sep and len(fields) >= 6 and len(tail) >= 1 and fields[0] == str(mount_id)
                and fields[2] == f'{device[0]}:{device[1]}' and fields[3:5] == ['/', '/proc']
                and tail[0] == 'proc'):
            matches.append(line)
    if len(matches) != 1:
        raise ValueError('effective /proc mount identity differs')
    return matches[0]


def _private_proc(parent: dict[str, str], parent_host_pid: int) -> dict[str, Any]:
    current = _namespaces('self')
    if any(current[name] == parent[name] for name in current) or os.getpid() != 1 or parent_host_pid <= 1:
        raise ValueError('receiver must run as PID 1 in distinct private namespaces')
    descriptor = os.open('/proc', os.O_PATH | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        fdinfo = Path(f'/proc/self/fdinfo/{descriptor}').read_text()
        mount_id = int(next(line for line in fdinfo.splitlines() if line.startswith('mnt_id:')).split()[1])
        mount = _effective_proc_mount(Path('/proc/self/mountinfo').read_text().splitlines(), mount_id,
                                      (os.major(info.st_dev), os.minor(info.st_dev)))
    finally:
        os.close(descriptor)
    if sorted(p for p in os.listdir('/proc') if p.isdecimal()) != ['1']:
        raise ValueError('private procfs contains another process')
    if Path('/proc', str(parent_host_pid)).exists():
        raise ValueError('outer producer host PID is visible in private procfs')
    return {'namespaces': current, 'parent_namespaces': parent, 'private_proc': True,
            'mountinfo': mount, 'mount_id': mount_id, 'initial_pids': [1], 'outer_pid_hidden': parent_host_pid}


def _translate_child(namespace_pid: int, parent_host_pid: int, parent_namespace: str) -> dict[str, Any]:
    """Translate a Popen PID through host-mounted proc using parent/namespace identity."""
    children = Path(f'/proc/{parent_host_pid}/task/{parent_host_pid}/children').read_text().split()
    matches = []
    for value in children:
        try:
            candidate = _proc_identity(int(value))
        except (OSError, StopIteration, ValueError):
            continue
        if (candidate['namespace_pids'][-1:] == [namespace_pid]
                and candidate['namespaces']['pid'] == parent_namespace):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError('cannot uniquely translate child PID in parent namespace')
    return matches[0]


def _observe_native(work: Path, worker_pid: int, worker_ns: dict[str, str], nonce: str,
                    campaign_pgid: int, stop: threading.Event) -> None:
    handle = work / 'native-observer.json'
    children_path = Path(f'/proc/{worker_pid}/task/{worker_pid}/children')
    while not stop.wait(0.01):
        try:
            children = children_path.read_text().split()
            if len(children) != 1:
                continue
            identity = _proc_identity(int(children[0]))
            if (identity['namespaces'] != worker_ns or identity['namespace_pids'][-1] <= 1
                    or identity['host_pgid'] != campaign_pgid or identity['host_ppid'] != worker_pid):
                continue
            identity['nonce'] = nonce
            # Atomic publication prevents the restricted worker reading a partial record.
            temp = work / 'native-observer.tmp'
            temp.write_text(json.dumps(identity, sort_keys=True))
            temp.replace(handle)
            return
        except (OSError, StopIteration, ValueError):
            continue


def _birth() -> str:
    return Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]


def _denials(evidence: dict[str, list[dict[str, Any]]]) -> None:
    for rows in evidence.values():
        for row in rows:
            info = os.stat(row['path'])
            if (info.st_dev, info.st_ino) != (row['device'], row['inode']):
                raise ValueError('protected target identity changed before denial probe')
            try:
                with open(row['path'], 'rb') as f:
                    f.read(1)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    raise RuntimeError('denial inconclusive') from exc
                row['errno'] = exc.errno
            else:
                raise RuntimeError('protected file remains readable')


def verify_receipt(receipt: dict[str, Any], *, expected_context: dict[str, Any], expected_config_sha256: str) -> dict[str, Any]:
    proof = receipt.get('isolation')
    if (receipt.get('scope') != SCOPE or not isinstance(proof, dict)
            or proof.get('context') != expected_context or proof.get('config_sha256') != expected_config_sha256
            or proof.get('landlock_abi', 0) < 3 or proof.get('handled_mask') != leaves.handled_rights(proof['landlock_abi'])
            or proof.get('threads') != 1 or proof.get('extra_fds') is not False
            or proof.get('pid') != proof.get('postexec_pid') or proof.get('birth') != proof.get('postexec_birth')
            or not isinstance(proof.get('decoder'), dict) or not proof['decoder'].get('pid')
            or proof['decoder'].get('returncode') != 0):
        raise ValueError('isolated receiver evidence missing or inconsistent')
    namespace = proof.get('namespaces')
    observer = proof.get('worker_observer')
    campaign = proof.get('campaign_host')
    launcher = proof.get('launcher_observer')
    if (not isinstance(campaign, dict) or not isinstance(launcher, dict)
            or type(campaign.get('campaign_pgid')) is not int or campaign['campaign_pgid'] <= 0
            or campaign.get('host_pgid') != campaign['campaign_pgid']
            or type(campaign.get('host_pid')) is not int or campaign['host_pid'] <= 0
            or type(campaign.get('start_ticks')) is not int or campaign['start_ticks'] <= 0
            or launcher.get('host_pgid') != campaign['campaign_pgid']
            or launcher.get('host_ppid') != campaign['host_pid']
            or type(launcher.get('pid')) is not int or launcher['pid'] <= 0
            or not launcher.get('birth') or not launcher.get('namespace_pids')):
        raise ValueError('campaign or launcher host identity missing')
    if (not isinstance(namespace, dict) or namespace.get('private_proc') is not True
            or namespace.get('initial_pids') != [1] or proof['pid'] != 1 or not isinstance(observer, dict)
            or type(namespace.get('outer_pid_hidden')) is not int or namespace['outer_pid_hidden'] <= 1
            or not isinstance(namespace.get('mountinfo'), str) or ' - proc ' not in namespace['mountinfo']
            or observer.get('namespace_pids', [])[-1:] != [1] or type(observer.get('pid')) is not int
            or observer.get('birth') != proof['birth'] or observer.get('namespaces') != namespace.get('namespaces')
            or observer.get('host_pgid') != campaign['campaign_pgid']
            or observer.get('host_ppid') != launcher['pid']
            or observer.get('launcher_observer') != launcher
            or namespace.get('outer_pid_hidden') != campaign['host_pid']
            or any(namespace['namespaces'].get(k) == namespace['parent_namespaces'].get(k)
                   for k in ('user', 'pid', 'mnt', 'ipc'))):
        raise ValueError('private receiver namespace proof missing')
    decoder = proof['decoder']
    argv = decoder.get('argv')
    if (type(decoder.get('pid')) is not int or decoder['pid'] <= 1
            or type(decoder.get('started_monotonic_ns')) is not int or decoder['started_monotonic_ns'] <= 0
            or not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv)
            or decoder.get('argv_sha256') != _object_sha(argv)):
        raise ValueError('native child execution identity missing')
    for name in ('profile_sha256', 'actual_policy_sha256', 'runtime_inventory_sha256'):
        value = proof.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('policy inventory digest missing')
    if not isinstance(proof.get('actual_policy'), dict) or _object_sha(proof['actual_policy']) != proof['actual_policy_sha256']:
        raise ValueError('actual policy digest differs')
    policy = proof['actual_policy']
    nonce = proof.get('nonce')
    if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
        raise ValueError('native observer nonce missing')
    native_observer = decoder.get('outer_observation')
    if (not isinstance(native_observer, dict) or native_observer.get('namespaces') != namespace['namespaces']
            or native_observer.get('namespace_pids', [])[-1:] != [decoder['pid']]
            or type(native_observer.get('pid')) is not int or not native_observer.get('birth')
            or native_observer.get('nonce') != proof.get('nonce')
            or native_observer.get('host_pgid') != campaign['campaign_pgid']
            or native_observer.get('host_ppid') != observer['pid']):
        raise ValueError('outer native decoder observation missing')
    preexec = decoder.get('preexec')
    if (not isinstance(preexec, dict) or preexec.get('pid') != decoder['pid']
            or preexec.get('birth') != native_observer['birth'] or preexec.get('observer') != native_observer
            or preexec.get('argv_sha256') != decoder['argv_sha256'] or preexec.get('closed_extra_fds') is not True):
        raise ValueError('native pre-exec barrier evidence missing')
    if (policy.get('private_proc_read_only') is not True or policy.get('namespaces') != namespace
            or decoder.get('executable_resolved') not in policy.get('execute_files', [])
            or policy.get('context') != expected_context or policy.get('pid') != proof['pid']
            or policy.get('birth') != proof['birth'] or policy.get('handled_mask') != proof['handled_mask']
            or policy.get('campaign_host') != campaign or policy.get('launcher_observer') != launcher):
        raise ValueError('actual policy process or context differs')
    denied = proof.get('denials')
    if not isinstance(denied, dict) or set(denied) != set(CATEGORIES):
        raise ValueError('denial categories incomplete')
    for rows in denied.values():
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) or row.get('errno') not in (errno.EACCES, errno.EPERM)
            or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
            or not isinstance(row.get('path'), str) or not row.get('inode') for row in rows
        ):
            raise ValueError('post-exec denial evidence incomplete')
    if policy.get('denials') != {key: [{k: v for k, v in row.items() if k != 'errno'} for row in rows]
                                  for key, rows in denied.items()}:
        raise ValueError('actual denial targets differ from policy')
    for key in ('wire_sha256', 'resource_map_sha256', 'native_sha256', 'decoded_yuv_sha256', 'rendered_rgb_sha256'):
        if not isinstance(receipt.get(key), str) or len(receipt[key]) != 64:
            raise ValueError('receiver output hash missing')
    return proof


def _launch_receiver(command: list[str], env: dict[str, str]) -> subprocess.Popen[Any]:
    """Preserve the campaign process group; isolation comes from namespaces."""
    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, close_fds=True)


def _stop_launcher(process: subprocess.Popen[Any]) -> None:
    """Kill only our Popen child; unshare kills its private PID namespace."""
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def replay(wire_path: Path, map_path: Path, work: Path, *, config_path: Path, config_sha256: str,
           protected_paths: dict[str, list[Path]], producer_context: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one CPU required')
    _context(producer_context)
    scratch_value = os.environ.get('PS_SCRATCH_DIR')
    if not scratch_value or not work.resolve().is_relative_to(Path(scratch_value).resolve(strict=True)):
        raise ValueError('receiver output must be inside fleet scratch')
    if sha(config_path) != config_sha256:
        raise ValueError('policy configuration digest differs')
    config = json.loads(config_path.read_text())
    if config['execution_identity'] != producer_context['execution_identity']:
        raise ValueError('policy source identity differs')
    files, execute, system = inventory(config)
    from experiments.tennis.k5d_worker import resource
    from src.pointstream import ball_stream
    stream = ball_stream.unpack(wire_path.read_bytes())
    installed = resource(stream, json.loads(map_path.read_text()))
    for key in (('decoder',) if stream.codec == 'svt-av1-ivf' else ('image_ckpt', 'video_ckpt')):
        target = _canonical(Path(installed[key]))
        if target not in files + execute:
            raise ValueError('installed decoder resource is absent from exact inventory')
        if key == 'decoder' and target not in execute:
            raise ValueError('decoder executable is not granted execute')
    evidence, protected = _protected(protected_paths)
    denied_inodes = {(p.stat().st_dev, p.stat().st_ino) for p in protected}
    if any((p.stat().st_dev, p.stat().st_ino) in denied_inodes for p in files + execute + system + [wire_path, map_path]):
        raise ValueError('protected target aliases granted file')
    work.mkdir(mode=0o700, parents=False, exist_ok=False)
    for source, name in ((wire_path, 'packet.psbl'), (map_path, 'resources.json')):
        shutil.copyfile(source, work / name)
    from experiments.tennis.k5d_scoped_namespace import host_group_identity
    campaign = host_group_identity(Path(os.environ['PS_STAGE_DIR']).parent / 'run' / 'status.json')
    launch = {'campaign_host': campaign, 'config': config, 'config_sha256': config_sha256, 'denials': evidence, 'context': producer_context,
              'files': [str(p) for p in files], 'execute': [str(p) for p in execute], 'system': [str(p) for p in system],
              'wire_sha256': sha(wire_path), 'map_sha256': sha(map_path),
              'parent_namespaces': _namespaces('self'), 'outer_pid': int(os.readlink('/proc/self')),
              'nonce': secrets.token_hex(16)}
    (work / 'launch.json').write_text(json.dumps(launch, sort_keys=True))
    profile = config['profiles'][os.uname().nodename.split('.')[0]]
    python = _canonical(Path(profile['python']))
    if python not in execute:
        raise ValueError('interpreter not in executable inventory')
    import_directories = _import_directories(profile)
    env = {'PATH': '/nonexistent', 'PYTHONPATH': os.pathsep.join(str(p) for p in import_directories), 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PS_SCRATCH_DIR': str(work),
           'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work), 'XDG_CACHE_HOME': str(work)}
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    unshare_row = profile['unshare']
    unshare = _file(Path(unshare_row['path']), unshare_row['sha256'], unshare_row['size'])
    command = [str(unshare), '--user', '--map-root-user', '--pid', '--mount', '--ipc', '--fork',
               '--kill-child=SIGKILL', '--mount-proc', str(python), '-S', '-m', 'experiments.tennis.k5d_isolated_receiver',
               'worker', str(work)]
    stop = threading.Event()
    monitor: threading.Thread | None = None
    with _launch_receiver(command, env) as process:
        try:
            if process.stdout is None or not select.select([process.stdout], [], [], 120)[0]:
                raise RuntimeError('private receiver did not start')
            if process.stdout.readline() != b'READY\n':
                raise RuntimeError('private receiver startup failed')
            launcher = _translate_child(process.pid, launch['outer_pid'], launch['parent_namespaces']['pid'])
            if launcher['host_pgid'] != campaign['campaign_pgid'] or launcher['host_ppid'] != campaign['host_pid']:
                raise ValueError('launcher left campaign process group')
            children = Path(f"/proc/{launcher['pid']}/task/{launcher['pid']}/children").read_text().split()
            if len(children) != 1:
                raise ValueError('private receiver launcher has unexpected children')
            observed = _proc_identity(int(children[0]))
            if (observed['host_pgid'] != campaign['campaign_pgid'] or observed['host_ppid'] != launcher['pid']
                    or observed['namespace_pids'][-1] != 1
                    or any(observed['namespaces'][k] == launch['parent_namespaces'][k]
                           for k in launch['parent_namespaces'])):
                raise ValueError('outer observer did not identify isolated PID 1')
            observed['launcher_observer'] = launcher
            (work / 'worker-observer.json').write_text(json.dumps(observed, sort_keys=True))
            monitor = threading.Thread(target=_observe_native, args=(work, observed['pid'], observed['namespaces'],
                                                                       launch['nonce'], campaign['campaign_pgid'], stop))
            monitor.start()
            _, stderr = process.communicate(timeout=3600)
        except BaseException:
            _stop_launcher(process)
            raise
        finally:
            stop.set()
            if monitor is not None:
                monitor.join(timeout=10)
        if process.returncode:
            raise RuntimeError(f'isolated receiver failed: {process.returncode}: {stderr[-2000:]!r}')
    receipt = json.loads((work / 'result' / 'receiver.json').read_text())
    proof = verify_receipt(receipt, expected_context=producer_context, expected_config_sha256=config_sha256)
    if (proof['profile_sha256'] != _object_sha(profile)
            or proof['runtime_inventory_sha256'] != _object_sha(profile['runtime_files'])):
        raise ValueError('policy inventory identity differs')
    native_observation = json.loads((work / 'native-observer.json').read_text())
    if (proof['pid'] != 1 or proof['birth'] != observed['birth']
            or proof['namespaces']['namespaces'] != observed['namespaces']
            or proof['decoder'].get('outer_observation') != native_observation
            or native_observation['namespace_pids'][-1] != proof['decoder']['pid']):
        raise ValueError('outer and receiver lineage observations differ')
    if (receipt['wire_sha256'] != launch['wire_sha256'] or receipt['resource_map_sha256'] != launch['map_sha256']
            or receipt['decoded_yuv_sha256'] != sha(work / 'result' / 'native.yuv')
            or receipt['rendered_rgb_sha256'] != sha(work / 'result' / 'rendered.rgb')):
        raise ValueError('actual output or input identity differs')
    for path in (work / 'result').iterdir():
        path.rename(work / path.name)
    (work / 'result').rmdir()
    return receipt


def _worker(work: Path) -> None:
    if len(os.listdir('/proc/self/task')) != 1 or any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('worker is not single-threaded or inherited descriptors')
    launch = json.loads((work / 'launch.json').read_text())
    namespace_proof = _private_proc(launch['parent_namespaces'], launch['outer_pid'])
    print('READY', flush=True)
    observer_path = work / 'worker-observer.json'
    deadline = time.monotonic() + 30
    while not observer_path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('outer worker observation missing')
        time.sleep(0.01)
    observer = json.loads(observer_path.read_text())
    if (observer.get('host_pgid') != launch['campaign_host']['campaign_pgid']
            or observer.get('launcher_observer', {}).get('host_pgid') != launch['campaign_host']['campaign_pgid']
            or observer.get('birth') != _birth() or observer.get('namespaces') != namespace_proof['namespaces']
            or observer.get('namespace_pids', [])[-1:] != [1]):
        raise ValueError('outer worker observation differs')
    files, execute, system = inventory(launch['config'])
    if [str(p) for p in files] != launch['files'] or [str(p) for p in execute] != launch['execute']:
        raise ValueError('inventory changed before restriction')
    files += [work / 'packet.psbl', work / 'resources.json']
    pid, birth = os.getpid(), _birth()
    profile = launch['config']['profiles'][os.uname().nodename.split('.')[0]]
    import_directories = _import_directories(profile)
    actual_policy = {'read_files': [str(p) for p in files], 'execute_files': [str(p) for p in execute],
                     'system_leaves': [str(p) for p in system], 'writable_output': str(work),
                     'read_directories': sorted(str(p) for p in _read_directories(files, execute, system, import_directories)),
                     'private_proc_read_only': True, 'namespaces': namespace_proof,
                     'campaign_host': launch['campaign_host'], 'launcher_observer': observer['launcher_observer'],
                     'denials': launch['denials'], 'context': launch['context'], 'pid': pid, 'birth': birth}
    # The denial records are later augmented in place, so retain the pre-restriction inventory separately.
    actual_policy = json.loads(json.dumps(actual_policy))
    if len(os.listdir('/proc/self/task')) != 1 or any(int(fd) > 2 and Path('/proc/self/fd', fd).exists()
                                                        for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('threads or descriptors appeared before restriction')
    abi, mask = _restrict(files, execute, system, work, import_directories)
    actual_policy['handled_mask'] = mask
    _denials(launch['denials'])
    if sha(work / 'packet.psbl') != launch['wire_sha256'] or sha(work / 'resources.json') != launch['map_sha256']:
        raise ValueError('positive control identity differs')
    profile = launch['config']['profiles'][os.uname().nodename.split('.')[0]]
    sys.prefix = str(_canonical(Path(profile['runtime_prefix'])))
    from experiments.tennis import k5d_worker
    proof = {'context': launch['context'], 'config_sha256': launch['config_sha256'], 'landlock_abi': abi,
             'handled_mask': mask, 'threads': 1, 'extra_fds': False, 'pid': pid, 'postexec_pid': os.getpid(),
             'birth': birth, 'postexec_birth': birth, 'denials': launch['denials'],
             'namespaces': namespace_proof, 'worker_observer': observer, 'nonce': launch['nonce'],
             'campaign_host': launch['campaign_host'], 'launcher_observer': observer['launcher_observer'],
             'actual_policy': actual_policy, 'actual_policy_sha256': _object_sha(actual_policy),
             'profile_sha256': _object_sha(launch['config']['profiles'][os.uname().nodename.split('.')[0]]),
             'runtime_inventory_sha256': _object_sha(launch['config']['profiles'][os.uname().nodename.split('.')[0]]['runtime_files'])}
    k5d_worker.decode(work / 'packet.psbl', work / 'resources.json', work / 'result', isolation=proof)


def _native_wrapper(payload_path: Path) -> None:
    """Wait for an outer birth observation before replacing this process with the codec."""
    payload = json.loads(payload_path.read_text())
    observer_path = payload_path.parent / 'native-observer.json'
    deadline = time.monotonic() + 30
    while not observer_path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('native exec observer did not acknowledge')
        time.sleep(0.01)
    observer = json.loads(observer_path.read_text())
    birth = _birth()
    if (type(payload.get('campaign_pgid')) is not int or payload['campaign_pgid'] <= 0
            or observer.get('nonce') != payload['nonce']
            or observer.get('namespace_pids', [])[-1:] != [os.getpid()] or observer.get('birth') != birth
            or observer.get('namespaces') != _namespaces('self')
            or observer.get('host_pgid') != payload.get('campaign_pgid')):
        raise ValueError('native pre-exec observer differs')
    command = payload['argv']
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('native argv malformed')
    preexec = {'pid': os.getpid(), 'birth': birth, 'argv_sha256': _object_sha(command),
               'observer': observer, 'closed_extra_fds': True}
    for fd in os.listdir('/proc/self/fd'):
        if int(fd) > 2:
            try:
                os.close(int(fd))
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    (payload_path.parent / 'native-preexec.json').write_text(json.dumps(preexec, sort_keys=True))
    os.execve(command[0], command, payload['env'])


def cpu_oracle(wire_path: Path, map_path: Path, work: Path, *, wire_sha256: str, map_sha256: str,
               expected_yuv_sha256: str, expected_rgb_sha256: str, config_path: Path,
               config_sha256: str, protected_paths: dict[str, list[Path]],
               producer_context: dict[str, Any]) -> dict[str, Any]:
    """Replay an authenticated stored conventional packet, checking both pixel hashes."""
    from src.pointstream import ball_stream
    if producer_context.get('gpu') != {'kind': 'cpu'}:
        raise ValueError('CPU oracle requires CPU producer context')
    if sha(wire_path) != wire_sha256 or sha(map_path) != map_sha256:
        raise ValueError('authenticated oracle input identity differs')
    if ball_stream.unpack(wire_path.read_bytes()).codec != 'svt-av1-ivf':
        raise ValueError('CPU oracle admits only stored conventional SVT packets')
    receipt = replay(wire_path, map_path, work, config_path=config_path, config_sha256=config_sha256,
                     protected_paths=protected_paths, producer_context=producer_context)
    if (receipt['decoded_yuv_sha256'] != expected_yuv_sha256
            or receipt['rendered_rgb_sha256'] != expected_rgb_sha256):
        raise ValueError('isolated CPU oracle pixels differ from authenticated replay')
    return receipt


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    child = commands.add_parser('worker')
    child.add_argument('work', type=Path)
    wrapper = commands.add_parser('native-wrapper')
    wrapper.add_argument('payload', type=Path)
    oracle = commands.add_parser('cpu-oracle')
    for name in ('wire', 'resources', 'work', 'config', 'protected', 'context'):
        oracle.add_argument('--' + name, type=Path, required=True)
    for name in ('wire-sha256', 'map-sha256', 'expected-yuv-sha256', 'expected-rgb-sha256', 'config-sha256'):
        oracle.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        _worker(args.work)
        return
    if args.command == 'native-wrapper':
        _native_wrapper(args.payload)
        return
    protected = {k: [Path(p) for p in v] for k, v in json.loads(args.protected.read_text()).items()}
    receipt = cpu_oracle(args.wire, args.resources, args.work, wire_sha256=args.wire_sha256,
                         map_sha256=args.map_sha256, expected_yuv_sha256=args.expected_yuv_sha256,
                         expected_rgb_sha256=args.expected_rgb_sha256, config_path=args.config,
                         config_sha256=args.config_sha256, protected_paths=protected,
                         producer_context=json.loads(args.context.read_text()))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    _main()
