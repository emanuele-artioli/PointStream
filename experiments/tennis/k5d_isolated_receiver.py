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
import shutil
import signal
import stat
import subprocess
import sys
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
                or gpu.get('name') != actual.get('name') or gpu.get('capability') != actual.get('capability')
                or gpu.get('uuid') != os.environ.get('CUDA_VISIBLE_DEVICES')):
            raise ValueError('producer GPU differs')


def _restrict(files: list[Path], execute: list[Path], system: list[Path], output: Path) -> tuple[int, int]:
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
    dirs = {p.parent for p in files + execute + system}
    grants = [(p, 4) for p in files] + [(p, 5) for p in execute] + [(p, 8) for p in dirs]
    for path in system:
        bits = 4
        if stat.S_ISCHR(path.stat().st_mode) and (str(path).startswith('/dev/nvidia') or path == Path('/dev/null')):
            bits |= 2 | (leaves.IOCTL_DEV if abi >= 5 else 0)
        grants.append((path, bits))
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
    if (decoder.get('executable_resolved') not in policy.get('execute_files', [])
            or policy.get('context') != expected_context or policy.get('pid') != proof['pid']
            or policy.get('birth') != proof['birth'] or policy.get('handled_mask') != proof['handled_mask']):
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
    launch = {'config': config, 'config_sha256': config_sha256, 'denials': evidence, 'context': producer_context,
              'files': [str(p) for p in files], 'execute': [str(p) for p in execute], 'system': [str(p) for p in system],
              'wire_sha256': sha(wire_path), 'map_sha256': sha(map_path)}
    (work / 'launch.json').write_text(json.dumps(launch, sort_keys=True))
    profile = config['profiles'][os.uname().nodename.split('.')[0]]
    python = _canonical(Path(profile['python']))
    if python not in execute:
        raise ValueError('interpreter not in executable inventory')
    import_paths = [_canonical(Path(p)) for p in profile['runtime_import_paths']]
    runtime_root = Path(profile['runtime_root'])
    runtime_prefix = _canonical(Path(profile['runtime_prefix']))
    if not runtime_prefix.is_dir() or not runtime_prefix.is_relative_to(runtime_root):
        raise ValueError('runtime prefix must be within packed runtime')
    if not import_paths or any(not p.is_dir() or not p.is_relative_to(runtime_root) for p in import_paths):
        raise ValueError('runtime import paths must stay in packed runtime')
    env = {'PATH': '/nonexistent', 'PYTHONPATH': os.pathsep.join([str(_source_root(profile)), *(str(p) for p in import_paths)]), 'PYTHONNOUSERSITE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PS_SCRATCH_DIR': str(work),
           'TMPDIR': str(work), 'TMP': str(work), 'TEMP': str(work), 'XDG_CACHE_HOME': str(work)}
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    with subprocess.Popen([str(python), '-S', '-m', 'experiments.tennis.k5d_isolated_receiver', 'worker', str(work)], env=env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          close_fds=True, start_new_session=True) as process:
        try:
            observed_birth = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            _, stderr = process.communicate(timeout=3600)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f'isolated receiver failed: {process.returncode}: {stderr[-2000:]!r}')
    receipt = json.loads((work / 'result' / 'receiver.json').read_text())
    proof = verify_receipt(receipt, expected_context=producer_context, expected_config_sha256=config_sha256)
    if (proof['profile_sha256'] != _object_sha(profile)
            or proof['runtime_inventory_sha256'] != _object_sha(profile['runtime_files'])):
        raise ValueError('policy inventory identity differs')
    if proof['pid'] != process.pid or proof['birth'] != observed_birth:
        raise ValueError('launcher and worker process identity differ')
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
    files, execute, system = inventory(launch['config'])
    if [str(p) for p in files] != launch['files'] or [str(p) for p in execute] != launch['execute']:
        raise ValueError('inventory changed before restriction')
    files += [work / 'packet.psbl', work / 'resources.json']
    pid, birth = os.getpid(), _birth()
    actual_policy = {'read_files': [str(p) for p in files], 'execute_files': [str(p) for p in execute],
                     'system_leaves': [str(p) for p in system], 'writable_output': str(work),
                     'read_directories': sorted(str(p) for p in {p.parent for p in files + execute + system}),
                     'denials': launch['denials'], 'context': launch['context'], 'pid': pid, 'birth': birth}
    # The denial records are later augmented in place, so retain the pre-restriction inventory separately.
    actual_policy = json.loads(json.dumps(actual_policy))
    abi, mask = _restrict(files, execute, system, work)
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
             'actual_policy': actual_policy, 'actual_policy_sha256': _object_sha(actual_policy),
             'profile_sha256': _object_sha(launch['config']['profiles'][os.uname().nodename.split('.')[0]]),
             'runtime_inventory_sha256': _object_sha(launch['config']['profiles'][os.uname().nodename.split('.')[0]]['runtime_files'])}
    k5d_worker.decode(work / 'packet.psbl', work / 'resources.json', work / 'result', isolation=proof)


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
    oracle = commands.add_parser('cpu-oracle')
    for name in ('wire', 'resources', 'work', 'config', 'protected', 'context'):
        oracle.add_argument('--' + name, type=Path, required=True)
    for name in ('wire-sha256', 'map-sha256', 'expected-yuv-sha256', 'expected-rgb-sha256', 'config-sha256'):
        oracle.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        _worker(args.work)
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
