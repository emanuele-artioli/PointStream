"""Single-command, CPU-only native W8 decode in a separate namespace.

The launcher must create the user/network namespaces and enforce an external
wall deadline. Landlock and the audit hook are operational controls for this
reviewed worker, not a security boundary against hostile native code.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import socket
import struct
import subprocess
import sys
from typing import Any

FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
IVF_SHA = 'f9622b6af4cddea3d70226c0b22e12374f4c77aadf2120e54d5d21ba55bfbbb8'
EXPECTED_BYTES = 240 * 480 * 272 * 3 // 2
EXPECTED_YUV = '7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff'


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_argv(binary: Path, ivf: Path, output: Path) -> list[str]:
    return [str(binary), '-nostdin', '-v', 'error', '-threads', '1', '-c:v', 'libdav1d',
            '-i', str(ivf), '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-n', str(output)]


def validate_ivf_header(header: bytes) -> None:
    if len(header) != 32 or header[:4] != b'DKIF':
        raise ValueError('invalid IVF header')
    version, size, codec, width, height, rate, scale, frames, _ = struct.unpack('<HH4sHHIIII', header[4:])
    if (version, size, codec, width, height, rate, scale, frames) != (0, 32, b'AV01', 480, 272, 50, 1, 240):
        raise ValueError('unexpected IVF format')


def _audit_policy(expected: list[str]) -> Any:
    consumed = False
    def audit(event: str, args: tuple[Any, ...]) -> None:
        nonlocal consumed
        if event == 'subprocess.Popen':
            if consumed or str(args[0]) != expected[0] or list(args[1]) != expected:
                raise PermissionError('native command is not the sole permitted command')
            consumed = True
        elif event in {'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.spawn', 'os.system',
                       'os.exec', 'socket.connect', 'socket.bind'}:
            raise PermissionError(f'native operation denied: {event}')
    return audit


def _landlock(files: list[Path], dirs: list[Path], execute: list[Path], output: Path) -> int:
    """Grant exact file reads/executions and reads/writes only in fresh output."""
    execute_bit, write_file, read_file, read_dir = 1, 2, 4, 8
    remove_dir, remove_file, make_dir, make_reg = 16, 32, 128, 256
    handled = execute_bit | write_file | read_file | read_dir | remove_dir | remove_file | make_dir | make_reg
    libc = ctypes.CDLL(None, use_errno=True)
    abi = int(libc.syscall(444, 0, 0, 1))
    if abi < 1:
        raise RuntimeError('Landlock unavailable')
    fd = int(libc.syscall(444, ctypes.byref(ctypes.c_uint64(handled)), ctypes.sizeof(ctypes.c_uint64), 0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    try:
        grants = [(path, read_file) for path in files] + [(path, read_dir) for path in dirs]
        grants += [(path, execute_bit | read_file) for path in execute]
        grants += [(output, read_dir | read_file | write_file | make_reg)]
        for path, bits in grants:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(bits, descriptor)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), f'landlock rule {path}')
            finally:
                os.close(descriptor)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return abi


def run(config: dict[str, Any]) -> None:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one CPU required')
    namespaces = {name: os.readlink(f'/proc/self/ns/{name}') for name in ('user', 'net')}
    if (namespaces['user'] == config['parent_user_namespace']
            or namespaces['net'] == config['parent_network_namespace']
            or sorted(name for _, name in socket.if_nameindex()) != ['lo']):
        raise ValueError('distinct loopback-only user/network namespaces required')
    if len(os.listdir('/proc/self/task')) != 1 or any(int(n) > 2 for n in os.listdir('/proc/self/fd')
                                                       if Path('/proc/self/fd', n).exists()):
        raise ValueError('unexpected threads or inherited descriptors')
    output = Path(config['output']).resolve(strict=True)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('fresh empty output required')
    binary, ivf = (Path(config[key]).resolve(strict=True) for key in ('binary', 'ivf'))
    loader = Path(config['loader']['path']).resolve(strict=True)
    libraries = {Path(row['path']).resolve(strict=True): row['sha256'] for row in config['libraries']}
    if not libraries or len(libraries) != len(config['libraries']):
        raise ValueError('exact nonempty library list required')
    for path, expected in {binary: FFMPEG_SHA, ivf: IVF_SHA, loader: config['loader']['sha256'], **libraries}.items():
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f'native input hash: {path}')
    with ivf.open('rb') as stream:
        validate_ivf_header(stream.read(32))
    denied = [Path(x).resolve(strict=True) for x in config['denied_paths']]
    files = [ivf, *libraries]
    executables = [binary, loader]
    if not denied or any(p == output or p.is_relative_to(output) or p in files + executables for p in denied):
        raise ValueError('protected inputs overlap grants')
    parent_cpu, child_cpu, wall = (config[k] for k in ('parent_cpu_seconds', 'child_cpu_seconds', 'wall_seconds'))
    if any(type(n) is not int or not 1 <= n <= 600 for n in (parent_cpu, child_cpu, wall)):
        raise ValueError('bounded CPU and wall limits required')
    current = resource.getrlimit(resource.RLIMIT_CPU)
    if current[0] != parent_cpu or current[1] != max(parent_cpu, child_cpu):
        raise ValueError('worker CPU limits differ from frozen config')
    env = {'PATH': '/nonexistent', 'XDG_CACHE_HOME': str(output), 'TMPDIR': str(output), 'TMP': str(output),
           'TEMP': str(output), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'LD_LIBRARY_PATH': config['library_path']}
    os.environ.clear()
    os.environ.update(env)
    argv = decode_argv(binary, ivf, output / 'native.yuv')
    # Any directory grants permit listing only, not reads of arbitrary files.
    dirs = sorted({p.parent for p in files + executables}, key=str)
    abi = _landlock(files, dirs, executables, output)
    probes = {}
    for path in denied:
        try:
            with path.open('rb') as stream:
                stream.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f'denial probe inconclusive: {path}') from exc
            probes[str(path)] = exc.errno
        else:
            raise ValueError(f'protected input readable: {path}')
    sys.addaudithook(_audit_policy(argv))
    parent_pid = os.getpid()
    def limit_child() -> None:
        parent_death_signal(parent_pid)
        resource.setrlimit(resource.RLIMIT_CPU, (child_cpu, child_cpu))
    started = time.monotonic()
    with (output / 'native.stdout').open('xb') as stdout, (output / 'native.stderr').open('xb') as stderr:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                    preexec_fn=limit_child, env=env, close_fds=True)
        except OSError as exc:
            (output / 'native-execution.json').write_text(json.dumps(
                {'status': 'launch_failed', 'pid': None, 'pgid': os.getpgrp(),
                 'started_monotonic': started, 'error': str(exc)}, sort_keys=True))
            raise
        if proc.stdin is not None:
            proc.stdin.close()
        handle = {'pid': proc.pid, 'pgid': os.getpgrp(), 'started_monotonic': started}
        (output / 'native-handle.json').write_text(json.dumps(handle, sort_keys=True))
        status = 'completed'
        try:
            returncode = proc.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            proc.kill()
            returncode = proc.wait(timeout=10)
        (output / 'native-execution.json').write_text(json.dumps(
            {**handle, 'status': status, 'returncode': returncode, 'wall_seconds': time.monotonic()-started},
            sort_keys=True))
    target = output / 'native.yuv'
    if status != 'completed' or returncode or not target.is_file() or target.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f'native decode failed: {status}, {returncode}')
    if file_sha(target) != EXPECTED_YUV:
        raise ValueError('native decoded YUV does not match the original')
    receipt = {'version': 1, 'binary_sha256': FFMPEG_SHA, 'ivf_sha256': IVF_SHA,
               'loader_sha256': config['loader']['sha256'],
               'libraries': {str(path): sha for path, sha in libraries.items()},
               'yuv_sha256': file_sha(target), 'yuv_bytes': target.stat().st_size,
               'exit_code': returncode, 'decoder': 'libdav1d', 'argv': argv,
               'landlock_abi': abi, 'denials': probes, 'namespaces': namespaces,
               'parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
               'child_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2])}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True))


def parent_death_signal(expected_parent: int) -> None:
    """Ensure task-owned child dies even if the Python parent is CPU-killed."""
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG')
    if os.getppid() != expected_parent:
        raise RuntimeError('parent disappeared before death signal was set')


def launch(config: Path, digest: str, parent_cpu: int, worker_cpu: int,
           child_cpu: int, wall: int, control: Path) -> None:
    """Launch one isolated worker in an owned process group and retain receipts."""
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one CPU required')
    if any(type(x) is not int or not 1 <= x <= 600 for x in (parent_cpu, worker_cpu, child_cpu, wall)):
        raise ValueError('invalid native launch bound')
    resource.setrlimit(resource.RLIMIT_CPU, (parent_cpu, max(parent_cpu, worker_cpu, child_cpu)))
    raw = config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('native configuration changed')
    frozen = json.loads(raw)
    if (frozen['parent_cpu_seconds'] != worker_cpu or frozen['child_cpu_seconds'] != child_cpu
            or frozen['wall_seconds'] > wall):
        raise ValueError('native launch limits disagree with config')
    control.mkdir(mode=0o700, parents=False, exist_ok=False)
    argv = ['/usr/bin/unshare', '--user', '--map-root-user', '--net', sys.executable,
            '-m', 'experiments.tennis.w8_receiver_native_bootstrap', '--phase', 'worker',
            '--config', str(config), '--config-sha256', digest, '--parent-cpu', str(worker_cpu),
            '--child-cpu', str(child_cpu), '--supervisor-pid', str(os.getpid())]
    # The caller supplies a namespace-wrapped invocation; the worker checks it
    # is already in a distinct user/network namespace before opening inputs.
    env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    parent_pid = os.getpid()
    def worker_limits() -> None:
        parent_death_signal(parent_pid)
        resource.setrlimit(resource.RLIMIT_CPU, (worker_cpu, max(worker_cpu, child_cpu)))
    started = time.monotonic()
    with (control / 'stdout.log').open('xb') as stdout, (control / 'stderr.log').open('xb') as stderr:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=env, close_fds=True, start_new_session=True, preexec_fn=worker_limits)
        except OSError as exc:
            (control / 'execution.json').write_text(json.dumps(
                {'status': 'launch_failed', 'pid': None, 'pgid': None,
                 'started_monotonic': started, 'error': str(exc)}, sort_keys=True))
            raise
        handle = {'pid': proc.pid, 'pgid': proc.pid, 'started_monotonic': started, 'wall_seconds': wall,
                  'parent_cpu_limit': parent_cpu, 'worker_cpu_limit': worker_cpu, 'child_cpu_limit': child_cpu,
                  'config_sha256': digest}
        (control / 'handle.json').write_text(json.dumps(handle, sort_keys=True))
        status = 'completed'
        try:
            code = proc.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = proc.wait(timeout=10)
        result = {**handle, 'status': status, 'returncode': code, 'wall_elapsed': time.monotonic()-started,
                  'launcher_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
                  'waited_children_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2])}
        (control / 'execution.json').write_text(json.dumps(result, sort_keys=True))
        if status != 'completed' or code:
            raise RuntimeError(f'native worker failed: {status}, {code}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('worker', 'launch'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--parent-cpu', type=int, required=True)
    parser.add_argument('--child-cpu', type=int, required=True)
    parser.add_argument('--worker-cpu', type=int)
    parser.add_argument('--wall-seconds', type=int)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--supervisor-pid', type=int)
    args = parser.parse_args()
    if any(not 1 <= n <= 600 for n in (args.parent_cpu, args.child_cpu)):
        raise ValueError('invalid native process CPU limits')
    if args.phase == 'launch':
        if args.worker_cpu is None or args.wall_seconds is None or args.control is None:
            raise ValueError('native launch limits and receipt directory required')
        launch(args.config, args.config_sha256, args.parent_cpu, args.worker_cpu,
               args.child_cpu, args.wall_seconds, args.control)
        return
    if args.supervisor_pid is None or args.supervisor_pid <= 1:
        raise ValueError('native worker requires its owning supervisor PID')
    parent_death_signal(args.supervisor_pid)
    resource.setrlimit(resource.RLIMIT_CPU, (args.parent_cpu, max(args.parent_cpu, args.child_cpu)))
    data = args.config.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.config_sha256:
        raise ValueError('native configuration changed')
    config = json.loads(data)
    if config['parent_cpu_seconds'] != args.parent_cpu or config['child_cpu_seconds'] != args.child_cpu:
        raise ValueError('native worker limits disagree with config')
    run(config)


if __name__ == '__main__':
    main()
