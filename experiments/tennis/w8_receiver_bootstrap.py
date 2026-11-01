"""Standard-library entry for a source-hidden W8 generation worker.

Requires a separately launched user/network namespace. This is an operational
containment mechanism for reviewed code, not a hostile-code security boundary.
Actual GPU/runtime and deny probes require host qualification before use.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
from typing import Any


def _deny_processes(event: str, _args: tuple[object, ...]) -> None:
    if event in {'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.spawn', 'subprocess.Popen',
                 'os.system', 'os.exec', 'socket.connect', 'socket.bind'}:
        raise PermissionError(f'worker operation denied: {event}')


def _restrict(paths: list[Path]) -> int:
    """Permit reads only under explicit roots; no file receives execute rights."""
    lib = ctypes.CDLL(None, use_errno=True)
    abi = int(lib.syscall(444, 0, 0, 1))
    if abi < 1:
        raise RuntimeError('Landlock unavailable')
    handled = ctypes.c_uint64(1 | 4 | 8)  # EXECUTE, READ_FILE, READ_DIR
    fd = int(lib.syscall(444, ctypes.byref(handled), ctypes.sizeof(handled), 0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    try:
        for path in paths:
            if not path.is_absolute() or not path.exists():
                raise ValueError(f'missing or relative allow path: {path}')
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(4 | (8 if path.is_dir() else 0), parent)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), f'landlock rule: {path}')
            finally:
                os.close(parent)
        if lib.prctl(38, 1, 0, 0, 0) or lib.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return abi


def _check_denials(paths: list[Path]) -> dict[str, int]:
    result = {}
    for path in paths:
        try:
            with path.open('rb') as stream:
                stream.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f'probe did not establish denial: {path}') from exc
            result[str(path)] = exc.errno
        else:
            raise RuntimeError(f'protected path readable: {path}')
    return result


def flatten_denials(groups: dict[str, list[str]]) -> list[Path]:
    return [Path(path) for paths in groups.values() for path in paths]


def bootstrap(config: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one-CPU affinity required')
    for key in ('parent_user_namespace', 'parent_network_namespace'):
        if not isinstance(config.get(key), str):
            raise ValueError(f'missing {key}')
    namespaces = {name: os.readlink(f'/proc/self/ns/{name}') for name in ('user', 'net')}
    if namespaces['user'] == config['parent_user_namespace'] or namespaces['net'] == config['parent_network_namespace']:
        raise RuntimeError('distinct user and network namespaces required')
    if sorted(name for _, name in socket.if_nameindex()) != ['lo']:
        raise RuntimeError('network namespace must be loopback-only')
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('pre-isolation process must have one thread')
    fds = [fd for fd in os.listdir('/proc/self/fd') if int(fd) > 2 and Path('/proc/self/fd', fd).exists()]
    if fds:
        raise RuntimeError('unexpected inherited file descriptors')
    allowed = [Path(p).resolve(strict=True) for p in config['allow_paths']]
    denied = flatten_denials(config['denied_paths'])
    if not denied or not allowed:
        raise ValueError('allow paths and protected probes required')
    for path in denied:
        resolved = path.resolve(strict=True)
        if any(resolved == root or resolved.is_relative_to(root) for root in allowed):
            raise ValueError(f'allow path exposes protected input: {path}')
    cpu = config['cpu_seconds']
    if type(cpu) is not int or not 1 <= cpu <= 3600:
        raise ValueError('bounded CPU seconds required')
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    safe_env = config['environment']
    if not isinstance(safe_env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in safe_env.items()):
        raise ValueError('explicit environment required')
    permitted = {'CUDA_VISIBLE_DEVICES', 'PS_GPU_MEMORY_LIMIT_MIB', 'PS_SCRATCH_DIR', 'PS_STAGE_DIR',
                 'PS_STAGE', 'TMPDIR', 'TMP', 'TEMP', 'HOME', 'XDG_CACHE_HOME', 'TORCH_HOME',
                 'LD_LIBRARY_PATH'}
    if set(safe_env) - permitted:
        raise ValueError('unexpected environment key')
    os.environ.clear()
    os.environ.update(safe_env)
    os.environ.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    # Only generic runtime imports and synthetic constants are touched before isolation.
    import torch
    from experiments.tennis.w8_input_job import synthetic_torch_bootstrap

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before isolation')
    before = synthetic_torch_bootstrap(torch)
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('runtime warmup created threads before isolation')
    if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('runtime warmup left a file descriptor open')
    sys.addaudithook(_deny_processes)
    abi = _restrict(allowed)
    probes = _check_denials(denied)
    after = synthetic_torch_bootstrap(torch)
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError('post-isolation CPU bootstrap failed')
    return {'landlock_abi': abi, 'denials': probes, 'namespaces': namespaces,
            'network_interfaces': ['lo'], 'pre_sandbox_threads': 1, 'extra_fds': False,
            'cpu_bootstrap': {'before_landlock_sha256': before, 'after_landlock_sha256': after,
                              'cuda_initialized_before_isolation': False},
            'process_limit': 'audit hook operational control, not a hostile-code boundary'}


def load_config(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.stat().st_size > 2_000_000:
        raise ValueError('config too large')
    raw = path.read_bytes()
    if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('config size or identity mismatch')
    config = json.loads(raw)
    required = {'parent_user_namespace', 'parent_network_namespace', 'allow_paths', 'denied_paths',
                'cpu_seconds', 'environment', 'files', 'tensor_fingerprints', 'unit', 'output', 'worker_kind',
                'protected_receipt'}
    if not isinstance(config, dict) or set(config) not in (required, required | {'native_receipt'},
                                                         required | {'probe_steps'}):
        raise ValueError('unexpected configuration fields')
    if ('native_receipt' in config) != (config.get('worker_kind') == 'rgb'):
        raise ValueError('native receipt required only for RGB conversion')
    if config['worker_kind'] not in ('packet', 'generate', 'cost_probe', 'preflight', 'sam', 'rgb', 'compose'):
        raise ValueError('unknown worker kind')
    if config['worker_kind'] == 'cost_probe':
        if type(config.get('probe_steps')) is not int or config['probe_steps'] not in (1, 3, 6):
            raise ValueError('cost probe steps must be exactly 1, 3 or 6')
    elif 'probe_steps' in config:
        raise ValueError('probe steps allowed only for cost probe')
    if not isinstance(config['files'], dict) or not isinstance(config['tensor_fingerprints'], dict):
        raise ValueError('invalid asset or fingerprint manifest')
    if config['worker_kind'] != 'preflight' and len(config['tensor_fingerprints']) != 480:
        raise ValueError('480 tensor fingerprints required')
    if not isinstance(config['allow_paths'], list) or not config['allow_paths'] or not all(
            isinstance(p, str) for p in config['allow_paths']):
        raise ValueError('explicit allow paths required')
    denials = config['denied_paths']
    if (not isinstance(denials, dict) or set(denials) != {'source', 'prepared', 'fit', 'golden', 'masks', 'aliases'}
            or not all(isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths)
                       for paths in denials.values())):
        raise ValueError('all protected path and alias probes are required')
    for item in config['files'].values():
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'}
                or not isinstance(item['path'], str) or not isinstance(item['sha256'], str)
                or len(item['sha256']) != 64):
            raise ValueError('invalid asset entry')
    from experiments.tennis.w8_receiver_preflight import PROTECTED

    protected_receipt = config['protected_receipt']
    if not isinstance(protected_receipt, dict) or set(protected_receipt) != {'path', 'sha256'}:
        raise ValueError('protected receipt identity required')
    protected_raw = Path(protected_receipt['path']).read_bytes()
    if hashlib.sha256(protected_raw).hexdigest() != protected_receipt['sha256']:
        raise ValueError('protected receipt identity mismatch')
    verified = json.loads(protected_raw)['protected_inputs']
    identities = {}
    for group in ('source', 'prepared', 'fit', 'golden', 'masks'):
        expected_path, expected_sha = PROTECTED[group]
        if denials[group] != [expected_path]:
            raise ValueError('protected probe is not the fixed original input')
        st = Path(expected_path).stat()
        if verified.get(group) != {'path': expected_path, 'sha256': expected_sha, 'device': st.st_dev, 'inode': st.st_ino}:
            raise ValueError('protected probe differs from verified identity receipt')

        identities[group] = {(Path(path).stat().st_dev, Path(path).stat().st_ino) for path in denials[group]}
    seen: set[tuple[int, int]] = set()
    for group in ('source', 'prepared', 'fit', 'golden', 'masks'):
        if seen.intersection(identities[group]):
            raise ValueError('protected categories must name distinct actual files')
        seen.update(identities[group])
    protected_spellings = {str(Path(path).absolute()) for group in ('source', 'prepared', 'fit', 'golden', 'masks')
                           for path in denials[group]}
    if any((Path(path).stat().st_dev, Path(path).stat().st_ino) not in seen
           or str(Path(path).absolute()) in protected_spellings for path in denials['aliases']):
        raise ValueError('alias probes must be distinct paths to protected files')
    alias_ids = {(Path(path).stat().st_dev, Path(path).stat().st_ino) for path in denials['aliases']}
    if not all(identities[group] & alias_ids for group in identities):
        raise ValueError('each protected category needs an alias probe')
    roots = [Path(p).resolve(strict=True) for p in config['allow_paths']]
    if any(root.is_file() and (root.stat().st_dev, root.stat().st_ino) in seen for root in roots):
        raise ValueError('allowed file is a protected hardlink')
    forbidden_roots = {Path(p) for p in ('/', '/home', '/usr', '/etc', '/proc', '/sys', '/dev', '/opt')}
    if any(p in forbidden_roots or p.name in {'Datasets', 'Models'} for p in roots):
        raise ValueError('broad allow root prohibited')
    for item in config['files'].values():
        target = Path(item['path']).resolve(strict=True)
        if not target.is_file() or not any(target == root or target.is_relative_to(root) for root in roots):
            raise ValueError('asset not explicitly reachable')
        digest = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('asset hash mismatch')
    output = Path(config['output']).resolve(strict=True)
    if not output.is_dir() or any(output.iterdir()) or output not in roots:
        raise ValueError('explicit fresh output directory required')
    return config


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--cpu-seconds', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.cpu_seconds <= 3600:
        raise ValueError('invalid CPU limit')
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds, args.cpu_seconds))
    config = load_config(Path(args.config), args.config_sha256)
    if config['cpu_seconds'] != args.cpu_seconds:
        raise ValueError('CPU limit differs from frozen config')
    isolation = bootstrap(config)
    if config['worker_kind'] == 'preflight':
        if config['files']:
            raise ValueError('preflight must not read scientific inputs')
        import numpy
        import torch

        closure = {}
        for name, module in sorted(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if name.startswith('experiments.') and filename:
                path = Path(filename).resolve(strict=True)
                closure[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        receipt = {'isolation': isolation, 'executable': str(Path(sys.executable).resolve()),
                   'prefix': str(Path(sys.prefix).resolve()), 'torch': str(torch.__version__),
                   'numpy': str(numpy.__version__), 'cuda_initialized': torch.cuda.is_initialized(),
                   'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'source_modules': closure}
        (Path(config['output']) / 'preflight.json').write_text(json.dumps(receipt, sort_keys=True))
    elif config['worker_kind'] in ('packet', 'generate', 'cost_probe'):
        from experiments.tennis.w8_receiver_generate import run_isolated, run_packet_isolated
        if config['worker_kind'] == 'packet':
            run_packet_isolated(config, isolation)
        else:
            run_isolated(config, isolation)
    else:
        from experiments.tennis.w8_receiver_finish import (run_compose_isolated, run_conversion_isolated,
                                                            run_sam_isolated)
        {'sam': run_sam_isolated, 'rgb': run_conversion_isolated, 'compose': run_compose_isolated}[
            config['worker_kind']](config, isolation)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
