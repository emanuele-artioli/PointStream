"""Fresh GVC receiver and synthetic CUDA probe under a reviewed Landlock policy.

Called as a direct script, copied with the bridge into a minimal task-owned
code directory. Config and all input hashes must be frozen by the supervisor.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import resource
import socket
import stat
import sys
from pathlib import Path
from typing import Any

BRIDGE_SHA = 'cf42d1a115d905ed5faf3fdb01a8c7aa66c728091a18880c8cbf818cbcfbfd61'
I_SHA = 'c8a840415cc028307131c66f68076b7e9f6fcc4db7fed423c62e8b202f83a22d'
P_SHA = '7a89bd342a1484b3da7dfc3f03834ff81e0e4180e02f8a286dec0d1025c99f4e'
ENTROPY_SHA = 'f0951974be2a1168f87a67f8d3683844874aa1902b2d135cde46c2e5b915262e'
DENIALS = {'source_archive', 'source_rgb', 'source_yuv', 'sender_reference', 'ground_truth', 'aliases'}
POLICY = 'gvcrt-w8-twelve-leaf-mutation-v1'
APPROVED_LEAVES = (
    '/etc/ld.so.cache', '/proc/cpuinfo', '/proc/devices',
    '/proc/driver/nvidia/capabilities/mig/config', '/proc/driver/nvidia/capabilities/mig/monitor',
    '/proc/driver/nvidia/params', '/proc/self/maps', '/proc/self/status', '/proc/stat',
    '/proc/sys/vm/mmap_min_addr', '/sys/devices/system/cpu/online',
    '/sys/devices/system/memory/block_size_bytes',
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reject_hidden_hardlinks(roots: list[Path], protected: set[tuple[int, int]]) -> None:
    """An allowed tree must not contain another hardlink to protected bytes."""
    for root in roots:
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory) / name
                stat = path.lstat()
                if not path.is_symlink() and (stat.st_dev, stat.st_ino) in protected:
                    raise ValueError(f'allowed tree contains a protected hardlink: {path}')


def reject_shared_system_filesystem(directories: list[Path], protected: set[tuple[int, int]]) -> None:
    devices = {device for device, _ in protected}
    if any(path.stat().st_dev in devices for path in directories):
        raise ValueError('system library root shares a filesystem with protected inputs')


def load_config(path: Path, expected: str, *, probe: bool = False) -> dict[str, Any]:
    if path.stat().st_size > 200_000 or sha(path) != expected:
        raise ValueError('receiver config identity mismatch')
    c = json.loads(path.read_text())
    required = {'runtime', 'author', 'bridge', 'entropy', 'i_model', 'p_model', 'wire', 'output',
                'device_paths', 'denied_paths', 'canary_dir', 'parent_user', 'parent_net', 'visible_gpu', 'cpu_seconds',
                'policy_profile', 'loader_identity'}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError('unexpected receiver config keys')
    if c['policy_profile'] != POLICY:
        raise ValueError('unexpected policy profile')
    if set(c['denied_paths']) != DENIALS or any(not v or not isinstance(v, list) for v in c['denied_paths'].values()):
        raise ValueError('all source, reference, labels and alias denial groups required')
    names = ('runtime', 'author', 'bridge', 'entropy', 'i_model', 'p_model', 'wire', 'output', 'canary_dir')
    paths = [c[k] for k in names] + c['device_paths'] + [p for group in c['denied_paths'].values() for p in group]
    if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
        raise ValueError('absolute receiver paths required')
    fixed = {'/usr/lib', '/lib', '/lib64', '/dev/null', '/dev/urandom', '/proc/driver/nvidia/version'}
    if any(p not in fixed and not (p.startswith('/dev/nvidia') and '..' not in Path(p).parts)
           for p in c['device_paths']):
        raise ValueError('unexpected device access path')
    if not c['visible_gpu'].startswith('GPU-') or ',' in c['visible_gpu']:
        raise ValueError('one physical GPU UUID required')
    if type(c['cpu_seconds']) is not int or not 1 <= c['cpu_seconds'] <= 600:
        raise ValueError('invalid CPU bound')
    expected_files = {'bridge': BRIDGE_SHA, 'entropy': ENTROPY_SHA, 'i_model': I_SHA, 'p_model': P_SHA}
    for name, expected_sha in expected_files.items():
        target = Path(c[name])
        if not target.is_file() or (not (probe and name in ('i_model', 'p_model')) and sha(target) != expected_sha):
            raise ValueError(f'{name} identity mismatch')
    runtime, author = Path(c['runtime']).resolve(strict=True), Path(c['author']).resolve(strict=True)
    wire, output = Path(c['wire']).resolve(strict=True), Path(c['output']).resolve(strict=True)
    if runtime.name != 'tree' or runtime.parent.name != '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661':
        raise ValueError('unexpected runtime')
    if not author.is_dir() or sha(author / '.gvcrt-source.json') != 'b211cbcc4280acad546a41283ec95c63c7ca95c8d0fd6fdb07af7120932cbe97':
        raise ValueError('author source marker mismatch')
    if not wire.is_dir() or not output.is_dir() or any(output.iterdir()):
        raise ValueError('source and fresh wire/output directories required')
    canary = Path(c['canary_dir']).resolve(strict=True)
    expected_canaries = {'write', 'truncate', 'remove', 'rename', 'link'}
    if (not canary.is_dir() or canary.parent != output.parent or canary == output
            or {p.name for p in canary.iterdir()} != expected_canaries
            or any(p.is_symlink() or not p.is_file() or p.read_bytes() != b'canary' for p in canary.iterdir())):
        raise ValueError('fresh task-owned mutation canaries required')
    if {p.name for p in wire.iterdir()} != {'stream.bin', 'manifest.json'}:
        raise ValueError('receiver wire directory must contain exactly two files')
    if any(p.is_symlink() or not p.is_file() or p.resolve().parent != wire for p in wire.iterdir()):
        raise ValueError('receiver wire files must be regular local files')
    roots = [runtime, author, wire, output, *(Path(c[k]).resolve(strict=True) for k in expected_files),
             *(Path(p).resolve(strict=True) for p in c['device_paths'])]
    denied = [Path(p) for paths in c['denied_paths'].values() for p in paths]
    ids: dict[str, set[tuple[int, int]]] = {}
    for name, paths in c['denied_paths'].items():
        ids[name] = set()
        for p in paths:
            target = Path(p).resolve(strict=True)
            if any(target == root or target.is_relative_to(root) for root in roots):
                raise ValueError(f'allowed path exposes denied input: {p}')
            st = target.stat()
            ids[name].add((st.st_dev, st.st_ino))
    spellings = {str(Path(p).absolute()) for name, paths in c['denied_paths'].items()
                 if name != 'aliases' for p in paths}
    if any(str(Path(p).absolute()) in spellings for p in c['denied_paths']['aliases']):
        raise ValueError('aliases must use distinct spellings')
    categories = list(DENIALS - {'aliases'})
    if any(ids[a] & ids[b] for i, a in enumerate(categories) for b in categories[i + 1:]):
        raise ValueError('protected categories must have distinct identities')
    for name in categories:
        if not ids[name].issubset(ids['aliases']):
            raise ValueError(f'no actual alias for {name}')
    protected = set().union(*(v for k, v in ids.items() if k != 'aliases'))
    if ids['aliases'] != protected:
        raise ValueError('alias does not identify a protected input')
    reject_hidden_hardlinks([runtime, author, wire], protected)
    system_dirs = [Path(p).resolve(strict=True) for p in c['device_paths'] if Path(p).is_dir()]
    reject_shared_system_filesystem(system_dirs, protected)
    for root in [*roots, *wire.iterdir()]:
        if root.is_file() and (root.stat().st_dev, root.stat().st_ino) in protected:
            raise ValueError('allow file is a protected hardlink')
    if any(canary == root or canary.is_relative_to(root) for root in roots):
        raise ValueError('mutation canary must be outside all allowed roots')
    if len({p.resolve() for p in denied}) < len(DENIALS) - 1:
        raise ValueError('protected categories are not distinct')
    return c


EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = (1 << i for i in range(4))
REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR, MAKE_REG = (1 << i for i in range(4, 9))
MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = (1 << i for i in range(9, 13))
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15


def handled_rights(abi: int) -> int:
    if abi < 3:
        raise RuntimeError('Landlock ABI >=3 required to deny truncation')
    return (1 << 15) - 1 | (IOCTL_DEV if abi >= 5 else 0)


def allowed_rights(path: Path, output: Path, abi: int) -> int:
    if path == output:
        return READ_FILE | READ_DIR | WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | REFER | TRUNCATE
    if path.is_dir():
        return READ_FILE | READ_DIR
    rights = READ_FILE
    if str(path).startswith('/dev/nvidia') and stat.S_ISCHR(path.stat().st_mode):
        rights |= WRITE_FILE | (IOCTL_DEV if abi >= 5 else 0)
    elif path == Path('/dev/null') and stat.S_ISCHR(path.stat().st_mode):
        rights |= WRITE_FILE
    return rights


def restrict(paths: list[Path], output: Path) -> tuple[int, int]:
    lib = ctypes.CDLL(None, use_errno=True)
    abi = int(lib.syscall(444, 0, 0, 1))
    mask = handled_rights(abi)
    handled = ctypes.c_uint64(mask)
    fd = int(lib.syscall(444, ctypes.byref(handled), ctypes.sizeof(handled), 0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    try:
        for path in paths:
            opened = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(allowed_rights(path, output, abi), opened)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'landlock_add_rule')
            finally:
                os.close(opened)
        if lib.prctl(38, 1, 0, 0, 0) or lib.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return abi, mask


def mutation_probes(canary: Path, output: Path) -> dict[str, int]:
    """Only fresh task-owned canaries are touched, never source or model paths."""
    denied = {}
    def expect(name: str, action: Any) -> None:
        try:
            action()
        except OSError as exc:
            allowed = (errno.EACCES, errno.EPERM, errno.EXDEV) if name in ('rename', 'link') else (errno.EACCES, errno.EPERM)
            if exc.errno not in allowed:
                raise RuntimeError(f'mutation probe inconclusive: {name}') from exc
            assert exc.errno is not None
            denied[name] = exc.errno
        else:
            raise RuntimeError(f'mutation unexpectedly allowed: {name}')
    def opened(path: Path, flags: int) -> None:
        fd = os.open(path, flags, 0o600)
        os.close(fd)
    expect('write', lambda: opened(canary / 'write', os.O_WRONLY))
    expect('truncate', lambda: opened(canary / 'truncate', os.O_WRONLY | os.O_TRUNC))
    expect('create', lambda: opened(canary / 'new-file', os.O_WRONLY | os.O_CREAT | os.O_EXCL))
    expect('remove', lambda: os.unlink(canary / 'remove'))
    expect('mkdir', lambda: os.mkdir(canary / 'new-dir'))
    expect('rename', lambda: os.rename(canary / 'rename', output / 'unexpected-rename'))
    expect('link', lambda: os.link(canary / 'link', output / 'unexpected-link'))
    test = output / '.output-write-probe'
    test.write_bytes(b'output-ok')
    if test.read_bytes() != b'output-ok':
        raise RuntimeError('output leaf write/read failed')
    test.unlink()
    return denied


def identities(c: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for group, paths in c['denied_paths'].items():
        result[group] = []
        for path in paths:
            p = Path(path)
            stat = p.stat()
            result[group].append({'path': str(p), 'resolved': str(p.resolve(strict=True)),
                                  'device': stat.st_dev, 'inode': stat.st_ino})
    return result


def approved_leaves(c: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    protected = {(Path(p).stat().st_dev, Path(p).stat().st_ino)
                 for group in c['denied_paths'].values() for p in group}
    loader = c['loader_identity']
    if not isinstance(loader, dict) or set(loader) != {'device', 'inode', 'mode', 'uid'} or loader['uid'] != 0 or loader['mode'] & stat.S_IWOTH:
        raise ValueError('supervisor loader identity is unsafe')
    paths, receipt = [], []
    for name in APPROVED_LEAVES:
        path = Path(name).resolve(strict=True)
        info = path.stat()
        expected = name.replace('/proc/self/', f'/proc/{os.getpid()}/')
        if str(path) != expected or not stat.S_ISREG(info.st_mode):
            raise ValueError('approved system leaf missing or noncanonical')
        if (info.st_dev, info.st_ino) in protected:
            raise ValueError('system leaf aliases protected data')
        if name == '/etc/ld.so.cache' and (info.st_dev, info.st_ino, info.st_mode) != (loader['device'], loader['inode'], loader['mode']):
            raise ValueError('loader inode differs from supervisor identity')
        paths.append(path)
        receipt.append({'name': name, 'resolved': str(path), 'device': info.st_dev, 'inode': info.st_ino,
                        'mode': info.st_mode, 'uid': info.st_uid})
    return paths, receipt


def bootstrap(c: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux one-CPU affinity required')
    ns = {k: os.readlink(f'/proc/self/ns/{k}') for k in ('user', 'net')}
    if ns['user'] == c['parent_user'] or ns['net'] == c['parent_net']:
        raise RuntimeError('fresh user and network namespaces required')
    if sorted(name for _, name in socket.if_nameindex()) != ['lo']:
        raise RuntimeError('receiver network must be loopback-only')
    if len(os.listdir('/proc/self/task')) != 1 or any(
            int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('unexpected thread or inherited fd')
    resource.setrlimit(resource.RLIMIT_CPU, (c['cpu_seconds'], c['cpu_seconds']))
    os.environ.clear()
    os.environ.update({'CUDA_VISIBLE_DEVICES': c['visible_gpu'], 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                       'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                       'HOME': c['output'], 'TMPDIR': c['output']})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before isolation')
    def cpu_test() -> bytes:
        return (torch.tensor([-1.0, 0.0, 1.0]) * 2 + 1).numpy().tobytes()
    before = cpu_test()
    if len(os.listdir('/proc/self/task')) != 1 or any(
            int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('CPU warmup leaked thread or fd')
    roots = [Path(c[k]).resolve(strict=True) for k in ('runtime', 'author', 'bridge', 'entropy', 'i_model',
                                                        'p_model', 'wire', 'output')]
    roots.extend(Path(p).resolve(strict=True) for p in c['device_paths'])
    leaves, leaf_receipt = approved_leaves(c)
    roots.extend(leaves)
    source_identities = identities(c)
    allowed_roots = [str(p) for p in roots]
    abi, handled = restrict(roots, Path(c['output']).resolve(strict=True))
    mutations = mutation_probes(Path(c['canary_dir']), Path(c['output']))
    denied = {}
    for path in (Path(p) for paths in c['denied_paths'].values() for p in paths):
        try:
            with path.open('rb') as f:
                f.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f'denial not established: {path}') from exc
            denied[str(path)] = exc.errno
        else:
            raise RuntimeError(f'protected path readable: {path}')
    if before != cpu_test() or torch.cuda.is_initialized():
        raise RuntimeError('post-isolation CPU warmup failed')
    return {'policy_profile': POLICY, 'approved_leaves': leaf_receipt, 'pid': os.getpid(),
            'landlock_abi': abi, 'handled_fs_rights': handled, 'denials': denied,
            'mutation_denials': mutations, 'output_write_probe': True, 'namespaces': ns, 'network_interfaces': ['lo'],
            'protected_identities': source_identities, 'allow_roots': allowed_roots,
            'pre_sandbox_threads': 1, 'extra_fds': False,
            'cpu_before_sha256': hashlib.sha256(before).hexdigest(),
            'cpu_after_sha256': hashlib.sha256(cpu_test()).hexdigest(), 'cuda_preinitialized': False}


def cuda_probe() -> dict[str, Any]:
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('expected one CUDA device')
    p = torch.cuda.get_device_properties(0)
    x = torch.arange(1024, dtype=torch.float32, device='cuda:0')
    y = x * 2 + 1
    torch.cuda.synchronize()
    if not bool(torch.isfinite(y).all().item()) or float(y.sum().item()) != 1048576.0:
        raise RuntimeError('invalid synthetic CUDA result')
    return {'name': p.name, 'uuid': str(getattr(p, 'uuid', '')), 'count': 1,
            'capability': [p.major, p.minor], 'sum': 1048576.0, 'torch': str(torch.__version__)}


def load_entropy(path: str) -> None:
    """Load the already authenticated extension by file, without directory access."""
    name = 'MLCodec_extensions_cpp'
    if name in sys.modules:
        raise RuntimeError('entropy extension was loaded before isolation')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load pinned entropy extension')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('probe', 'generic', 'decode'))
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    args = p.parse_args()
    c = load_config(args.config, args.config_sha256, probe=args.mode in ('probe', 'generic'))
    if args.mode == 'generic':
        (Path(c['output']) / 'generic.json').write_text(json.dumps(cuda_probe(), sort_keys=True))
        return 0
    spec = importlib.util.spec_from_file_location('gvcrt_bridge', c['bridge'])
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import pinned bridge')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    receipt = bootstrap(c)
    (Path(c['output']) / 'isolation.json').write_text(json.dumps(receipt, sort_keys=True))
    if args.mode == 'probe':
        (Path(c['output']) / 'probe.json').write_text(json.dumps(cuda_probe(), sort_keys=True))
        return 0
    load_entropy(c['entropy'])
    bridge.decode(argparse.Namespace(official_repo=Path(c['author']), i_checkpoint=Path(c['i_model']),
        p_checkpoint=Path(c['p_model']), i_sha256=I_SHA, p_sha256=P_SHA, disable_fused=True,
        manifest=Path(c['wire']) / 'manifest.json', output_dir=Path(c['output']) / 'decoded',
        deny_source=[c['denied_paths']['source_archive'][0]]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
