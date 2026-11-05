"""Paired CUDA-only diagnostic under the relocated historical W8 policy.

This module deliberately does not authorize or dispatch fleet jobs. The caller
must supply authenticated historical inputs and reserve exclusive GPU time.
"""
from __future__ import annotations

import argparse
import errno
import ctypes
import stat
import hashlib
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

PROFILE = 'historical-runtime-policy-relocated-no-weight-read-v1'
PORTABLE_PROFILE = 'historical-runtime-subset-portable-no-weight-read-v2'
FACTORIAL_PROFILE = 'w8-approved-system-leaves-mutation-v1'
RUNTIME_ROOT = '/local/users/emanuele/pointstream/cache/' + 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e/tree'
SYSTEM_GRANTS = ('/usr/lib', '/lib', '/lib64', '/dev/nvidia-modeset', '/dev/nvidia-uvm',
                 '/dev/nvidia-uvm-tools', '/dev/nvidia0', '/dev/nvidia1', '/dev/nvidiactl',
                 '/dev/nvidia-caps/nvidia-cap1', '/dev/nvidia-caps/nvidia-cap2',
                 '/dev/null', '/dev/urandom', '/proc/driver/nvidia/version')
OLD_CONFIG_SHA = 'fd97361b467896222f02b1e9e07510a0a445cb7140fe9c6f7ad749b2178904ec'
SOURCE_SHA = '6412af951f9b48a47fc189390e03609627425e401e5ccaf1b158e73108f1a713'
RUNTIME_SHA = 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e'
OLD_SOURCE = '/local/users/emanuele/pointstream/sources/20261101T151030Z-df638d2f'
OLD_SCRATCH = '/local/users/emanuele/pointstream/scratch/20261101T151030Z-df638d2f-smoke'
OLD_OUTPUT = OLD_SCRATCH + '/cost-worker/output'
OLD_ALIASES = OLD_SCRATCH + '/protected-aliases'
GPU_SECONDS = 180
CPU_SECONDS = 240
ARM_SECONDS = 60
RESERVE_SECONDS = 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def replace_prefix(path: str, old: str, new: str) -> str:
    if path == old or path.startswith(old + '/'):
        return new + path[len(old):]
    return path


def normalize(original: dict[str, Any], source: Path, output: Path, aliases: Path,
              visible: str, parent_user: str, parent_net: str) -> dict[str, Any]:
    """Produce the entire explicit diff from an authenticated original config."""
    config = json.loads(json.dumps(original))
    def relocate(path: str) -> str:
        for old, new in ((OLD_SOURCE, str(source)), (OLD_OUTPUT, str(output)),
                         (OLD_ALIASES, str(aliases))):
            path = replace_prefix(path, old, new)
        return path
    config['allow_paths'] = [relocate(p) for p in config['allow_paths']]
    config['denied_paths'] = {k: [relocate(p) for p in v] for k, v in config['denied_paths'].items()}
    config['output'] = str(output)
    config['parent_user_namespace'] = parent_user
    config['parent_network_namespace'] = parent_net
    config['cpu_seconds'] = ARM_SECONDS
    config['worker_kind'] = 'preflight'
    config.pop('probe_steps')
    config['files'] = {}
    config['environment'] = {k: relocate(v) for k, v in config['environment'].items()}
    config['environment']['CUDA_VISIBLE_DEVICES'] = visible
    return config


def normalize_portable(original: dict[str, Any], source: Path, output: Path, aliases: Path,
                       visible: str, parent_user: str, parent_net: str, receipt: dict[str, str]) -> dict[str, Any]:
    config = normalize(original, source, output, aliases, visible, parent_user, parent_net)
    retained = [OLD_SOURCE, OLD_OUTPUT, RUNTIME_ROOT, *SYSTEM_GRANTS]
    if len(set(retained)) != len(retained) or any(original['allow_paths'].count(p) != 1 for p in retained):
        raise ValueError('historical grants do not contain the exact portable subset')
    config['allow_paths'] = [str(source), str(output), RUNTIME_ROOT, *SYSTEM_GRANTS]
    config['protected_receipt'] = receipt
    return config


def protected_identity(original: dict[str, Any], receipt_path: Path) -> dict[str, str]:
    # Supervisor only: stream each protected file once, recording stable file identity.
    from experiments.tennis.w8_receiver_preflight import PROTECTED

    protected = {}
    for group, (name, expected) in PROTECTED.items():
        if original['denied_paths'][group] != [name]:
            raise ValueError('historical protected path mismatch')
        path = Path(name)
        if str(path.resolve(strict=True)) != name:
            raise ValueError('protected path is not canonical')
        with path.open('rb') as f:
            before = os.fstat(f.fileno())
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
            after = os.fstat(f.fileno())
        current = path.stat()
        def identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
            return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
        if identity(before) != identity(after) or identity(before) != identity(current) or h.hexdigest() != expected:
            raise ValueError('protected input changed or identity mismatch: ' + group)
        protected[group] = {'path': name, 'sha256': expected, 'device': before.st_dev, 'inode': before.st_ino}
    write_json(receipt_path, {'protected_inputs': protected})
    return {'path': str(receipt_path), 'sha256': digest(receipt_path)}


def diff(original: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    return {k: {'before': original.get(k), 'after': normalized.get(k)}
            for k in sorted(original.keys() | normalized.keys()) if original.get(k) != normalized.get(k)}


def enough_budget(elapsed: float, cpu: float, profile: str = PROFILE) -> bool:
    limit, reserve, cpu_limit = (220, 60, 220) if profile == PORTABLE_PROFILE else (GPU_SECONDS, RESERVE_SECONDS, CPU_SECONDS)
    return (elapsed + ARM_SECONDS + reserve <= limit
            and cpu + ARM_SECONDS + reserve <= cpu_limit)



def check_isolation(receipt: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {p for paths in config['denied_paths'].values() for p in paths}
    denials = receipt.get('denials', {})
    cpu = receipt.get('cpu_bootstrap', {})
    namespaces = receipt.get('namespaces', {})
    if (set(denials) != expected or not expected
            or any(v not in (errno.EACCES, errno.EPERM) for v in denials.values())
            or type(receipt.get('landlock_abi')) is not int or receipt['landlock_abi'] < 1
            or receipt.get('network_interfaces') != ['lo'] or receipt.get('pre_sandbox_threads') != 1
            or receipt.get('extra_fds') is not False
            or not cpu.get('before_landlock_sha256')
            or cpu.get('before_landlock_sha256') != cpu.get('after_landlock_sha256')
            or cpu.get('cuda_initialized_before_isolation') is not False
            or not namespaces.get('user') or not namespaces.get('net')
            or namespaces['user'] == config['parent_user_namespace']
            or namespaces['net'] == config['parent_network_namespace']):
        raise ValueError('invalid historical isolation receipt')


T = TypeVar('T')


def without_legacy_worker_argv(call: Callable[[], T]) -> T:
    """Prevent archived import-time CLI limits from treating this worker as its own."""
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        return call()
    finally:
        sys.argv = saved


def canonical_gpu_uuid(value: str) -> uuid.UUID:
    """Compare only canonical UUID spellings, with NVIDIA's optional GPU- prefix."""
    if value.startswith('GPU-'):
        value = value[4:]
    parsed = uuid.UUID(value)
    if str(parsed) != value.lower():
        raise ValueError('noncanonical GPU UUID')
    return parsed



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




APPROVED_LEAVES = (
    '/etc/ld.so.cache', '/proc/cpuinfo', '/proc/devices',
    '/proc/driver/nvidia/capabilities/mig/config', '/proc/driver/nvidia/capabilities/mig/monitor',
    '/proc/driver/nvidia/params', '/proc/self/maps', '/proc/self/status', '/proc/stat',
    '/proc/sys/vm/mmap_min_addr', '/sys/devices/system/cpu/online',
    '/sys/devices/system/memory/block_size_bytes',
)


def approved_leaves(denied: list[Path], loader: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    protected = {(p.stat().st_dev, p.stat().st_ino) for p in denied}
    paths: list[Path] = []
    receipt: list[dict[str, Any]] = []
    for name in APPROVED_LEAVES:
        path = Path(name).resolve(strict=True)
        info = path.stat()
        expected = name.replace('/proc/self/', f'/proc/{os.getpid()}/')
        if str(path) != expected or not stat.S_ISREG(info.st_mode):
            raise ValueError('approved leaf is not the expected regular kernel or loader file: ' + name)
        if name == '/etc/ld.so.cache' and (loader['uid'] != 0 or loader['mode'] & stat.S_IWOTH
                or (info.st_dev, info.st_ino, info.st_mode) != (loader['device'], loader['inode'], loader['mode'])):
            raise ValueError('loader cache differs from supervisor-verified root-owned inode')
        if (info.st_dev, info.st_ino) in protected:
            raise ValueError('approved leaf aliases protected input')
        paths.append(path)
        receipt.append({'name': name, 'resolved': str(path), 'device': info.st_dev, 'inode': info.st_ino,
                        'mode': info.st_mode, 'uid': info.st_uid})
    return paths, receipt


def validate_leaf_receipt(receipt: list[dict[str, Any]], loader: dict[str, Any]) -> None:
    import re
    if [row.get('name') for row in receipt] != list(APPROVED_LEAVES):
        raise ValueError('approved leaf list differs')
    for row in receipt:
        name, path = row['name'], row['resolved']
        if name.startswith('/proc/self/'):
            suffix = re.escape(name.removeprefix('/proc/self/'))
            if not re.fullmatch(r'/proc/[0-9]+/' + suffix, path):
                raise ValueError('self leaf was not child-resolved')
        elif path != name:
            raise ValueError('approved leaf resolved to different path')
        if not stat.S_ISREG(row['mode']) or any(type(row[k]) is not int or row[k] < 0 for k in ('device', 'inode', 'mode', 'uid')):
            raise ValueError('approved leaf file identity malformed')
        if name == '/etc/ld.so.cache' and (loader['uid'] != 0 or loader['mode'] & stat.S_IWOTH
                or any(row[k] != loader[k] for k in ('device', 'inode', 'mode'))):
            raise ValueError('loader cache is unsafe')


def run_worker(args: argparse.Namespace) -> int:
    import socket
    sys.path.insert(0, args.source)
    from experiments.tennis import w8_receiver_bootstrap as bootstrap
    if not Path(bootstrap.__file__).resolve().is_relative_to(Path(args.source).resolve()):
        raise ValueError('bootstrap source identity mismatch')
    config = bootstrap.load_config(Path(args.config), args.config_sha256)
    loader_path = Path(args.config).parent / 'loader-identity.json'
    if digest(loader_path) != args.loader_sha256:
        raise ValueError('loader identity receipt mismatch')
    loader = json.loads(loader_path.read_text())
    ns = True
    strict = args.condition == 'strict'
    add_leaves = args.condition != 'portable'
    result: dict[str, Any] = {'condition': args.condition, 'status': 'failed', 'phase': 'bootstrap'}
    try:
        if len(os.sched_getaffinity(0)) != 1:
            raise ValueError('one CPU required')
        namespaces = {n: os.readlink('/proc/self/ns/' + n) for n in ('user', 'net')}
        expected = {'user': config['parent_user_namespace'], 'net': config['parent_network_namespace']}
        if any((namespaces[k] != expected[k]) != ns for k in expected):
            raise ValueError('unexpected namespace identity')
        if ns and sorted(name for _, name in socket.if_nameindex()) != ['lo']:
            raise ValueError('unexpected namespace interfaces')
        def clean() -> None:
            if len(os.listdir('/proc/self/task')) != 1:
                raise ValueError('unexpected thread count')
            if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
                raise ValueError('unexpected open file descriptor')
        clean()
        allowed = [Path(p).resolve(strict=True) for p in config['allow_paths']]
        denied = bootstrap.flatten_denials(config['denied_paths'])
        for p in denied:
            target = p.resolve(strict=True)
            if any(target == root or target.is_relative_to(root) for root in allowed):
                raise ValueError('protected path allowed')
        resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
        os.environ.clear()
        os.environ.update(config['environment'])
        os.environ.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                           'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        import torch
        def import_warmup() -> Any:
            from experiments.tennis.w8_input_job import synthetic_torch_bootstrap
            return synthetic_torch_bootstrap
        warmup = without_legacy_worker_argv(import_warmup)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():
            raise ValueError('CUDA initialized before isolation')
        before = warmup(torch)
        clean()
        sys.addaudithook(bootstrap._deny_processes)
        leaf_receipt: list[dict[str, Any]] = []
        if add_leaves:
            leaf_paths, leaf_receipt = approved_leaves(denied, loader)
            allowed.extend(leaf_paths)
        canary = Path(config['output']).parent / 'mutation-canary'
        if strict:
            abi, mask = restrict(allowed, Path(config['output']).resolve(strict=True))
            mutation = mutation_probes(canary, Path(config['output']))
        else:
            abi, mask = bootstrap._restrict(allowed), 13
            mutation = {}
        denials = bootstrap._check_denials(denied)
        after = warmup(torch)
        if before != after or torch.cuda.is_initialized():
            raise ValueError('CPU warmup or CUDA preinitialization mismatch')
        result['isolation'] = {'namespaces': namespaces, 'landlock_abi': abi, 'handled_mask': mask, 'approved_leaves': leaf_receipt, 'mutation_denials': mutation, 'output_write_ok': True if strict else None,
                               'denials': denials, 'cpu_before': before, 'cpu_after': after,
                               'cuda_preinitialized': False, 'threads': 1, 'extra_fds': False}
        result['phase'] = 'cuda_probe'
        if torch.cuda.device_count() != 1:
            raise RuntimeError('expected one visible CUDA device')
        props = torch.cuda.get_device_properties(0)
        x = torch.arange(1024, dtype=torch.float32, device='cuda:0')
        y = x * 2 + 1
        torch.cuda.synchronize()
        total = float(y.sum().item())
        if not bool(torch.isfinite(y).all().item()) or total != 1048576.0:
            raise RuntimeError('invalid CUDA result')
        result['cuda'] = {'name': props.name, 'uuid': str(getattr(props, 'uuid', '')),
                          'capability': [props.major, props.minor], 'total_memory_bytes': props.total_memory,
                          'device_count': 1, 'sum': total, 'torch': str(torch.__version__),
                          'torch_cuda': str(torch.version.cuda), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        result['status'] = 'passed'
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc(file=sys.stderr)
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['usage'] = {'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime}
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'passed' else 1


def cell_command(python: str, script: str, condition: str, config: Path, sha: str, source: Path, loader_sha: str) -> list[str]:
    if condition not in ('portable', 'leaves', 'strict'):
        raise ValueError('unknown policy cell')
    cmd = [python, script, 'worker', '--condition', condition, '--config', str(config),
           '--config-sha256', sha, '--source', str(source), '--loader-sha256', loader_sha]
    return ['/usr/bin/unshare', '--user', '--map-root-user', '--net', *cmd]


def check_gpu(visible: str) -> dict[str, Any]:
    command = ['/usr/bin/nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits']
    query = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    rows = [line.split(', ') for line in query.stdout.splitlines() if line.strip()]
    selected = [row for row in rows if row[0] == visible]
    if len(selected) != 1:
        raise ValueError('claimed physical GPU identity missing')
    apps = subprocess.run(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                           '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if fields[0] == visible:
            raise ValueError('GPU has a foreign or lingering compute process')
    return {'uuid': selected[0][0], 'name': selected[0][1], 'memory_mib': int(selected[0][2]),
            'compute_processes': []}


def check_dispatch(dispatch: dict[str, Any], python: Path) -> dict[str, Any]:
    packed = dispatch['staged_environment']
    if (dispatch['gpu_sharing']['mode'] != 'exclusive' or packed['sha256'] != RUNTIME_SHA
            or Path(packed['python']).resolve() != python.resolve()):
        raise ValueError('fleet runtime or GPU dispatch identity mismatch')
    return packed


def make_spec(deadline: str) -> dict[str, Any]:
    base = Path(__file__).resolve().parents[2] / 'docs/experiments/w8-inputs/receiver-20261101/w8-cuda-portable-v2-repair1-prospective-20261105.json'
    spec = json.loads(base.read_text())
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_cuda_system_leaves']
    spec['arguments'] = ['run', '--source-archive', '{staged:historical_source}', '--phase', '{phase}']
    spec['smoke'] = {'seconds': 100, 'representative_basis': 'Three namespace-on fresh processes compare exact portable policy, twelve approved system-file leaves, and those leaves with full mutation confinement.'}
    spec['full'] = {'seconds': 1}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_cuda_system_leaves', 'validate']
    spec['validator_seconds'] = 15
    spec['budget_seconds'] = 120
    spec['stall_seconds'] = 100
    spec['deadline'] = deadline
    return spec


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    stage = Path(args.stage).resolve(strict=True)
    if any(stage.iterdir()):
        raise ValueError('stage must be empty')
    if os.environ.get('PS_GPU_SHARING', 'exclusive') != 'exclusive':
        raise ValueError('exclusive GPU required')
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one-CPU affinity required')
    if digest(Path(args.original_config)) != OLD_CONFIG_SHA or digest(Path(args.source_archive)) != SOURCE_SHA:
        raise ValueError('historical identity mismatch')
    historical_raw = Path(args.original_config).read_bytes()
    original = json.loads(historical_raw)
    with (stage / 'historical-config.json').open('xb') as f:
        f.write(historical_raw)
    scratch = Path(os.environ['PS_SCRATCH_DIR']).resolve(strict=True) / 'w8-cuda-diagnostic'
    scratch.mkdir()
    source = scratch / 'historical-source'
    source.mkdir()
    with tarfile.open(args.source_archive) as archive:
        archive.extractall(source, filter='data')
    source_members = {}
    with tarfile.open(args.source_archive) as archive:
        for relative in ('experiments/tennis/w8_receiver_bootstrap.py', 'experiments/tennis/w8_input_job.py'):
            archived = archive.extractfile(relative)
            if archived is None:
                raise ValueError('authenticated source member missing')
            member_sha = hashlib.sha256(archived.read()).hexdigest()
            if digest(source / relative) != member_sha:
                raise ValueError('extracted source member differs')
            source_members[relative] = member_sha
    for p in source.rglob('*'):
        if not p.is_symlink():
            p.chmod(0o555 if p.is_dir() else 0o444)
    source.chmod(0o555)
    aliases = scratch / 'aliases'
    aliases.mkdir()
    for group in ('source', 'prepared', 'fit', 'golden', 'masks'):
        (aliases / group).symlink_to(Path(original['denied_paths'][group][0]).resolve(strict=True))
    output = scratch / 'isolated-output'
    output.mkdir()
    canary = scratch / 'mutation-canary'
    canary.mkdir()
    for name in ('write', 'truncate', 'remove', 'rename', 'link'):
        (canary / name).write_bytes(b'canary')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != os.environ.get('PS_CLAIMED_GPU_UUID'):
        raise ValueError('fleet claimed UUID differs from visibility')
    if not visible.startswith('GPU-') or ',' in visible:
        raise ValueError('one physical GPU UUID required')
    dispatch = json.loads((Path(os.environ['PS_STAGE_DIR']) / 'dispatch.json').read_text())
    packed = check_dispatch(dispatch, Path(sys.executable))
    clock = dispatch.get('campaign_clock')
    if not isinstance(clock, dict) or not isinstance(clock.get('budget_end'), (int, float)):
        raise ValueError('single-CPU fleet campaign clock required')
    gpu_info = check_gpu(visible)
    parent_user, parent_net = os.readlink('/proc/self/ns/user'), os.readlink('/proc/self/ns/net')
    if args.profile == PORTABLE_PROFILE:
        receipt = protected_identity(original, stage / 'protected-identity.json')
        config = normalize_portable(original, source, output, aliases, visible, parent_user, parent_net, receipt)
    else:
        config = normalize(original, source, output, aliases, visible, parent_user, parent_net)
    loader_path = Path('/etc/ld.so.cache').resolve(strict=True)
    loader_stat = loader_path.stat()
    if str(loader_path) != '/etc/ld.so.cache' or not stat.S_ISREG(loader_stat.st_mode) or loader_stat.st_uid != 0 or loader_stat.st_mode & stat.S_IWOTH:
        raise ValueError('loader cache owner or permissions invalid')
    loader_receipt = {'device': loader_stat.st_dev, 'inode': loader_stat.st_ino,
                      'mode': loader_stat.st_mode, 'uid': loader_stat.st_uid}
    write_json(stage / 'loader-identity.json', loader_receipt)
    loader_sha = digest(stage / 'loader-identity.json')
    config_path = stage / 'worker-config.json'
    write_json(config_path, config)
    config_sha = digest(config_path)
    write_json(stage / 'manifest.json', {'profile': FACTORIAL_PROFILE, 'historical_config_sha256': OLD_CONFIG_SHA,
        'source_sha256': SOURCE_SHA, 'source_member_sha256': source_members, 'runtime_archive_sha256': RUNTIME_SHA,
        'script_sha256': digest(Path(__file__)), 'config_sha256': config_sha, 'loader_identity_sha256': loader_sha,
        'normalized_diff': diff(original, config), 'gpu_uuid': visible,
        'dispatch_sha256': digest(Path(os.environ['PS_STAGE_DIR']) / 'dispatch.json'),
        'gpu_before': gpu_info, 'staged_environment': packed,
        'limitations': ('Supervisor hashes opaque protected archives, which may contain weights; worker arms never read or deserialize weights, construct a model or perform inference. Stricter subset: pass establishes only runtime/policy CUDA availability; failure may be subset-specific. '
                        'Neither outcome diagnoses the past failure or qualifies model generation.' if args.profile == PORTABLE_PROFILE
                        else 'CUDA diagnostic only; no model qualification')})
    runtime = Path(sys.executable).resolve(strict=True)
    expected_root = Path('/local/users/emanuele/pointstream/cache') / RUNTIME_SHA / 'tree'
    if not runtime.is_relative_to(expected_root.resolve(strict=True)):
        raise ValueError('not the qualified JCV runtime')
    results: list[dict[str, Any]] = []
    env = {'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', **config['environment']}
    for condition in ('portable', 'leaves', 'strict'):
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = own.ru_utime + own.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime + children.ru_utime + children.ru_stime
        if (time.monotonic() - started + 25 + 15 > 100 or cpu + 25 + 15 > 100
                or time.monotonic() + 25 + 15 > clock['budget_end']):
            results.append({'condition': condition, 'admitted': False, 'reason': 'budget'})
            break
        check_gpu(visible)
        arm_env = env
        cmd = cell_command(str(runtime), str(Path(__file__).resolve()), condition, config_path, config_sha, source, loader_sha)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        timed_out = False
        with (stage / f'{condition}.stdout').open('xb') as stdout, (stage / f'{condition}.stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=arm_env, close_fds=True)
            try:
                status = proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                timed_out = True
                from experiments.visor.gvcrt_entropy_pilot import kill_owned_tree
                kill_owned_tree(proc.pid)
                status = proc.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        item = {'condition': condition, 'argv': cmd, 'admitted': True, 'status': status, 'timed_out': timed_out,
                'wall_seconds': time.monotonic() - start,
                'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                'stdout_sha256': digest(stage / f'{condition}.stdout'),
                'stderr_sha256': digest(stage / f'{condition}.stderr')}
        write_json(stage / f'{condition}.exit.json', item)
        results.append(item)
        try:
            check_gpu(visible)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            results.append({'condition': 'gpu_exclusivity', 'admitted': False, 'reason': str(exc)})
            break
    canary_ok = all((canary / name).read_bytes() == b'canary' for name in ('write', 'truncate', 'remove', 'rename', 'link')) and {x.name for x in canary.iterdir()} == {'write', 'truncate', 'remove', 'rename', 'link'}
    write_json(stage / 'canary-integrity.json', {'preserved': canary_ok, 'names': sorted(x.name for x in canary.iterdir())})
    write_json(stage / 'summary.json', {'profile': FACTORIAL_PROFILE, 'arms': results,
               'elapsed_seconds': time.monotonic() - started})
    return validate(stage)


def validate(stage: Path) -> int:
    manifest = json.loads((stage / 'manifest.json').read_text())
    config = json.loads((stage / 'worker-config.json').read_text())
    if (manifest['profile'] != FACTORIAL_PROFILE or manifest['source_sha256'] != SOURCE_SHA
            or manifest['runtime_archive_sha256'] != RUNTIME_SHA or digest(stage / 'worker-config.json') != manifest['config_sha256']
            or manifest['script_sha256'] != digest(Path(__file__))
            or manifest['dispatch_sha256'] != digest(stage.parent / 'dispatch.json')):
        raise ValueError('factorial identity mismatch')
    original = json.loads((stage / 'historical-config.json').read_text())
    if digest(stage / 'historical-config.json') != OLD_CONFIG_SHA:
        raise ValueError('historical config identity mismatch')
    receipt = config['protected_receipt']
    if digest(Path(receipt['path'])) != receipt['sha256']:
        raise ValueError('protected receipt mismatch')
    expected = normalize_portable(original, Path(config['allow_paths'][0]), Path(config['output']),
        Path(config['denied_paths']['aliases'][0]).parent, manifest['gpu_uuid'],
        config['parent_user_namespace'], config['parent_network_namespace'], receipt)
    if config != expected or manifest['normalized_diff'] != diff(original, config):
        raise ValueError('portable policy differs')
    if digest(stage / 'loader-identity.json') != manifest['loader_identity_sha256']:
        raise ValueError('loader identity receipt mismatch')
    loader = json.loads((stage / 'loader-identity.json').read_text())
    source = Path(config['allow_paths'][0])
    dispatch = json.loads((stage.parent / 'dispatch.json').read_text())
    check_dispatch(dispatch, Path(sys.executable))
    staged = [item for item in dispatch['staged_inputs'] if item['name'] == 'historical_source']
    if len(staged) != 1 or staged[0]['sha256'] != SOURCE_SHA or digest(Path(staged[0]['path'])) != SOURCE_SHA:
        raise ValueError('historical source archive identity mismatch')
    with tarfile.open(staged[0]['path']) as archive:
        for relative, expected_sha in manifest['source_member_sha256'].items():
            archived = archive.extractfile(relative)
            if archived is None or hashlib.sha256(archived.read()).hexdigest() != expected_sha:
                raise ValueError('archived source member mismatch')
    if len(manifest['source_member_sha256']) != 2:
        raise ValueError('source member identities missing')
    summary = json.loads((stage / 'summary.json').read_text())
    arms = summary['arms']
    complete = len(arms) == 3 and [x['condition'] for x in arms] == ['portable', 'leaves', 'strict']
    outcomes = {}
    for arm in arms:
        name = arm['condition']
        if name not in ('portable', 'leaves', 'strict') or not arm.get('admitted'):
            complete = False
            continue
        expected_argv = cell_command(str(Path(manifest['staged_environment']['python']).resolve()), str(Path(__file__).resolve()), name, stage / 'worker-config.json', manifest['config_sha256'], source, manifest['loader_identity_sha256'])
        if arm.get('argv') != expected_argv:
            raise ValueError('worker argv differs')
        if any(digest(stage / (name + suffix)) != arm[key] for suffix, key in (('.stdout', 'stdout_sha256'), ('.stderr', 'stderr_sha256'))):
            raise ValueError('worker output identity mismatch')
        if arm['timed_out']:
            complete = False
            continue
        try:
            record = json.loads((stage / (name + '.stdout')).read_text())
            isolation = record['isolation']
            expected_namespaces = {'user': config['parent_user_namespace'], 'net': config['parent_network_namespace']}
            if any(isolation['namespaces'][k] == v for k, v in expected_namespaces.items()):
                raise ValueError('namespace condition mismatch')
            if isolation['cpu_before'] != isolation['cpu_after'] or isolation['cuda_preinitialized'] is not False or isolation['threads'] != 1 or isolation['extra_fds'] is not False:
                raise ValueError('warmup checks failed')
            denied = {p for paths in config['denied_paths'].values() for p in paths}
            if isolation['landlock_abi'] < 1 or set(isolation['denials']) != denied or any(x not in (1, 13) for x in isolation['denials'].values()):
                raise ValueError('Landlock denial check failed')
            leaves = isolation['approved_leaves']
            if name == 'portable':
                if leaves or isolation['handled_mask'] != 13:
                    raise ValueError('portable policy differs')
            else:
                validate_leaf_receipt(leaves, loader)
                if name == 'leaves' and isolation['handled_mask'] != 13:
                    raise ValueError('read policy differs')
                if name == 'strict':
                    if isolation['handled_mask'] != handled_rights(isolation['landlock_abi']):
                        raise ValueError('strict mutation rights differ')
                    mutation = isolation['mutation_denials']
                    if (isolation['output_write_ok'] is not True or set(mutation) != {'write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'}
                            or any(v not in ((1, 13, 18) if k in ('rename', 'link') else (1, 13)) for k, v in mutation.items())):
                        raise ValueError('mutation probes incomplete')
            if record['phase'] != 'cuda_probe' or record['condition'] != name:
                raise ValueError('probe did not execute')
            if record['status'] == 'passed':
                cuda = record['cuda']
                if (arm['status'] != 0 or cuda['sum'] != 1048576.0 or cuda['device_count'] != 1
                        or canonical_gpu_uuid(cuda['uuid']) != canonical_gpu_uuid(manifest['gpu_uuid'])
                        or cuda['name'] != manifest['gpu_before']['name'] or cuda['visible_devices'] != manifest['gpu_uuid']):
                    raise ValueError('CUDA identity mismatch')
            elif arm['status'] == 0 or not record.get('error'):
                raise ValueError('failed result missing receipt')
            outcomes[name] = record['status']
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            complete = False
    if not json.loads((stage / 'canary-integrity.json').read_text())['preserved']:
        complete = False
    verdict = {'measurement_complete': complete, 'outcomes': outcomes, 'generation_gate_passed': False}
    target = stage / 'validation.json'
    if target.exists():
        if json.loads(target.read_text()) != verdict:
            raise ValueError('existing validation differs')
    else:
        write_json(target, verdict)
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--stage')
    p.add_argument('--phase', choices=('smoke', 'full'), required=True)
    p.add_argument('--original-config', default=str(Path(__file__).with_name('w8_cuda_historical_config.json')))
    p.add_argument('--source-archive', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--condition', choices=('portable', 'leaves', 'strict'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--loader-sha256', required=True)
    p = sub.add_parser('make-spec')
    p.add_argument('--deadline', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('validate')
    p.add_argument('--stage')
    args = parser.parse_args()
    if args.command == 'make-spec':
        spec = make_spec(args.deadline)
        from experiments.jobs.inbox import validate_spec
        validate_spec(spec)
        write_json(args.output, spec)
        return 0
    if args.command == 'worker':
        return run_worker(args)
    if args.command == 'validate':
        stage = Path(args.stage) if args.stage else Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-leaves'
        status = validate(stage)
        if os.environ.get('PS_VALIDATION_PATH'):
            verdict = json.loads((stage / 'validation.json').read_text())
            write_json(Path(os.environ['PS_VALIDATION_PATH']), {'passed': status == 0, 'checks': verdict})
        return status
    if os.environ.get('PS_STAGE') != args.phase:
        raise ValueError('fleet phase mismatch')
    if args.phase == 'full':
        return 0
    args.profile = PORTABLE_PROFILE
    if args.stage is None:
        args.stage = str(Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-leaves')
        Path(args.stage).mkdir()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
