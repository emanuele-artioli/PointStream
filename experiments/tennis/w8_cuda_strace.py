"""Paired CUDA-only diagnostic under the relocated historical W8 policy.

This module deliberately does not authorize or dispatch fleet jobs. The caller
must supply authenticated historical inputs and reserve exclusive GPU time.
"""
from __future__ import annotations

import argparse
import errno
import ctypes
import hashlib
import json
import os
import resource
import shutil
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
FACTORIAL_PROFILE = 'portable-w8-nnp-landlock-strace-v1'
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


def worker_command(python: str, script: str, condition: str, config: Path, sha: str,
                   source: Path) -> list[str]:
    cmd = [python, script, 'worker', '--condition', condition, '--config', str(config),
           '--config-sha256', sha, '--source', str(source)]
    return ['/usr/bin/unshare', '--user', '--map-root-user', '--net', *cmd] if condition == 'isolated' else cmd


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


def run_worker(args: argparse.Namespace) -> int:
    import socket
    sys.path.insert(0, args.source)
    from experiments.tennis import w8_receiver_bootstrap as bootstrap
    if not Path(bootstrap.__file__).resolve().is_relative_to(Path(args.source).resolve()):
        raise ValueError('bootstrap source identity mismatch')
    config = bootstrap.load_config(Path(args.config), args.config_sha256)
    ns = False
    ll = args.condition == 'landlock'
    nnp = args.condition == 'nnp'
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
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
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
        libc = ctypes.CDLL(None, use_errno=True)
        before_nnp = libc.prctl(39, 0, 0, 0, 0)
        if before_nnp != 0:
            raise RuntimeError('NoNewPrivs already enabled or unreadable')
        sys.addaudithook(bootstrap._deny_processes)
        if nnp and libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS')
        abi = bootstrap._restrict(allowed) if ll else None
        after_nnp = libc.prctl(39, 0, 0, 0, 0)
        if after_nnp != (1 if nnp or ll else 0):
            raise RuntimeError('NoNewPrivs condition mismatch')
        denials = bootstrap._check_denials(denied) if ll else {}
        after = warmup(torch)
        if before != after or torch.cuda.is_initialized():
            raise ValueError('CPU warmup or CUDA preinitialization mismatch')
        result['isolation'] = {'namespaces': namespaces, 'landlock_abi': abi, 'handled_mask': 13 if ll else 0,
                               'denials': denials, 'cpu_before': before, 'cpu_after': after,
                               'cuda_preinitialized': False, 'threads': 1, 'extra_fds': False,
                               'nnp_before': before_nnp, 'nnp_after': after_nnp}
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


def cell_command(python: str, script: str, condition: str, config: Path, sha: str, source: Path) -> list[str]:
    if condition not in ('baseline', 'nnp', 'landlock'):
        raise ValueError('unknown trace cell')
    trace = source.parent / f'{condition}.trace'
    return ['/usr/bin/strace', '-f', '-s', '512', '-o', str(trace), '-e',
            'trace=open,openat,access,faccessat,faccessat2,statx,newfstatat,mmap,mprotect,ioctl,prctl',
            python, script, 'worker', '--condition', condition, '--config', str(config),
            '--config-sha256', sha, '--source', str(source)]


def denied_candidates(trace: str, denied_paths: list[str]) -> list[str]:
    return [line for line in trace.splitlines() if ('EACCES' in line or 'EPERM' in line)
            and not any(path in line for path in denied_paths)]


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
    spec['entrypoint'] = ['-m', 'experiments.tennis.w8_cuda_strace']
    spec['arguments'] = ['run', '--source-archive', '{staged:historical_source}', '--phase', '{phase}']
    spec['smoke'] = {'seconds': 120, 'representative_basis': 'Three fresh equally traced controls: baseline, NoNewPrivs only and exact portable Landlock; no namespaces.'}
    spec['full'] = {'seconds': 1}
    spec['validator'] = ['{python}', '-m', 'experiments.tennis.w8_cuda_strace', 'validate']
    spec['validator_seconds'] = 20
    spec['budget_seconds'] = 150
    spec['stall_seconds'] = 120
    spec['required_commands'] = ['nvidia-smi', 'strace']
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
    config_path = stage / 'worker-config.json'
    write_json(config_path, config)
    config_sha = digest(config_path)
    trace_tool = Path('/usr/bin/strace')
    trace_sha = digest(trace_tool)
    trace_version = subprocess.check_output([str(trace_tool), '--version'], text=True, timeout=5).splitlines()[0]
    write_json(stage / 'manifest.json', {'strace_sha256': trace_sha, 'strace_version': trace_version, 'profile': FACTORIAL_PROFILE, 'historical_config_sha256': OLD_CONFIG_SHA,
        'source_sha256': SOURCE_SHA, 'source_member_sha256': source_members, 'runtime_archive_sha256': RUNTIME_SHA,
        'script_sha256': digest(Path(__file__)), 'config_sha256': config_sha,
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
    for condition in ('baseline', 'nnp', 'landlock'):
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = own.ru_utime + own.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime + children.ru_utime + children.ru_stime
        if (time.monotonic() - started + 30 + 20 > 120 or cpu + 30 + 20 > 120
                or time.monotonic() + 30 + 20 > clock['budget_end']):
            results.append({'condition': condition, 'admitted': False, 'reason': 'budget'})
            break
        check_gpu(visible)
        arm_env = env
        cmd = cell_command(str(runtime), str(Path(__file__).resolve()), condition, config_path, config_sha, source)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        timed_out = False
        with (stage / f'{condition}.stdout').open('xb') as stdout, (stage / f'{condition}.stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=arm_env, close_fds=True)
            try:
                status = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                timed_out = True
                from experiments.visor.gvcrt_entropy_pilot import kill_owned_tree
                kill_owned_tree(proc.pid)
                status = proc.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        local_trace = source.parent / f'{condition}.trace'
        if local_trace.is_file():
            shutil.copyfile(local_trace, stage / f'{condition}.trace')
        else:
            (stage / f'{condition}.trace').touch(exist_ok=False)
        trace_text = (stage / f'{condition}.trace').read_text(errors='replace')
        excluded = [p for paths in config['denied_paths'].values() for p in paths]
        candidates = denied_candidates(trace_text, excluded)
        write_json(stage / f'{condition}.trace-summary.json', {'count': len(candidates), 'candidates_first_200': candidates[:200], 'intentional_protected_probes_excluded': True})
        item = {'condition': condition, 'argv': cmd, 'admitted': True, 'status': status, 'timed_out': timed_out,
                'wall_seconds': time.monotonic() - start,
                'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                'stdout_sha256': digest(stage / f'{condition}.stdout'),
                'stderr_sha256': digest(stage / f'{condition}.stderr'),
                'trace_sha256': digest(stage / f'{condition}.trace'),
                'trace_summary_sha256': digest(stage / f'{condition}.trace-summary.json')}
        write_json(stage / f'{condition}.exit.json', item)
        results.append(item)
        try:
            check_gpu(visible)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            results.append({'condition': 'gpu_exclusivity', 'admitted': False, 'reason': str(exc)})
            break
    write_json(stage / 'summary.json', {'profile': FACTORIAL_PROFILE, 'arms': results,
               'elapsed_seconds': time.monotonic() - started})
    return validate(stage)


def validate(stage: Path) -> int:
    manifest = json.loads((stage / 'manifest.json').read_text())
    config = json.loads((stage / 'worker-config.json').read_text())
    if (manifest['profile'] != FACTORIAL_PROFILE or manifest['source_sha256'] != SOURCE_SHA
            or manifest['runtime_archive_sha256'] != RUNTIME_SHA or digest(stage / 'worker-config.json') != manifest['config_sha256']
            or manifest['script_sha256'] != digest(Path(__file__))
            or manifest['strace_sha256'] != digest(Path('/usr/bin/strace'))
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
    complete = len(arms) == 3 and [x['condition'] for x in arms] == ['baseline', 'nnp', 'landlock']
    outcomes = {}
    for arm in arms:
        name = arm['condition']
        if name not in ('baseline', 'nnp', 'landlock') or not arm.get('admitted'):
            complete = False
            continue
        expected_argv = cell_command(str(Path(manifest['staged_environment']['python']).resolve()), str(Path(__file__).resolve()), name, stage / 'worker-config.json', manifest['config_sha256'], source)
        if arm.get('argv') != expected_argv:
            raise ValueError('worker argv differs')
        if any(digest(stage / (name + suffix)) != arm[key] for suffix, key in (('.stdout', 'stdout_sha256'), ('.stderr', 'stderr_sha256'))):
            raise ValueError('worker output identity mismatch')
        if digest(stage / (name + '.trace')) != arm['trace_sha256']:
            raise ValueError('trace identity mismatch')
        trace = (stage / (name + '.trace')).read_text(errors='replace')
        excluded = [p for paths in config['denied_paths'].values() for p in paths]
        candidates = denied_candidates(trace, excluded)
        trace_summary_path = stage / (name + '.trace-summary.json')
        if (digest(trace_summary_path) != arm['trace_summary_sha256'] or json.loads(trace_summary_path.read_text()) !=
                {'count': len(candidates), 'candidates_first_200': candidates[:200], 'intentional_protected_probes_excluded': True}):
            raise ValueError('trace summary mismatch')
        if arm['timed_out']:
            complete = False
            continue
        try:
            record = json.loads((stage / (name + '.stdout')).read_text())
            isolation = record['isolation']
            expected_namespaces = {'user': config['parent_user_namespace'], 'net': config['parent_network_namespace']}
            if any(isolation['namespaces'][k] != v for k, v in expected_namespaces.items()):
                raise ValueError('namespace condition mismatch')
            if isolation['cpu_before'] != isolation['cpu_after'] or isolation['cuda_preinitialized'] is not False or isolation['threads'] != 1 or isolation['extra_fds'] is not False:
                raise ValueError('warmup checks failed')
            if name == 'landlock':
                denied = {p for paths in config['denied_paths'].values() for p in paths}
                if isolation['handled_mask'] != 13 or isolation['landlock_abi'] < 1 or set(isolation['denials']) != denied or any(x not in (1, 13) for x in isolation['denials'].values()):
                    raise ValueError('Landlock denial check failed')
            elif isolation['handled_mask'] != 0 or isolation['landlock_abi'] is not None or isolation['denials']:
                raise ValueError('unexpected Landlock or protected read')
            if isolation['nnp_before'] != 0 or isolation['nnp_after'] != (0 if name == 'baseline' else 1):
                raise ValueError('NoNewPrivs mismatch')
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
    p.add_argument('--condition', choices=('baseline', 'nnp', 'landlock'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--source', required=True)
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
        stage = Path(args.stage) if args.stage else Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-strace'
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
        args.stage = str(Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-strace')
        Path(args.stage).mkdir()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
