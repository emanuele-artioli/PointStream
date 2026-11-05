"""Paired CUDA-only diagnostic under the relocated historical W8 policy.

This module deliberately does not authorize or dispatch fleet jobs. The caller
must supply authenticated historical inputs and reserve exclusive GPU time.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

PROFILE = 'historical-runtime-policy-relocated-no-weight-read-v1'
PORTABLE_PROFILE = 'historical-runtime-subset-portable-no-weight-read-v2'
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


def run_worker(args: argparse.Namespace) -> int:
    # The supervisor is a directly executed script; importing the archived
    # package here must precede any import from the current checkout.
    sys.path.insert(0, args.source)
    from experiments.tennis import w8_receiver_bootstrap as bootstrap
    if not Path(bootstrap.__file__).resolve().is_relative_to(Path(args.source).resolve()):
        raise ValueError('bootstrap did not come from authenticated source')
    resource.setrlimit(resource.RLIMIT_CPU, (ARM_SECONDS, ARM_SECONDS))
    config = bootstrap.load_config(Path(args.config), args.config_sha256)
    isolation = None
    result: dict[str, Any] = {'condition': args.condition, 'status': 'failed'}
    try:
        if args.condition == 'isolated':
            isolation = bootstrap.bootstrap(config)
            result['isolation'] = isolation
            check_isolation(isolation, config)
        import torch
        torch.set_num_threads(1)
        result['phase'] = 'cuda_probe'
        if torch.cuda.device_count() != 1:
            raise RuntimeError('expected one visible CUDA device')
        p = torch.cuda.get_device_properties(0)
        x = torch.arange(1024, dtype=torch.float32, device='cuda:0')
        y = x * 2 + 1
        torch.cuda.synchronize()
        total = float(y.sum().item())
        if x.device.type != 'cuda' or not bool(torch.isfinite(y).all().item()) or total != 1048576.0:
            raise RuntimeError('invalid CUDA result')
        result['cuda'] = {'name': p.name, 'uuid': str(getattr(p, 'uuid', '')),
                          'capability': [p.major, p.minor], 'total_memory': p.total_memory,
                          'device_count': 1, 'sum': total, 'tensor_bytes': 8192,
                          'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda),
                          'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        result['status'] = 'passed'
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc(file=sys.stderr)
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['usage'] = {'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime}
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'passed' else 1


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
    packed = dispatch['staged_environment']
    if (dispatch['gpu_sharing']['mode'] != 'exclusive' or packed['sha256'] != RUNTIME_SHA
            or Path(packed['python']).resolve() != Path(sys.executable).resolve()):
        raise ValueError('fleet runtime or GPU dispatch identity mismatch')
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
    write_json(stage / 'manifest.json', {'profile': args.profile, 'historical_config_sha256': OLD_CONFIG_SHA,
        'source_sha256': SOURCE_SHA, 'runtime_archive_sha256': RUNTIME_SHA,
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
    for condition in ('generic', 'isolated'):
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = own.ru_utime + own.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime + children.ru_utime + children.ru_stime
        reserve = 60 if args.profile == PORTABLE_PROFILE else RESERVE_SECONDS
        if (not enough_budget(time.monotonic() - started, cpu, args.profile)
                or time.monotonic() + ARM_SECONDS + reserve > clock['budget_end']):
            results.append({'condition': condition, 'admitted': False, 'reason': 'budget'})
            break
        check_gpu(visible)
        if condition == 'generic':
            generic_tmp = scratch / 'generic-tmp'
            generic_tmp.mkdir()
            arm_env = {**env, 'TMPDIR': str(generic_tmp), 'TMP': str(generic_tmp), 'TEMP': str(generic_tmp),
                       'XDG_CACHE_HOME': str(generic_tmp / '.cache'), 'TORCH_HOME': str(generic_tmp / '.cache' / 'torch')}
        else:
            arm_env = env
        cmd = worker_command(str(runtime), str(Path(__file__).resolve()), condition, config_path, config_sha, source)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        timed_out = False
        with (stage / f'{condition}.stdout').open('xb') as stdout, (stage / f'{condition}.stderr').open('xb') as stderr:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env=arm_env, close_fds=True)
            try:
                status = proc.wait(timeout=ARM_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                status = proc.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        item = {'condition': condition, 'admitted': True, 'status': status, 'timed_out': timed_out,
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
    write_json(stage / 'summary.json', {'profile': args.profile, 'arms': results,
               'elapsed_seconds': time.monotonic() - started})
    return validate(stage)


def validate(stage: Path) -> int:
    manifest = json.loads((stage / 'manifest.json').read_text())
    summary = json.loads((stage / 'summary.json').read_text())
    if (manifest['profile'] not in (PROFILE, PORTABLE_PROFILE) or manifest['historical_config_sha256'] != OLD_CONFIG_SHA
            or manifest['source_sha256'] != SOURCE_SHA or manifest['runtime_archive_sha256'] != RUNTIME_SHA
            or digest(stage / 'historical-config.json') != OLD_CONFIG_SHA
            or digest(stage / 'worker-config.json') != manifest['config_sha256']):
        raise ValueError('manifest identity mismatch')
    original = json.loads((stage / 'historical-config.json').read_text())
    config = json.loads((stage / 'worker-config.json').read_text())
    if diff(original, config) != manifest['normalized_diff']:
        raise ValueError('normalized configuration diff mismatch')
    source = Path(config['allow_paths'][0])
    output = Path(config['output'])
    aliases = Path(config['denied_paths']['aliases'][0]).parent
    if manifest['profile'] == PORTABLE_PROFILE:
        receipt_path = stage / 'protected-identity.json'
        receipt = {'path': str(receipt_path), 'sha256': digest(receipt_path)}
        from experiments.tennis.w8_receiver_preflight import PROTECTED
        identities = json.loads(receipt_path.read_text())['protected_inputs']
        if set(identities) != set(PROTECTED) or any(
            identities[k].get('path') != v[0] or identities[k].get('sha256') != v[1]
            or type(identities[k].get('device')) is not int or type(identities[k].get('inode')) is not int
            for k, v in PROTECTED.items()
        ):
            raise ValueError('portable protected identity receipt mismatch')
        expected = normalize_portable(original, source, output, aliases, manifest['gpu_uuid'],
                                      config['parent_user_namespace'], config['parent_network_namespace'], receipt)
    else:
        expected = normalize(original, source, output, aliases, manifest['gpu_uuid'],
                             config['parent_user_namespace'], config['parent_network_namespace'])
    if expected != config:
        raise ValueError('configuration differs from the approved normalized profile')
    arms = summary['arms']
    complete = len(arms) == 2 and [x['condition'] for x in arms] == ['generic', 'isolated']
    passed = complete
    for arm in arms:
        if not arm.get('admitted'):
            complete = passed = False
            continue
        name = arm['condition']
        if json.loads((stage / f'{name}.exit.json').read_text()) != arm:
            raise ValueError('exit receipt mismatch')
        if any(digest(stage / f'{name}.{suffix}') != arm[f'{suffix}_sha256'] for suffix in ('stdout', 'stderr')):
            raise ValueError('output receipt mismatch')
        if arm['timed_out']:
            complete = passed = False
            continue
        try:
            result = json.loads((stage / f'{name}.stdout').read_text())
        except (json.JSONDecodeError, OSError):
            complete = passed = False
            continue
        if result.get('condition') != name:
            raise ValueError('worker condition mismatch')
        if arm['status'] != 0:
            if (result.get('status') != 'failed' or not result.get('error')
                    or result.get('phase') != 'cuda_probe'):
                complete = False
            if name == 'isolated':
                try:
                    check_isolation(result.get('isolation', {}), json.loads((stage / 'worker-config.json').read_text()))
                except (ValueError, KeyError, TypeError):
                    complete = False
            passed = False
            continue
        if result.get('status') != 'passed' or result.get('cuda', {}).get('sum') != 1048576.0:
            raise ValueError('invalid successful worker result')
        cuda = result.get('cuda', {})
        if (cuda.get('visible_devices') != manifest['gpu_uuid'] or cuda.get('device_count') != 1
                or cuda.get('name') != manifest['gpu_before']['name']
                or not isinstance(cuda.get('capability'), list) or len(cuda['capability']) != 2
                or not cuda.get('torch') or not cuda.get('torch_cuda')
                or abs(cuda.get('total_memory', 0) - manifest['gpu_before']['memory_mib'] * 1048576) > 16 * 1048576
                or (cuda.get('uuid') and cuda['uuid'].lower() != manifest['gpu_uuid'].lower())):
            raise ValueError('worker physical GPU or runtime identity mismatch')
        if name == 'isolated':
            check_isolation(result.get('isolation', {}), json.loads((stage / 'worker-config.json').read_text()))
    verdict = {'measurement_complete': complete, 'both_cuda_passed': passed,
               'generation_gate_passed': False}
    validation = stage / 'validation.json'
    if validation.exists():
        if json.loads(validation.read_text()) != verdict:
            raise ValueError('existing validation differs')
    else:
        write_json(validation, verdict)
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--stage')
    p.add_argument('--phase', choices=('smoke', 'full'), required=True)
    p.add_argument('--profile', choices=(PROFILE, PORTABLE_PROFILE), default=PROFILE)
    p.add_argument('--original-config', default=str(Path(__file__).with_name('w8_cuda_historical_config.json')))
    p.add_argument('--source-archive', required=True)
    p = sub.add_parser('worker')
    p.add_argument('--condition', choices=('generic', 'isolated'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--source', required=True)
    p = sub.add_parser('validate')
    p.add_argument('--stage')
    args = parser.parse_args()
    if args.command == 'worker':
        return run_worker(args)
    if args.command == 'validate':
        stage = Path(args.stage) if args.stage else Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-diagnostic'
        status = validate(stage)
        if os.environ.get('PS_VALIDATION_PATH'):
            verdict = json.loads((stage / 'validation.json').read_text())
            write_json(Path(os.environ['PS_VALIDATION_PATH']), {
                'passed': status == 0, 'checks': {'paired_measurement_complete': verdict['measurement_complete'],
                'both_cuda_passed': verdict['both_cuda_passed'], 'generation_gate_passed': False}})
        return status
    if os.environ.get('PS_STAGE') != args.phase:
        raise ValueError('fleet phase mismatch')
    if args.phase == 'full':
        return 0
    if args.stage is None:
        args.stage = str(Path(os.environ['PS_STAGE_DIR']) / 'w8-cuda-diagnostic')
        Path(args.stage).mkdir()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
