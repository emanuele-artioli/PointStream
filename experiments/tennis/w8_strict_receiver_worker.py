"""Source-hidden strict-policy packet or bounded generation worker.

A supervisor must authenticate and stage all inputs and a minimal code closure.
This direct-script worker does not schedule or admit a fleet campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import socket
import sys
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preload() -> tuple[Any, Any, Any, Any]:
    """Import with no legacy positional command that could change CPU limits."""
    prior = sys.argv
    try:
        sys.argv = [prior[0]]
        from experiments.tennis import w8_cuda_system_leaves as policy
        from experiments.tennis import w8_input_job as fixed
        from experiments.tennis import w8_receiver_bootstrap as bootstrap
        from experiments.tennis import w8_receiver_generate as receiver
        return policy, fixed, bootstrap, receiver
    finally:
        sys.argv = prior


def validate_source_origins(source_root: Path) -> dict[str, str]:
    """Reject checkout or site-package shadowing before source-hidden isolation."""
    root = source_root.resolve(strict=True)
    if Path.cwd().resolve(strict=True) != root or not Path(__file__).resolve(strict=True).is_relative_to(root):
        raise ValueError('worker is not executing from authenticated source root')
    origins: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == 'experiments' or name.startswith('experiments.') or name == 'src'
                or name.startswith('src.') or name == 'tools' or name.startswith('tools.')):
            continue
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('project module outside authenticated source root: ' + name)
            origins[name] = str(path.relative_to(root))
        for package_path in getattr(module, '__path__', ()):
            if not Path(package_path).resolve(strict=True).is_relative_to(root):
                raise ValueError('project package outside authenticated source root: ' + name)
    if not origins or 'experiments.tennis.w8_receiver_generate' not in origins:
        raise ValueError('required receiver source closure not loaded')
    return origins


def clean_process() -> None:
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('pre-isolation process is not single-threaded')
    if any(int(fd) > 2 and Path('/proc/self/fd', fd).exists() for fd in os.listdir('/proc/self/fd')):
        raise RuntimeError('unexpected open descriptor')


def cpu_runtime_identity(torch: Any) -> dict[str, str]:
    """Report the actual executing CPU leaf runtime without initializing CUDA."""
    return {'executable': str(Path(sys.executable).resolve(strict=True)),
            'prefix': str(Path(sys.prefix).resolve(strict=True)), 'torch': str(torch.__version__)}


def launch(config_path: Path, config_sha: str, loader_path: Path, loader_sha: str,
           canary: Path, source_root: Path, *, materialize_profile: bool = False,
           clock_receipt: Path | None = None, clock_receipt_sha: str | None = None,
           witness_mode: str | None = None) -> None:
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError('Linux and one CPU required')
    policy, fixed, bootstrap, receiver = preload()
    finish_modes = ('context_rgb', 'context_sam', 'context_compose', 'rgb', 'sam', 'compose', 'isolation_proof')
    if ((clock_receipt is None) != (clock_receipt_sha is None)
            or (clock_receipt is not None and not materialize_profile
                and witness_mode not in finish_modes)):
        raise ValueError('clocksource profile requires a bound receipt and materialized model worker')
    clock_profile: Any = None
    if clock_receipt is not None:
        from experiments.tennis import w8_clock_profile as clock_profile_module
        clock_profile = clock_profile_module
    if witness_mode in finish_modes and witness_mode != 'isolation_proof':
        from experiments.tennis import w8_receiver_finish as finish
    source_origins = validate_source_origins(source_root)
    config = bootstrap.load_config(config_path, config_sha)
    if Path(config['allow_paths'][0]).resolve(strict=True) != source_root.resolve(strict=True):
        raise ValueError('config source grant differs from authenticated source root')
    allowed_kinds = ('packet', 'cost_probe') if witness_mode is None else (witness_mode,)
    if witness_mode not in (None, 'packet', 'generate', 'context_qualification', *finish_modes) or config['worker_kind'] not in allowed_kinds:
        raise ValueError('only packet decode and bounded cost probe are admitted')
    if witness_mode in ('generate', 'context_qualification') and (not materialize_profile or clock_receipt is None):
        raise ValueError('witness modes require materialized source and clock profile')
    if witness_mode in finish_modes and (materialize_profile or clock_receipt is None):
        raise ValueError('qualification finish requires clock profile and no model materialization')
    if witness_mode == 'packet' and (materialize_profile or clock_receipt is not None):
        raise ValueError('witness packet must remain CPU-only and unmaterialized')
    if witness_mode == 'isolation_proof' and config['environment'].get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('isolation proof requires hidden CUDA')
    if config['worker_kind'] == 'cost_probe' and config.get('probe_steps') not in (1, 3):
        raise ValueError('only 1/3 step qualification runs are admitted')
    if digest(loader_path) != loader_sha:
        raise ValueError('loader receipt identity mismatch')
    loader = json.loads(loader_path.read_text())
    if loader.get('uid') != 0 or loader.get('mode', 0) & 2:
        raise ValueError('loader is not host-root-owned and non-world-writable')
    output = Path(config['output']).resolve(strict=True)
    canary = canary.resolve(strict=True)
    names = {'write', 'truncate', 'remove', 'rename', 'link'}
    if (not canary.is_dir() or canary.parent != output.parent or canary == output
            or {x.name for x in canary.iterdir()} != names
            or any((canary / name).is_symlink() or (canary / name).read_bytes() != b'canary' for name in names)):
        raise ValueError('fresh task-owned canaries required')
    namespaces = {name: os.readlink('/proc/self/ns/' + name) for name in ('user', 'net')}
    if (namespaces['user'] == config['parent_user_namespace'] or namespaces['net'] == config['parent_network_namespace']
            or sorted(name for _, name in socket.if_nameindex()) != ['lo']):
        raise ValueError('fresh user and network namespaces required')
    clean_process()
    allowed = [Path(p).resolve(strict=True) for p in config['allow_paths']]
    denied = bootstrap.flatten_denials(config['denied_paths'])
    for path in denied:
        actual = path.resolve(strict=True)
        if any(actual == root or actual.is_relative_to(root) for root in allowed):
            raise ValueError('protected path in allowed roots')
    if any(canary == root or canary.is_relative_to(root) for root in allowed):
        raise ValueError('canary must be outside allowed roots')
    extra, leaves = policy.approved_leaves(denied, loader)
    if witness_mode is not None:
        with (output.parent / 'self-leaves.json').open('x') as stream:
            json.dump(leaves, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    allowed.extend(extra)
    clock_identity = None
    if clock_profile is not None:
        assert clock_receipt is not None and clock_receipt_sha is not None
        clock_paths, clock_identity = clock_profile.prepare_child(clock_receipt, clock_receipt_sha, denied)
        allowed.extend(clock_paths)
    cpu_seconds = config['cpu_seconds']
    if type(cpu_seconds) is not int or not 1 <= cpu_seconds <= 3600:
        raise ValueError('invalid worker CPU limit')
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    os.environ.clear()
    os.environ.update(config['environment'])
    os.environ.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'NUMEXPR_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    if witness_mode is not None:
        import tempfile
        tempfile.tempdir = None
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized before isolation')
    before = fixed.synthetic_torch_bootstrap(torch)
    clean_process()
    sys.addaudithook(bootstrap._deny_processes)
    abi, mask = (clock_profile.restrict(policy, allowed, output) if clock_profile is not None
                 else policy.restrict(allowed, output))
    mutation = policy.mutation_probes(canary, output)
    probes = bootstrap._check_denials(denied, open_only=witness_mode is not None)
    after = fixed.synthetic_torch_bootstrap(torch)
    if before != after or torch.cuda.is_initialized():
        raise RuntimeError('post-isolation CPU warmup failed')
    isolation = {'landlock_abi': abi, 'handled_mask': mask, 'denials': probes, 'namespaces': namespaces,
                 'network_interfaces': ['lo'], 'pre_sandbox_threads': 1, 'extra_fds': False,
                 'cpu_bootstrap': {'before_landlock_sha256': before, 'after_landlock_sha256': after,
                                   'cuda_initialized_before_isolation': False},
                 'approved_leaves': leaves, 'mutation_denials': mutation, 'output_write_ok': True,
                 'loader_identity_sha256': loader_sha, 'source_origins': source_origins,
                 'policy': 'w8-approved-system-leaves-mutation-v1'}
    if clock_identity is not None:
        isolation['clock_profile'] = clock_identity
    if config['worker_kind'] == 'packet':
        if materialize_profile:
            raise ValueError('materialization profile is not valid for packet workers')
        receiver.run_packet_isolated(config, isolation)
    elif witness_mode == 'isolation_proof':
        if torch.cuda.is_initialized():
            raise ValueError('isolation proof initialized CUDA')
        with (output / 'isolation-proof.json').open('x') as stream:
            json.dump({'scope': 'cpu_isolation_proof_only', 'isolation': isolation,
                       'cuda_initialized': False, 'visible_devices': '',
                       'runtime_identity': cpu_runtime_identity(torch)}, stream, sort_keys=True)
    elif witness_mode in finish_modes:
        finish_funcs: dict[str, Any] = {'context_rgb': finish.run_conversion_isolated, 'context_sam': finish.run_sam_isolated,
                        'context_compose': finish.run_compose_isolated, 'rgb': finish.run_conversion_isolated,
                        'sam': finish.run_sam_isolated, 'compose': finish.run_compose_isolated}
        if witness_mode.startswith('context_'):
            finish_funcs[witness_mode](config, isolation, qualification=True)
        else:
            finish_funcs[witness_mode](config, isolation, authenticated_full=True)
    elif materialize_profile:
        receiver.run_isolated(config, isolation, materialize_profile=True)
    else:
        receiver.run_isolated(config, isolation)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--loader', type=Path, required=True)
    p.add_argument('--loader-sha256', required=True)
    p.add_argument('--canary', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--witness-mode', choices=('packet', 'context_rgb', 'context_sam', 'context_compose', 'rgb', 'sam', 'compose', 'isolation_proof'))
    p.add_argument('--clock-receipt', type=Path)
    p.add_argument('--clock-receipt-sha256')
    args = p.parse_args()
    launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root,
           witness_mode=args.witness_mode, clock_receipt=args.clock_receipt, clock_receipt_sha=args.clock_receipt_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
