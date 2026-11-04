"""Building blocks for a prospective W8 CUDA diagnostic, not a fleet entrypoint.

A future reviewed supervisor must verify the frozen source/config/runtime,
launch each arm in a fresh process, and enforce all resource limits. Merely
importing this module does not import torch or initialize CUDA.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import resource
from typing import Any, Callable


def policy_fingerprint(config: dict[str, Any]) -> str:
    """Bind the complete allow and denial lists; do not normalize away aliases."""
    policy = {key: config[key] for key in ('allow_paths', 'denied_paths')}
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_isolation(receipt: dict[str, Any], config: dict[str, Any]) -> None:
    """Require the actual bootstrap receipt, including every protected alias."""
    expected = {path for paths in config['denied_paths'].values() for path in paths}
    denials = receipt.get('denials')
    cpu = receipt.get('cpu_bootstrap', {})
    namespaces = receipt.get('namespaces', {})
    if (not expected or not isinstance(denials, dict) or set(denials) != expected
            or any(code not in (errno.EACCES, errno.EPERM) for code in denials.values())):
        raise ValueError('incomplete or invalid file-denial receipt')
    if (type(receipt.get('landlock_abi')) is not int or receipt['landlock_abi'] < 1
            or receipt.get('network_interfaces') != ['lo'] or receipt.get('pre_sandbox_threads') != 1
            or receipt.get('extra_fds') is not False
            or not cpu.get('before_landlock_sha256')
            or cpu.get('before_landlock_sha256') != cpu.get('after_landlock_sha256')
            or cpu.get('cuda_initialized_before_isolation') is not False):
        raise ValueError('invalid isolation or CPU bootstrap receipt')
    if (not namespaces.get('user') or not namespaces.get('net')
            or namespaces['user'] == config['parent_user_namespace']
            or namespaces['net'] == config['parent_network_namespace']):
        raise ValueError('namespace identity missing or unchanged')


def cuda_probe(torch: Any) -> dict[str, Any]:
    """Probe one actual visible device using only a tiny deterministic tensor."""
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f'expected exactly one visible CUDA device, got {count}')
    props = torch.cuda.get_device_properties(0)
    device = torch.device('cuda:0')
    x = torch.arange(1024, dtype=torch.float32, device=device)
    y = x * 2 + 1
    torch.cuda.synchronize(device)
    if x.device.type != 'cuda' or y.device.type != 'cuda':
        raise RuntimeError('synthetic tensor did not execute on CUDA')
    if not bool(torch.isfinite(y).all().item()) or float(y.sum().item()) != 1048576.0:
        raise RuntimeError('synthetic CUDA kernel produced an invalid result')
    return {'device_count': count, 'name': props.name, 'capability': [props.major, props.minor],
            'total_memory': props.total_memory, 'uuid': str(getattr(props, 'uuid', '')),
            'tensor_bytes': 8192, 'sum': 1048576.0,
            'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def run_condition(condition: str, config: dict[str, Any], *,
                  bootstrap: Callable[[dict[str, Any]], dict[str, Any]],
                  probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one already-admitted arm; the caller must start a fresh process."""
    if condition not in ('generic', 'isolated'):
        raise ValueError('invalid condition')
    receipt: dict[str, Any] = {'condition': condition, 'status': 'failed',
                               'policy_sha256': policy_fingerprint(config),
                               'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        if condition == 'isolated':
            isolation = bootstrap(config)
            validate_isolation(isolation, config)
            receipt['isolation'] = isolation
        result = probe()
        if (not isinstance(result, dict) or result.get('device_count') != 1
                or not result.get('name') or not result.get('torch') or not result.get('torch_cuda')
                or result.get('sum') != 1048576.0 or result.get('tensor_bytes') != 8192):
            raise ValueError('invalid CUDA probe receipt')
        receipt['cuda'] = result
        receipt['status'] = 'passed'
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        receipt['usage'] = {'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
                            'max_rss': usage.ru_maxrss}
    return receipt
