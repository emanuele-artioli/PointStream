"""CPU-only wrapper for the strict W8 packet decoder; no generator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.tennis import w8_strict_receiver_worker as receiver


def assert_cpu_state(state, is_tensor):
    groups = ('reference_unet', 'denoising_unet', 'pose_guider')
    if not isinstance(state, dict) or any(not isinstance(state.get(group), dict) for group in groups):
        raise ValueError('decoded state groups missing')
    tensors = [value for group in groups for value in state[group].values()]
    if len(tensors) != 480 or any(not is_tensor(value) or value.device.type != 'cpu' for value in tensors):
        raise ValueError('decoded state contains non-CPU tensor or wrong count')
    return len(tensors)


def assert_cpu_units(units, names, ndarray_type):
    if not isinstance(units, dict) or set(units) != set(names):
        raise ValueError('decoded units differ')
    count = 0
    for values in units.values():
        if not isinstance(values, dict) or not values:
            raise ValueError('decoded unit is empty')
        for value in values.values():
            if not isinstance(value, ndarray_type):
                raise ValueError('decoded unit contains non-CPU array')
            count += 1
    return count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--loader', type=Path, required=True)
    p.add_argument('--loader-sha256', required=True)
    p.add_argument('--canary', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--witness-mode', choices=('packet',))
    args = p.parse_args()
    if receiver.digest(args.config) != args.config_sha256:
        raise ValueError('packet worker config identity mismatch')
    config = json.loads(args.config.read_text())
    if config.get('worker_kind') != 'packet' or config.get('environment', {}).get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('packet-only worker must explicitly hide CUDA')
    _, fixed, _, _ = receiver.preload()
    from experiments.tennis import w8_payload
    import numpy as np
    import torch
    observed = {}
    original_decode = fixed.decode_frozen_window
    original_unpack = w8_payload.unpack_match
    original_restore = w8_payload.restore_tensor

    def decode_checked(*args, **kwargs):
        arrays, bindings, match = original_decode(*args, **kwargs)
        observed['unit_arrays'] = assert_cpu_units(arrays, fixed.UNITS, np.ndarray)
        return arrays, bindings, match

    def restore_checked(*args, **kwargs):
        value = original_restore(*args, **kwargs)
        if not torch.is_tensor(value) or value.device.type != 'cpu':
            raise ValueError('restored tensor is not CPU before arithmetic')
        observed['restored_tensors'] = observed.get('restored_tensors', 0) + 1
        return value

    def unpack_checked(*args, **kwargs):
        base = args[3] if len(args) > 3 else kwargs['guider_base']
        if not isinstance(base, dict) or any(not torch.is_tensor(v) or v.device.type != 'cpu' for v in base.values()):
            raise ValueError('guider base contains non-CPU tensor')
        state = original_unpack(*args, **kwargs)
        observed['state_tensors'] = assert_cpu_state(state, torch.is_tensor)
        return state

    fixed.decode_frozen_window = decode_checked
    w8_payload.unpack_match = unpack_checked
    w8_payload.restore_tensor = restore_checked
    if args.witness_mode == 'packet':
        receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root,
                        witness_mode='packet')
    else:
        receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root)
    if observed.get('state_tensors') != 480 or observed.get('unit_arrays', 0) <= 0 or observed.get('restored_tensors', 0) <= 0:
        raise RuntimeError('pre-serialization CPU checks missing')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('packet decoder initialized or exposed CUDA')
    receipt = {'cuda_visible_devices': '', 'cuda_initialized_after_decode': False,
               'torch_version': str(torch.__version__), 'pre_serialization': observed, 'scope': 'CPU packet decoder only'}
    with (Path(config['output']) / 'cpu-only.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
