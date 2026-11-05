"""Strict host-cost model worker with a child-observed GPU identity receipt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.tennis import w8_strict_receiver_worker as receiver
from experiments.tennis import w8_cuda_system_leaves as policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--loader', type=Path, required=True)
    parser.add_argument('--loader-sha256', required=True)
    parser.add_argument('--canary', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    if receiver.digest(args.config) != args.config_sha256:
        raise ValueError('model worker config identity mismatch')
    config = json.loads(args.config.read_text())
    if config.get('worker_kind') != 'cost_probe':
        raise ValueError('host model wrapper only admits bounded cost probes')
    receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root)
    import torch
    if not torch.cuda.is_initialized() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one initialized CUDA device required')
    props = torch.cuda.get_device_properties(0)
    observed = str(getattr(props, 'uuid', ''))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (policy.canonical_gpu_uuid(observed) != policy.canonical_gpu_uuid(visible)
            or (props.major, props.minor) != (8, 9) or 'RTX 6000 Ada' not in props.name):
        raise ValueError('observed CUDA GPU does not match the selected Ada')
    output = Path(config['output'])
    result = {'visible_devices': visible, 'device_uuid': observed, 'name': props.name,
              'major': props.major, 'minor': props.minor, 'sealed_sha256': receiver.digest(output / 'sealed.json'),
              'config_sha256': args.config_sha256, 'cuda_device_count': 1}
    with (output / 'gpu-identity.json').open('x') as f:
        json.dump(result, f, sort_keys=True)
        f.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
