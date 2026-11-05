"""CPU-only wrapper for the strict W8 packet decoder; no generator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.tennis import w8_strict_receiver_worker as receiver


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--loader', type=Path, required=True)
    p.add_argument('--loader-sha256', required=True)
    p.add_argument('--canary', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    if receiver.digest(args.config) != args.config_sha256:
        raise ValueError('packet worker config identity mismatch')
    config = json.loads(args.config.read_text())
    if config.get('worker_kind') != 'packet' or config.get('environment', {}).get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('packet-only worker must explicitly hide CUDA')
    receiver.launch(args.config, args.config_sha256, args.loader, args.loader_sha256, args.canary, args.source_root)
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('packet decoder initialized or exposed CUDA')
    receipt = {'cuda_visible_devices': '', 'cuda_initialized_after_decode': False,
               'torch_version': str(torch.__version__), 'scope': 'CPU packet decoder only'}
    with (Path(config['output']) / 'cpu-only.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
