"""Opt-in GVC scorer wrapper; original metric/judge calculations are unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gvcrt_device_guard as guard
import gvcrt_peer_scorer as original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'decoded', 'labels', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, choices=(4, 16), required=True)
    parser.add_argument('--expected-uuid', required=True)
    args = parser.parse_args()
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != args.expected_uuid:
        raise ValueError('scorer CUDA visibility differs from fleet claim')
    # Initializes only after the separate isolated probe has passed in parent.
    if not torch.cuda.is_available():
        raise ValueError('CUDA provider unavailable')
    torch.cuda.init()
    before = guard.observe(torch, visible, args.expected_uuid)
    final = args.output
    temporary = final.with_name(final.name + '.guarded-partial')
    if final.exists() or temporary.exists():
        raise FileExistsError('scorer outputs must be fresh')
    args.output = temporary
    original.score(args)
    result = json.loads(temporary.read_text())
    after = guard.observe(torch, os.environ.get('CUDA_VISIBLE_DEVICES', ''), args.expected_uuid)
    if before != after:
        raise ValueError('scorer CUDA device changed')
    # original.score returns only after explicitly checking every metric/judge
    # model's first parameter is on CUDA and after completing all metric operations.
    device = {**after, 'provider': 'torch_cuda', 'model_first_parameters_on_cuda': True}
    guard.check_device(device, args.expected_uuid, scorer=True)
    result['device'] = device
    with final.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
