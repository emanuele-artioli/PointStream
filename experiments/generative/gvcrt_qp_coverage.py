"""CPU-only checkpoint/table qualification for registered GVC endpoint QPs."""

import argparse
import json
import os
from pathlib import Path

from experiments.generative import gvcrt_native_bridge as bridge


def check_tables(i_model, p_model, torch):
    if i_model.get_qp_num() != 10 or p_model.get_qp_num() != 10:
        raise ValueError('unexpected upstream QP count')
    sizes = {}
    for role, model, names, size in (
        ('I', i_model, ('q_scale_enc', 'q_scale_dec', 'q_scale_recon'), 10),
        ('P', p_model, ('q_scale_enc', 'q_scale_dec', 'q_scale_feature', 'q_scale_recon'), 12),
    ):
        for name in names:
            table = getattr(model, name)
            if table.shape[0] != size or not bool(torch.isfinite(table).all()):
                raise ValueError(f'invalid {role} {name} table')
            sizes[f'{role}.{name}'] = list(table.shape)
    placements = {}
    for qp in (0, 3, 9):
        shifted = [int(p_model.shift_qp(qp, index)) for index in (0, 1, 2)]
        if shifted != [qp, qp + 2, qp + 1] or min(shifted) < 0 or max(shifted) >= 12:
            raise ValueError('unexpected P shift policy')
        placements[str(qp)] = shifted
    return {'shapes': sizes, 'p_shift_indices': placements}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--i-checkpoint', type=Path, required=True)
    parser.add_argument('--p-checkpoint', type=Path, required=True)
    parser.add_argument('--i-sha256', required=True)
    parser.add_argument('--p-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('CUDA must be explicitly hidden')
    args.disable_fused = True
    torch, i_model, p_model, receipt = bridge.models(args, 'cpu')
    if not all(x['strict_complete_coverage'] for x in receipt['coverage']):
        raise ValueError('incomplete checkpoint coverage')
    receipt['tables'] = check_tables(i_model, p_model, torch)
    receipt['cuda_visible_devices'] = os.environ['CUDA_VISIBLE_DEVICES']
    with args.output.open('x') as out:
        json.dump(receipt, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
