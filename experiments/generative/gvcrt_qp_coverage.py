"""CPU-only checkpoint/table qualification for registered GVC endpoint QPs."""

import argparse
import json
import os
import resource
import sys
import tempfile
import time
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


def device_inventory(i_model, p_model):
    result = {}
    for role, model in (('I', i_model), ('P', p_model)):
        for kind, values in (('parameters', model.named_parameters()), ('buffers', model.named_buffers())):
            seen = list(values)
            devices = sorted({str(value.device) for _, value in seen})
            if any(value.device.type != 'cpu' for _, value in seen):
                raise ValueError(f'{role} {kind} contain non-CPU tensors')
            result[f'{role}.{kind}'] = {'count': len(seen), 'devices': devices}
    if not result['I.parameters']['count'] or not result['P.parameters']['count']:
        raise ValueError('missing model parameters')
    return result


def usage():
    return {name: {'user_seconds': value.ru_utime, 'system_seconds': value.ru_stime,
                   'max_rss': value.ru_maxrss} for name, value in (
        ('self', resource.getrusage(resource.RUSAGE_SELF)),
        ('children', resource.getrusage(resource.RUSAGE_CHILDREN)))}


def publish_once(output, receipt):
    # Hard-link publication cannot overwrite an existing result; remove only our temp.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix='.qp-proof-',
                                         delete=False) as out:
            temporary = Path(out.name)
            json.dump(receipt, out, sort_keys=True, indent=2, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    started = time.monotonic()
    usage_before = usage()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before model loading')
    args.disable_fused = True
    torch, i_model, p_model, receipt = bridge.models(args, 'cpu')
    if not all(x['strict_complete_coverage'] for x in receipt['coverage']):
        raise ValueError('incomplete checkpoint coverage')
    receipt['tables'] = check_tables(i_model, p_model, torch)
    receipt['devices'] = device_inventory(i_model, p_model)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized during CPU qualification')
    source_manifest = args.official_repo / '.gvcrt-source.json'
    if not source_manifest.is_file():
        raise ValueError('hashed source export manifest required')
    receipt['cpu_proof'] = {
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'cuda_initialized_before': False, 'cuda_initialized_after': False,
        'interpreter': str(Path(sys.executable).resolve(strict=True)),
        'interpreter_sha256': bridge.sha256(Path(sys.executable).resolve(strict=True)),
        'pid': os.getpid(), 'pgid': os.getpgid(0), 'affinity': sorted(os.sched_getaffinity(0)),
        'source_manifest_sha256': bridge.sha256(source_manifest),
        'helper_sha256': bridge.sha256(Path(__file__).resolve(strict=True)),
        'elapsed_seconds': time.monotonic() - started,
        'usage_before': usage_before, 'usage_after': usage(),
    }
    if len(receipt['cpu_proof']['affinity']) != 1:
        raise ValueError('qualification requires one CPU affinity')
    publish_once(args.output, receipt)


if __name__ == '__main__':
    main()
