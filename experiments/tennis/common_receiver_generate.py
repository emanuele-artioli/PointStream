"""Reconstruct common-tennis player conditioning and all crop ticks from a paid packet."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b_generate as generator
from experiments.tennis import player_requalification as replay
from experiments.tennis import w8_cuda_system_leaves as system
from experiments.tennis import w8_public_folder_materialize as folders
from experiments.tennis.w8_receiver_generate import FIT_CODE_SHA, GENERATOR_CODE_SHA


def conditioning(carrier: Path, expected_sha256: str, frames: int, units: list[str]):
    """Yield one production Renderer input at a time, including its packet-defined seed."""
    if carrier.stat().st_size > 4 * 1024 * 1024 or frames not in (24, 44, 240):
        raise ValueError('unsupported development window or oversized carrier')
    blob = carrier.read_bytes()
    if packet.wire.sha(blob) != expected_sha256:
        raise ValueError('wrong development carrier')
    sections, roles, decoded = packet.unpack(blob)
    if sum(roles.values()) != len(blob) or not sections['base'] or roles['match']:
        raise ValueError('carrier accounting')
    if not 1 <= len(units) <= 2 or len(set(units)) != len(units) or any(u not in decoded for u in units):
        raise ValueError('invalid selected player units')
    for name in units:
        runs = decoded[name]
        if len(runs) != 1 or not np.array_equal(runs[0]['t'], np.arange(240)):
            raise ValueError('requires continuous full-rally player')
        run = runs[0]
        poses = np.empty((frames, 784, 512, 3), np.uint8)
        for tick in range(frames):
            poses[tick] = packet.pose_for_tick(run, tick)
        arrays = {'reference': run['reference'], 'pose': poses, 't': run['t'][:frames],
                  'box': np.asarray(run['boxes'][:frames])}
        yield name, run['seed'], arrays


def write_record(path: Path, data: dict) -> None:
    with path.open('x') as f:
        json.dump(data, f, sort_keys=True)
        f.write('\n')


def run(carrier: Path, expected_sha256: str, weights: dict[str, Path], fit: Path | None, output: Path, *,
        frames: int, units: list[str], isolation: dict, conditioning_only: bool = False) -> dict:
    """Called by the isolated launcher. Local hash comparisons stay outside the receiver."""
    import cv2

    if (isolation.get('landlock_abi', 0) < 3 or not isolation.get('denials')
            or any(x not in (1, 13) for x in isolation['denials'].values()) or isolation.get('interfaces') != ['lo']):
        raise ValueError('missing measured receiver containment')
    output.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    records: list[dict[str, Any]] = []
    record: dict[str, Any] = {'carrier_sha256': expected_sha256, 'isolation': isolation,
                              'frames': frames, 'outputs': records, 'conditioning_only': conditioning_only}
    renderer = None
    copies = None
    scratch = output / 'model-scratch'
    if not conditioning_only:
        import torch
        from experiments.audit.env_smoke import model_on_cuda, profile_cuda
        from experiments.tennis import k6b_fit

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one CUDA GPU required')
        props = torch.cuda.get_device_properties(0)
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        if (system.canonical_gpu_uuid(str(props.uuid)) != system.canonical_gpu_uuid(visible)
                or (props.major, props.minor) != (8, 9) or 'RTX 6000 Ada' not in props.name):
            raise RuntimeError('assigned GPU differs from qualified Ada')
        limit = int(os.environ['PS_GPU_MEMORY_LIMIT_MIB'])
        if limit <= 0:
            raise ValueError('missing memory budget')
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(min(1., limit * 2**20 / props.total_memory))
        if (fit is None or generator.sha256(fit) != replay.BASE_SHA or set(weights) != set(replay.PINNED_WEIGHTS)
                or generator.sha256(Path(generator.__file__)) != GENERATOR_CODE_SHA or generator.sha256(Path(k6b_fit.__file__)) != FIT_CODE_SHA):
            raise ValueError('common model or renderer identity')
        file_records = {f'weight:{k}': {'path': str(v), 'sha256': generator.sha256(v)} for k,v in weights.items()}
        if any(file_records['weight:'+k]['sha256'] != v for k,v in replay.PINNED_WEIGHTS.items()):
            raise ValueError('original model weights differ')
        scratch.mkdir()
        copies = folders.materialize_public(file_records, scratch / 'folders', replay.PINNED_WEIGHTS)
        start = time.monotonic()
        renderer = generator.Renderer(weights, 'original', 'cuda', scratch, fit=fit, base=None, bits=0, cfg=3.5)
        if not all(model_on_cuda(m) for m in renderer.modules.values()):
            raise RuntimeError('renderer left a module off CUDA')
        replay.check_model_record(renderer.record, 'tennis_only', fit)
        record.update(device=props.name, gpu_uuid=visible, cuda=str(torch.version.cuda), torch=str(torch.__version__),
                      model_record=renderer.record, model_load_seconds=time.monotonic()-start, weight_hashes=replay.PINNED_WEIGHTS,
                      common_fit_sha256=replay.BASE_SHA, gpu_memory_limit_mib=limit)
    for name, seed, data in conditioning(carrier, expected_sha256, frames, units):
        row = {'unit': name, 'seed': seed, 'array_sha256': {k: replay.array_sha(a) for k,a in data.items()}}
        if renderer is not None:
            start = time.monotonic()
            crops, kernels = profile_cuda(lambda: renderer.render(data['reference'], data['pose'], 30, seed))
            if not replay.complete_crops(crops, frames) or kernels['kernel_launches'] <= 0:
                raise RuntimeError('incomplete, degenerate or non-CUDA generation')
            path = output / f'{name}.npz'
            np.savez_compressed(path, t=data['t'], box=data['box'], crops=crops)
            row.update(render_and_save_seconds=time.monotonic()-start, kernels=kernels, crops_file=path.name,
                       crops_file_sha256=generator.sha256(path), shape=list(crops.shape),
                       distinct_frames=len({replay.array_sha(c) for c in crops}))
            del crops
        records.append(row)
    if copies is not None:
        folders.cleanup_completed(scratch/'folders', copies)
        record.update(gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
    write_record(output / ('conditioning.json' if conditioning_only else 'generation.json'), record)
    return record
