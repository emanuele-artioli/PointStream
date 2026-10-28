"""Finite paid long/short decoded-reference pilot; not a neural-codec bound."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Any
import zlib

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, SOURCE_YUV, TOOLS, read
from experiments.tennis.k6b_ideal_player import WEIGHTS, digest, image_sha, write
from experiments.tennis.parallel_background import pack_residual, reconstruct
from experiments.tennis.parallel_motion import decode_motion, fit_motion, warp_reference

W, H, BLOCK = 960, 544, 32
CRFS = (43, 53, 63)


def resize_plane(frame: np.ndarray, width: int, height: int, target_w: int, target_h: int) -> np.ndarray:
    uv = frame[height:].reshape(2, height//2, width//2)
    out = [cv2.resize(frame[:height], (target_w, target_h), interpolation=cv2.INTER_CUBIC).ravel()]
    out.extend(cv2.resize(p, (target_w//2, target_h//2), interpolation=cv2.INTER_CUBIC).ravel() for p in uv)
    return np.concatenate(out).reshape(target_h*3//2, target_w)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Encoder-only boxes; delivered coordinates, not decoder mask knowledge."""
    boxes = np.zeros((*masks.shape[:2], 4), dtype=np.int32)
    for t in range(len(masks)):
        for p in range(masks.shape[1]):
            yy, xx = np.nonzero(cv2.resize(masks[t, p].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST))
            if len(xx):
                x, y = int(xx.min())//2*2, int(yy.min())//2*2
                x1, y1 = min(W, (int(xx.max())+2)//2*2), min(H, (int(yy.max())+2)//2*2)
                boxes[t, p] = [x, y, x1-x, y1-y]
    return boxes


def object_history(previous: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    result = previous.copy()
    src_uv = previous[H:].reshape(2, H//2, W//2)
    dst_uv = result[H:].reshape(2, H//2, W//2)
    for before, after in zip(old, new):
        x, y, w, h = (int(v) for v in before)
        nx, ny, nw, nh = (int(v) for v in after)
        if min(w, h, nw, nh) <= 0:
            continue
        result[ny:ny+nh, nx:nx+nw] = cv2.resize(previous[y:y+h, x:x+w], (nw, nh), interpolation=cv2.INTER_LINEAR)
        for i in range(2):
            dst_uv[i, ny//2:(ny+nh)//2, nx//2:(nx+nw)//2] = cv2.resize(
                src_uv[i, y//2:(y+h)//2, x//2:(x+w)//2], (nw//2, nh//2), interpolation=cv2.INTER_LINEAR)
    return result


def choose_blocks(source: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
    errors = []
    for candidate in candidates:
        error = (source[:H].astype(np.float32)-candidate[:H].astype(np.float32))**2
        errors.append(error.reshape(H//BLOCK, BLOCK, W//BLOCK, BLOCK).mean((1, 3)))
    return np.argmin(np.stack(errors), axis=0).astype(np.uint8)


def decode_blocks(candidates: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    if labels.shape != (H//BLOCK, W//BLOCK) or np.any(labels >= len(candidates)):
        raise ValueError('Invalid delivered reference selection')
    result = candidates[0].copy()
    for i, candidate in enumerate(candidates):
        mask = np.repeat(np.repeat(labels == i, BLOCK, 0), BLOCK, 1)
        result[:H][mask] = candidate[:H][mask]
        uv_mask = mask[::2, ::2]
        for target, plane in zip(result[H:].reshape(2, H//2, W//2), candidate[H:].reshape(2, H//2, W//2)):
            target[uv_mask] = plane[uv_mask]
    return result


def encode_header(boxes: np.ndarray, labels: np.ndarray, object_mode: bool) -> bytes:
    return zlib.compress(json.dumps({'schema': 1, 'canvas': [W, H], 'block': BLOCK,
        'object_mode': object_mode, 'boxes': boxes.tolist() if object_mode else [],
        'labels': labels.tolist(), 'fps': 25, 'history': 'decoded-base-only',
        'residual_scale': 2, 'residual_offset': -256}, separators=(',', ':'), sort_keys=True).encode(), 9)


def decode_header(packet: bytes, frames: int) -> tuple[np.ndarray, np.ndarray, bool]:
    h = json.loads(zlib.decompress(packet))
    labels = np.asarray(h['labels'])
    boxes = np.asarray(h['boxes'], dtype=np.int32)
    mode = h['object_mode']
    if (h['schema'] != 1 or h['canvas'] != [W, H] or h['block'] != BLOCK
            or type(mode) is not bool or h['history'] != 'decoded-base-only' or h['fps'] != 25
            or h['residual_scale'] != 2 or h['residual_offset'] != -256
            or labels.shape != (frames, H//BLOCK, W//BLOCK) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= (4 if mode else 3))):
        raise ValueError('Invalid memory header')
    if mode and (boxes.shape != (frames, 2, 4) or np.any(boxes < 0) or np.any(boxes % 2)
                 or np.any(boxes[..., 0]+boxes[..., 2] > W) or np.any(boxes[..., 1]+boxes[..., 3] > H)):
        raise ValueError('Invalid delivered object boxes')
    return boxes, labels.astype(np.uint8), mode


def predictions(base: np.ndarray, reference: np.ndarray, matrices: np.ndarray,
                boxes: np.ndarray, mode: bool) -> list[list[np.ndarray]]:
    candidates = []
    for t in range(len(base)):
        previous = base[max(0, t-1)]
        row = [base[t], previous, warp_reference(reference, matrices[t])]
        if mode:
            row.append(object_history(previous, boxes[max(0, t-1)], boxes[t]))
        candidates.append(row)
    return candidates


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_native_base import ivf
    from experiments.tennis.k6b_rate_streams import decode_masks
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, yuv420_to_rgb

    began = time.monotonic()
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualified A6000 required')
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(5000*2**20/torch.cuda.get_device_properties(0).total_memory)
    expected = {'source': INPUTS['source'], 'transport': INPUTS['transport'], 'images': INPUTS['images'], **WEIGHTS}
    identities = {k: digest(getattr(a, k)) for k in expected}
    if identities != expected:
        raise ValueError('Input changed')
    tools = {k: svtav1.tool(k) for k in TOOLS}
    if any(tools[k]['sha256'] != v for k, v in TOOLS.items()):
        raise ValueError('Native tool changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    out = scratch/'publish'
    out.mkdir()
    commands = []

    def command(cmd: list[str]) -> None:
        start = time.monotonic()
        subprocess.run(['taskset', '-c', str(min(os.sched_getaffinity(0))), *cmd], check=True,
                       capture_output=True, timeout=300)
        commands.append({'argv': cmd, 'seconds': time.monotonic()-start})

    def codec(raw: Path, name: str, width: int, height: int, frames: int, crf: int, full: bool = False) -> np.ndarray:
        stream, decoded, repeat = out/(name+'.ivf'), scratch/(name+'.yuv'), scratch/(name+'-repeat.yuv')
        command(svtav1.encode_command('SvtAv1EncApp', raw, stream, width=width, height=height,
            fps=Fraction(25), frames=frames, crf=crf, preset=4, full_range=full))
        ivf(stream, frames, width, height, 25)
        for target in (decoded, repeat):
            command(svtav1.decode_command('dav1d', stream, target, threads=1))
        if digest(decoded) != digest(repeat) or decoded.stat().st_size != frames*width*height*3//2:
            raise ValueError('Independent native decode mismatch')
        return np.asarray(np.memmap(decoded, np.uint8, 'r', shape=(frames, height*3//2, width))).copy()

    native = scratch/'source.yuv'
    command(convert_command('ffmpeg', a.source, native, 9, 240))
    if digest(native) != SOURCE_YUV:
        raise ValueError('Native source changed')
    source = np.memmap(native, np.uint8, 'r', shape=(240, 1620, 1920))
    small_path = downscaled_source(native, a.frames, 2, scratch/'source-half.yuv')
    small = np.asarray(np.memmap(small_path, np.uint8, 'r', shape=(a.frames, H*3//2, W)))
    base_path = downscaled_source(native, a.frames, 4, scratch/'source-base.yuv')
    base = codec(base_path, 'base', 480, 272, a.frames, 63)
    base_up = np.stack([resize_plane(f, 480, 272, W, H) for f in base])
    first = scratch/'first.yuv'
    first.write_bytes(small[0].tobytes())
    reference = codec(first, 'reference', W, H, 1, 35)[0]
    with tarfile.open(a.transport) as tar:
        _, regions = decode_masks(read(tar, 'packets/alcaraz_hurkacz__031@129/exact-rle.matte').getvalue())
    regions = regions[:a.frames]
    motion, fit = fit_motion(reference, small, regions.any(1))
    (out/'motion.bin').write_bytes(motion)
    matrices = decode_motion(motion, a.frames)
    boxes = boxes_from_masks(regions)
    configurations: list[dict[str, Any]] = []
    reconstructed = []
    predictor_hashes = {}
    for mode in (False, True):
        family = 'objects' if mode else 'background'
        choices = predictions(base_up, reference, matrices, boxes, mode)
        labels = np.stack([choose_blocks(small[t], row) for t, row in enumerate(choices)])
        packet = encode_header(boxes, labels, mode)
        decoded_boxes, decoded_labels, decoded_mode = decode_header(packet, a.frames)
        # Receiver builds from decoded base/reference and paid motion/header only.
        receiver = predictions(base_up, reference, matrices, decoded_boxes, decoded_mode)
        prediction = np.stack([decode_blocks(row, lab) for row, lab in zip(receiver, decoded_labels)])
        encoder_prediction = np.stack([decode_blocks(row, lab) for row, lab in zip(choices, labels)])
        if not np.array_equal(prediction, encoder_prediction):
            raise ValueError('Wire-only prediction replay mismatch')
        predictor_hashes[family] = image_sha(prediction)
        (out/(family+'.header')).write_bytes(packet)
        residual_path = scratch/(family+'-residual.yuv')
        with residual_path.open('xb') as f:
            for t in range(a.frames):
                f.write(pack_residual(small[t], prediction[t]).tobytes())
        shared_names = ['base.ivf', 'reference.ivf', 'motion.bin', family+'.header']
        for crf in CRFS:
            name = f'{family}-{crf}'
            residual = codec(residual_path, name, W, H, a.frames, crf, True)
            frames = np.stack([reconstruct(p, r) for p, r in zip(prediction, residual)])
            reconstructed.append(frames)
            components = {n: (out/n).stat().st_size for n in [*shared_names, name+'.ivf']}
            total = sum(components.values())
            configurations.append({'name': name, 'family': family, 'crf': crf, 'bytes': total,
                'kbps': total*8/(a.frames/25)/1000, 'components': components,
                'zero_correction_payload_kbps': sum(v for k, v in components.items() if k != name+'.ivf')*8/(a.frames/25)/1000,
                'warm_reference_kbps': (total-components['reference.ivf'])*8/(a.frames/25)/1000})
    for crf in CRFS:
        name = f'anchor-{crf}'
        reconstructed.append(codec(small_path, name, W, H, a.frames, crf))
        total = (out/(name+'.ivf')).stat().st_size
        configurations.append({'name': name, 'family': 'anchor', 'crf': crf, 'bytes': total,
                               'kbps': total*8/(a.frames/25)/1000, 'components': {name+'.ivf': total}})
    prep_seconds = time.monotonic()-began
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('Metrics not CUDA')
    with tarfile.open(a.images) as tar:
        golden = json.load(read(tar, 'manifest.json'))
    ticks = [0, 3, 6] if a.frames == 24 else list(range(0, a.frames, 3))
    scores = {m: np.full((9, 3, len(ticks)), np.nan) for m in WEIGHTS}
    tick_seconds, hashes = [], {}

    def rgb(plane: np.ndarray, width: int, height: int) -> np.ndarray:
        tensor = torch.from_numpy(np.array(plane, copy=True)[None]).to('cuda')
        return yuv420_to_rgb(tensor, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    for j, tick in enumerate(ticks):
        start = time.monotonic()
        truth = rgb(source[tick], 1920, 1080)
        if image_sha(truth) != golden['hashes'][str(tick)][0]:
            raise ValueError('Historical source RGB mismatch')
        fg = cv2.resize(regions[tick].any(0).astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        supports = [np.ones_like(fg), fg, ~fg]
        if not all(s.any() for s in supports):
            raise ValueError('Empty scoring region')
        images = [cv2.resize(rgb(row[tick], W, H), (1920, 1080), interpolation=cv2.INTER_CUBIC) for row in reconstructed]
        hashes[str(tick)] = [image_sha(im) for im in [truth, *images]]
        # A single native1080p pair at a time: qualified <=5000MiB metric path.
        for pi, image in enumerate(images):
            scores['lpips'][pi, :, j] = lpips_reduce(nets['lpips'], [image], truth, [supports], 'cuda')[0]
            for region, support in enumerate(supports):
                scores['dists'][pi, region, j] = dists_crops(nets['dists'], [image], truth, support, 'cuda')[0]
        if not all(np.isfinite(s[:, :, :j+1]).all() for s in scores.values()):
            raise ValueError('Nonfinite metric')
        if j == 0:
            np.savez_compressed(out/'replay.npz', images=np.stack([truth, *images]), foreground=fg,
                base=np.asarray(base_up[tick]), reference=reference, matrix=matrices[tick],
                reconstructed=np.stack([r[tick] for r in reconstructed]))
        np.savez_compressed(out/'scores.npz', ticks=ticks[:j+1], **{m: s[:, :, :j+1] for m, s in scores.items()})
        publish_progress(os.environ['PS_STAGE'], j+1)
        tick_seconds.append(time.monotonic()-start)
    summary = {'frames': a.frames, 'ticks': ticks, 'configurations': configurations,
        'regions': ['whole', 'foreground', 'background'], 'means': {m: s.mean(2).tolist() for m, s in scores.items()},
        'hashes': hashes, 'predictor_hashes': predictor_hashes, 'wire_prediction_equal': True,
        'native_double_decode_equal': True, 'source_rgb_checks': len(ticks), 'inputs': identities,
        'source_yuv_sha256': SOURCE_YUV, 'tools': tools, 'motion_fit': fit,
        'cuda': True, 'gpu': torch.cuda.get_device_name(0), 'timing_secondary': True,
        'prep_seconds': prep_seconds, 'tick_seconds': tick_seconds,
        'metric_setup_seconds': time.monotonic()-began-prep_seconds-sum(tick_seconds),
        'seconds': time.monotonic()-began,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
        'scope': 'Familiar Al prefix; native pixel residual feasibility only; no neural bound or confirmation.'}
    write(out/'summary.json', summary)
    write(Path(os.environ['PS_STAGE_DIR'])/'memory-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage/'memory-summary.json').read_text())
    spec = json.loads((stage.parent/'spec.json').read_text())
    execution = json.loads((stage/'execution.json').read_text())
    staging = json.loads((stage/'staging.json').read_text())
    projection = 1.35*(s['prep_seconds']*4 + max(s['tick_seconds'])*32 + s['metric_setup_seconds']
                       + max(0, execution['seconds']-s['seconds']) + staging['publication_seconds']*4)
    checks = {'frames': s['frames'] == 24 and s['ticks'] == [0, 3, 6],
        'source': s['source_yuv_sha256'] == SOURCE_YUV and s['source_rgb_checks'] == 3,
        'inputs': s['inputs'] == {'source': INPUTS['source'], 'transport': INPUTS['transport'], 'images': INPUTS['images'], **WEIGHTS},
        'tools': all(s['tools'][k]['sha256'] == v for k, v in TOOLS.items()),
        'cuda': s['cuda'] and 'A6000' in s['gpu'], 'wire': s['wire_prediction_equal'],
        'native': s['native_double_decode_equal'], 'projection': 0 < projection < spec['full']['seconds'],
        'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < 5000,
        'grid': len(s['configurations']) == 9 and s['regions'] == ['whole', 'foreground', 'background'],
        'budget': spec['budget_seconds'] <= 3000, 'timing_secondary': s['timing_secondary']}
    with tarfile.open(stage/'published.tar') as tar:
        checks['summary'] = json.load(read(tar, 'publish/summary.json')) == s
        with np.load(read(tar, 'publish/scores.npz')) as z:
            checks['scores'] = z['ticks'].tolist() == [0, 3, 6] and all(z[m].shape == (9, 3, 3) and np.isfinite(z[m]).all() for m in WEIGHTS)
        for row in s['configurations']:
            checks[row['name']+':bytes'] = all(tar.getmember('publish/'+k).size == n for k, n in row['components'].items()) and sum(row['components'].values()) == row['bytes']
        for family in ('background', 'objects'):
            decode_header(read(tar, 'publish/'+family+'.header').getvalue(), 24)
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks,
                                                'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    for k in ('source', 'transport', 'images', 'lpips', 'dists'):
        p.add_argument('--'+k, type=Path)
    p.add_argument('--frames', type=int, choices=(24, 96), default=24)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('source', 'transport', 'images', 'lpips', 'dists')):
        p.error('All five frozen inputs required')
    else:
        run(a)


if __name__ == '__main__':
    main()
