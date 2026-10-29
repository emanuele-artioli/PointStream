"""Paid single-stream, decoder-only motion-aligned temporal restoration pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, SOURCE_YUV, TOOLS, read
from experiments.tennis.k6b_ideal_player import WEIGHTS, digest, image_sha, write

PUBLICATION_SHA = '2933ca5bcb82f52b832e20c21db7737acf72103d6905b1d816baacc992fa6554'
ALPHAS = (.15, .35, .60)
CRFS = (43, 53, 63)
W, H = 960, 544


def header(alpha_index: int, frames: int) -> bytes:
    if alpha_index not in range(3) or frames not in (24, 96):
        raise ValueError('Unknown restoration profile')
    return struct.pack('<4sBBH', b'TRF1', 1, alpha_index, frames)


def parse_header(data: bytes) -> tuple[float, int]:
    if len(data) != 8:
        raise ValueError('Invalid restoration header length')
    magic, version, index, frames = struct.unpack('<4sBBH', data)
    if magic != b'TRF1' or version != 1 or index not in range(3) or frames not in (24, 96):
        raise ValueError('Invalid restoration profile')
    return ALPHAS[index], frames


def align(current: np.ndarray, neighbour: np.ndarray, flow: np.ndarray, backward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flow maps current coordinates to neighbour; all validity is decoder-derived."""
    if (current.shape != neighbour.shape or current.ndim != 3 or current.shape[2] != 3
            or flow.shape != (*current.shape[:2], 2) or backward.shape != flow.shape
            or not np.isfinite(flow).all() or not np.isfinite(backward).all()):
        raise ValueError('Finite RGB/flow pair required')
    height, width = current.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    mx, my = xx+flow[..., 0], yy+flow[..., 1]
    warped = cv2.remap(neighbour, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    reverse = cv2.remap(backward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1000., 1000.))
    consistency = np.linalg.norm(flow+reverse, axis=2)
    current_y = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
    warped_y = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    photo = np.abs(current_y-warped_y)
    valid = ((mx >= 0) & (mx <= width-1) & (my >= 0) & (my <= height-1)
             & (consistency <= 1.5) & (photo <= 12))
    weights = np.where(valid, np.exp(-(photo/12)**2), 0).astype(np.float32)
    return warped, weights


def fuse(current: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray]], alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError('Invalid restoration strength')
    value = current.astype(np.float32)
    numerator, denominator = value.copy(), np.ones(current.shape[:2], np.float32)
    for warped, weight in pairs:
        if warped.shape != current.shape or weight.shape != current.shape[:2] or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError('Invalid aligned neighbour')
        numerator += warped*weight[..., None]
        denominator += weight
    estimate = numerator/denominator[..., None]
    return np.clip(np.rint(value+alpha*(estimate-value)), 0, 255).astype(np.uint8)


def restore_inputs(frames: np.ndarray, tick: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv2.setRNGSeed(20261028)
    dis = getattr(cv2, 'DISOpticalFlow_create')(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    current = frames[tick]
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    pairs = []
    for neighbour_tick in (tick-1, tick+1):
        if not 0 <= neighbour_tick < len(frames):
            continue
        neighbour = frames[neighbour_tick]
        neighbour_gray = cv2.cvtColor(neighbour, cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray, neighbour_gray, None)
        reverse = dis.calc(neighbour_gray, gray, None)
        pairs.append(align(current, neighbour, flow, reverse))
    return pairs


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6b_native_base import ivf
    from experiments.tennis.k6b_rate_streams import decode_masks
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce, luma
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, yuv420_to_rgb

    began = time.monotonic()
    expected = {'source': INPUTS['source'], 'transport': INPUTS['transport'], 'images': INPUTS['images'],
                'publication': PUBLICATION_SHA, **WEIGHTS}
    identities = {k: digest(getattr(a, k)) for k in expected}
    if identities != expected:
        raise ValueError('Frozen input changed')
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualified A6000 required')
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(5000*2**20/torch.cuda.get_device_properties(0).total_memory)
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[name]['sha256'] != sha for name, sha in TOOLS.items()):
        raise ValueError('Native binary identity changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    out = scratch/'publish'
    out.mkdir()
    source = scratch/'source.yuv'
    subprocess.run(convert_command('ffmpeg', a.source, source, 9, 240), check=True, capture_output=True, timeout=120)
    if digest(source) != SOURCE_YUV:
        raise ValueError('Original native source changed')
    raw = np.memmap(source, np.uint8, 'r', shape=(240, 1620, 1920))
    with tarfile.open(a.transport) as tar:
        _, regions = decode_masks(read(tar, 'packets/alcaraz_hurkacz__031@129/exact-rle.matte').getvalue())
    with tarfile.open(a.images) as tar:
        golden = json.load(read(tar, 'manifest.json'))
    with tarfile.open(a.publication) as tar:
        previous = json.load(read(tar, 'publish/summary.json'))
        with np.load(read(tar, 'publish/scores.npz')) as z:
            old_scores = {m: z[m].copy() for m in WEIGHTS}
        for crf in CRFS:
            (out/f'anchor-{crf}.ivf').write_bytes(read(tar, f'publish/anchor-{crf}.ivf').getvalue())
    if previous['frames'] != 96 or previous['ticks'] != list(range(0, 96, 3)):
        raise ValueError('Wrong saved-anchor cohort')

    def rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        tensor = torch.from_numpy(np.array(frame, copy=True)[None]).to('cuda')
        return yuv420_to_rgb(tensor, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    decoded = []
    commands = []
    configs = []
    for ci, crf in enumerate(CRFS):
        stream = out/f'anchor-{crf}.ivf'
        info = ivf(stream, 96, W, H, 25)
        paths = [scratch/f'q{crf}.yuv', scratch/f'q{crf}-repeat.yuv']
        for path in paths:
            command = svtav1.decode_command('dav1d', stream, path, threads=1)
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            commands.append(command)
        if digest(paths[0]) != digest(paths[1]) or paths[0].stat().st_size != 96*H*W*3//2:
            raise ValueError('Independent native anchor decode mismatch')
        planes = np.memmap(paths[0], np.uint8, 'r', shape=(96, H*3//2, W))
        decoded.append(np.stack([rgb(frame, W, H) for frame in planes[:a.frames]]))
        configs.append({'name': f'q{crf}-plain', 'crf': crf, 'alpha': 0., 'bytes': info['bytes'],
            'header_bytes': 0, 'stream_bytes': info['bytes'], 'kbps': info['bytes']*8/(96/25)/1000})
        for ai, alpha in enumerate(ALPHAS):
            packet = header(ai, 96)
            if parse_header(packet) != (alpha, 96):
                raise ValueError('Profile replay failed')
            name = f'q{crf}-a{ai}'
            (out/(name+'.trf')).write_bytes(packet)
            configs.append({'name': name, 'crf': crf, 'alpha': alpha, 'bytes': info['bytes']+len(packet),
                'header_bytes': len(packet), 'stream_bytes': info['bytes'], 'kbps': (info['bytes']+len(packet))*8/(96/25)/1000})
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('Metric network not on CUDA')
    setup_seconds = time.monotonic()-began
    ticks = [0, 3, 6] if a.frames == 24 else list(range(0, 96, 3))
    scores = {m: np.full((12, 3, len(ticks)), np.nan) for m in WEIGHTS}
    temporal = np.full((12, 3, len(ticks)-1), np.nan)
    hashes, tick_seconds, controls = {}, [], {m: 0. for m in WEIGHTS}
    previous_images, previous_truth = None, None
    for j, tick in enumerate(ticks):
        start = time.monotonic()
        truth = rgb(raw[tick], 1920, 1080)
        if image_sha(truth) != golden['hashes'][str(tick)][0]:
            raise ValueError('Original source RGB identity mismatch')
        fg = cv2.resize(regions[tick].any(0).astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        supports = [np.ones_like(fg), fg, ~fg]
        if not all(s.any() for s in supports):
            raise ValueError('Empty scoring support')
        images: list[np.ndarray] = []
        confidence = []
        for ci, frames in enumerate(decoded):
            current = frames[tick]
            pairs = restore_inputs(frames, tick)
            confidence.append(float(np.mean([np.mean(w > 0) for _, w in pairs])))
            family = [current]+[fuse(current, pairs, alpha) for alpha in ALPHAS]
            images.extend(cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_CUBIC) for im in family)
            if image_sha(images[ci*4]) != previous['hashes'][str(tick)][7+ci]:
                raise ValueError('Saved native anchor RGB does not reproduce')
        for pi, im in enumerate(images):
            scores['lpips'][pi, :, j] = lpips_reduce(nets['lpips'], [im], truth, [supports], 'cuda')[0]
            for ri, support in enumerate(supports):
                scores['dists'][pi, ri, j] = dists_crops(nets['dists'], [im], truth, support, 'cuda')[0]
            if j and previous_images is not None and previous_truth is not None:
                delta = np.abs((luma(im)-luma(previous_images[pi]))-(luma(truth)-luma(previous_truth)))
                for ri, support in enumerate(supports):
                    temporal[pi, ri, j-1] = float(delta[support].mean())
        for m in WEIGHTS:
            for ci in range(3):
                error = float(np.max(np.abs(scores[m][ci*4, :, j]-old_scores[m][6+ci, :, tick//3])))
                controls[m] = max(controls[m], error)
                if error > 1e-4:
                    raise ValueError('Saved unfiltered metric control mismatch')
        if not all(np.isfinite(s[:, :, :j+1]).all() for s in scores.values()):
            raise ValueError('Nonfinite restoration score')
        hashes[str(tick)] = [image_sha(im) for im in [truth, *images]]
        if j == 0:
            np.savez_compressed(out/'replay.npz', images=np.stack([truth, *images]), foreground=fg,
                                decoder_frames=decoded[0][:2], confidence=np.array(confidence))
        np.savez_compressed(out/'scores.npz', ticks=ticks[:j+1], **{m: s[:, :, :j+1] for m, s in scores.items()},
                            temporal_residual=temporal[:, :, :j])
        previous_images, previous_truth = images, truth
        tick_seconds.append(time.monotonic()-start)
        publish_progress(os.environ['PS_STAGE'], j+1)
    summary = {'frames': a.frames, 'ticks': ticks, 'configs': configs, 'regions': ['whole', 'foreground', 'background'],
        'means': {m: s.mean(2).tolist() for m, s in scores.items()}, 'hashes': hashes, 'controls': controls,
        'temporal_means': temporal.mean(2).tolist(), 'temporal_scope': '3-native-frame stride, residual distortion, not experience-track flicker',
        'inputs': identities, 'source_yuv_sha256': SOURCE_YUV, 'tools': tools, 'commands': commands,
        'setup_seconds': setup_seconds, 'tick_seconds': tick_seconds, 'seconds': time.monotonic()-began,
        'decoder_source_access': False, 'native_double_decode': True, 'source_rgb_checks': len(ticks),
        'cuda': True, 'gpu': torch.cuda.get_device_name(0), 'timing_secondary': True,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
        'scope': 'Familiar Al96/32; decoder-only DIS temporal fusion, restricted960x544 SVT anchors; no trained restoration model.'}
    write(out/'restoration.json', summary)
    write(Path(os.environ['PS_STAGE_DIR'])/'restoration-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage/'restoration-summary.json').read_text())
    spec = json.loads((stage.parent/'spec.json').read_text())
    staging = json.loads((stage/'staging.json').read_text())
    execution = json.loads((stage/'execution.json').read_text())
    projection = 1.35*(s['setup_seconds']*4+max(s['tick_seconds'])*32+staging['publication_seconds']*4
                       +max(0, execution['seconds']-s['seconds']))
    checks = {'frames': s['frames'] == 24 and s['ticks'] == [0, 3, 6],
        'source': s['source_yuv_sha256'] == SOURCE_YUV and s['source_rgb_checks'] == 3,
        'inputs': s['inputs'] == {'source': INPUTS['source'], 'transport': INPUTS['transport'], 'images': INPUTS['images'], 'publication': PUBLICATION_SHA, **WEIGHTS},
        'cuda': s['cuda'] and 'A6000' in s['gpu'], 'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < 5000,
        'controls': all(v <= 1e-4 for v in s['controls'].values()), 'decoder_only': s['decoder_source_access'] is False,
        'native': s['native_double_decode'], 'projection': projection < spec['full']['seconds'],
        'budget': spec['budget_seconds'] <= 1800, 'timing_secondary': s['timing_secondary']}
    with tarfile.open(stage/'published.tar') as tar:
        with np.load(read(tar, 'publish/scores.npz')) as z:
            checks['scores'] = all(z[m].shape == (12, 3, 3) and np.isfinite(z[m]).all() for m in WEIGHTS)
        for row in s['configs']:
            count = tar.getmember(f'publish/anchor-{row["crf"]}.ivf').size
            if row['alpha']:
                packet = read(tar, 'publish/'+row['name']+'.trf').getvalue()
                checks[row['name']+':header'] = parse_header(packet) == (row['alpha'], 96)
                count += len(packet)
            checks[row['name']+':bytes'] = count == row['bytes']
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks,
                                                 'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    for name in ('source', 'transport', 'images', 'publication', 'lpips', 'dists'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--frames', type=int, choices=(24, 96), default=24)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('source', 'transport', 'images', 'publication', 'lpips', 'dists')):
        p.error('All six frozen inputs required')
    else:
        run(a)


if __name__ == '__main__':
    main()
