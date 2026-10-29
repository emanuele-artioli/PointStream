"""One paid cleaned-background point, with exact historical score controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Any

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, SOURCE_YUV, TOOLS, WINDOW, stream_info
from experiments.tennis.k6b_ideal_player import TICKS, WEIGHTS, control_error, digest, image_sha, write
from experiments.tennis.k6b_real_base_cost import resource_checks, separated_projection

POLICIES = ('base', 'actual', 'clean_actual', 'clean_ideal_R')


def anchor_at(curves: dict[str, Any], rate: float) -> dict[str, float] | None:
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid paid rate')
    if any(not c[0][0] <= rate <= c[-1][0] for c in curves.values()):
        return None
    return {m: -float(np.interp(rate, [r for r, _ in c], [q for _, q in c])) for m, c in curves.items()}


def assess(scores: dict[str, np.ndarray], anchor: dict[str, float] | None) -> dict[str, Any]:
    if set(scores) != set(WEIGHTS) or any(v.ndim != 2 or v.shape[0] != 4 or v.shape[1] == 0
                                        or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Four complete finite policy arrays required')
    means = {m: v.mean(1).tolist() for m, v in scores.items()}
    if anchor is None:
        return {'verdict': 'UNRESOLVED/ANCHOR-COVERAGE', 'means': means, 'anchor': None}
    if set(anchor) != set(WEIGHTS) or not all(np.isfinite(x) for x in anchor.values()):
        raise ValueError('Finite two-metric anchor required')
    wins = {p: {m: means[m][i] < anchor[m] - .001 for m in WEIGHTS} for i, p in enumerate(POLICIES)}
    losses = {p: {m: means[m][i] > anchor[m] + .001 for m in WEIGHTS} for i, p in enumerate(POLICIES)}
    verdict = 'MIXED/UNRESOLVED'
    if any(all(wins[p].values()) for p in POLICIES[2:]):
        verdict = 'HEADROOM'
    elif all(all(losses[p].values()) for p in POLICIES[2:]):
        verdict = 'NO-GO'
    return {'verdict': verdict, 'means': means, 'anchor': anchor, 'robust_wins': wins,
            'robust_losses': losses, 'sensitivity': .001,
            'scope': 'One finite paid cleaned-base point, optimistic side accounting; no BD-rate or codec win.'}


def new_images(images: np.ndarray, overlay: np.ndarray, written: np.ndarray,
               region: np.ndarray, base: np.ndarray) -> np.ndarray:
    if (images.shape != (3, 1080, 1920, 3) or images.dtype != np.uint8
            or overlay.shape != base.shape or base.shape != images[0].shape
            or overlay.dtype != np.uint8 or base.dtype != np.uint8
            or written.shape != region.shape or region.shape != (1080, 1920)
            or written.dtype != bool or region.dtype != bool):
        raise ValueError('Original full-resolution images and boolean support required')
    if not np.array_equal(np.where(written[..., None], overlay, images[1]), images[2]):
        raise ValueError('Historical actual-overlay replay failed')
    actual = np.where(written[..., None], overlay, base)
    ideal = np.where(region[..., None], images[0], base)
    if not np.array_equal(actual[~written], base[~written]) or not np.array_equal(ideal[~region], base[~region]):
        raise ValueError('Paid base fallback failed')
    return np.stack([images[0], images[1], images[2], actual, ideal])


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs.quality import load_lpips, yuv420_to_rgb
    from src.codecs.svtav1 import decode_command, tool

    began = time.monotonic()
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualified A6000 CUDA required')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.cuda.set_per_process_memory_fraction(8000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    for name in WEIGHTS:
        if digest(getattr(a, name)) != WEIGHTS[name]:
            raise ValueError('Metric backbone identity changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    inputs, publish = scratch / 'inputs', scratch / 'publish'
    inputs.mkdir()
    publish.mkdir()
    # Preparation publication contains one inner input archive and its receipt.
    with tarfile.open(a.prepared) as outer:
        stream = outer.extractfile('publish/input.tar')
        if stream is None:
            raise ValueError('Prepared input archive missing')
        with tarfile.open(fileobj=stream, mode='r:') as tar:
            tar.extractall(inputs, filter='data')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    if (manifest['window'] != WINDOW or manifest['frames'] != 240 or manifest['ticks'] != TICKS
            or manifest['policies'] != list(POLICIES) or manifest['input_sha256'] != INPUTS
            or manifest['source_yuv_sha256'] != SOURCE_YUV):
        raise ValueError('Prepared cohort or frozen input contract changed')
    for name, expected in manifest['files'].items():
        if digest(inputs / name) != expected:
            raise ValueError('Prepared member changed: ' + name)
    native = tool('dav1d')
    if native['sha256'] != TOOLS['dav1d'] or stream_info(inputs / 'clean.ivf', 240) != manifest['ivf']:
        raise ValueError('Native stream/decoder identity changed')
    fresh = scratch / 'fresh.yuv'
    subprocess.run(decode_command('dav1d', inputs / 'clean.ivf', fresh, threads=1),
                   check=True, capture_output=True, timeout=30)
    if digest(fresh) != manifest['files']['clean.yuv']:
        raise ValueError('Independent native decode failed')
    base_rate = manifest['ivf']['bytes'] * 8 / 9.6 / 1000
    if (abs(manifest['side_kbps'] - 5.607128888888889) > 1e-12
            or abs(manifest['rate_kbps'] - base_rate - manifest['side_kbps']) > 1e-12
            or (inputs / 'bootstrap.json').stat().st_size != manifest['bootstrap_bytes']):
        raise ValueError('Paid side/base accounting changed')
    header = json.loads((inputs / 'bootstrap.json').read_text())
    if header['base_ivf_sha256'] != manifest['ivf']['sha256'] or manifest['bootstrap_bytes'] != 444:
        raise ValueError('Receiver bootstrap identity changed')
    decoded = np.memmap(fresh, np.uint8, 'r', shape=(240, 408, 480))
    sources = np.memmap(inputs / 'source-ticks.yuv', np.uint8, 'r', shape=(80, 1620, 1920))
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('Metric networks must stay on CUDA')
    ticks = TICKS[:a.judged or None]
    scores = {m: np.full((4, len(ticks)), np.nan) for m in WEIGHTS}
    timings: dict[str, list[float]] = {p: [] for p in POLICIES}
    hashes = {}
    base_rgb = np.lib.format.open_memmap(publish / 'clean-base-rgb.npy', mode='w+', dtype=np.uint8,
                                       shape=(len(ticks), 1080, 1920, 3))
    all_mask = np.ones((1080, 1920), bool)
    for j, t in enumerate(ticks):
        started_frame = time.monotonic()
        with np.load(inputs / 'frames' / f'{t:03d}.npz') as z:
            original, overlay, written, region = [z[k].copy() for k in ('images', 'overlay', 'written', 'region')]
        expected = manifest['hashes'][str(t)]
        if ([image_sha(x) for x in original] != expected['images'] or image_sha(overlay) != expected['overlay']
                or image_sha(written) != expected['written'] or image_sha(region) != expected['region']
                or image_sha(sources[j]) != expected['source_yuv']):
            raise ValueError('Saved frame/support/source identity changed')

        def rgb(plane: np.ndarray, width: int, height: int) -> np.ndarray:
            gpu = torch.from_numpy(np.array(plane, copy=True)[None]).to('cuda')
            return yuv420_to_rgb(gpu, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

        source_rgb = rgb(sources[j], 1920, 1080)
        if image_sha(source_rgb) != expected['images'][0]:
            raise ValueError('Original CUDA source RGB identity failed')
        base = cv2.resize(rgb(decoded[t], 480, 272), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        # A repeated conversion checks the new native path, not just file size.
        if not np.array_equal(base, cv2.resize(rgb(decoded[t], 480, 272), (1920, 1080), interpolation=cv2.INTER_CUBIC)):
            raise ValueError('Repeated new-base CUDA conversion changed')
        images = new_images(original, overlay, written, region, base)
        base_rgb[j] = base
        hashes[str(t)] = [image_sha(x) for x in images]
        for i, policy in enumerate(POLICIES):
            torch.cuda.synchronize()
            began_policy = time.monotonic()
            scores['lpips'][i, j] = lpips_reduce(nets['lpips'], [images[i+1]], images[0], [[all_mask]], 'cuda')[0][0]
            scores['dists'][i, j] = dists_crops(nets['dists'], [images[i+1]], images[0], all_mask, 'cuda')[0]
            torch.cuda.synchronize()
            timings[policy].append(time.monotonic() - began_policy)
            if i == 1:
                control_error({m: v[:, :j+1] for m, v in scores.items()}, manifest['controls'], list(range(j+1)))
        if not all(np.isfinite(v[:, :j+1]).all() for v in scores.values()):
            raise ValueError('Nonfinite neural scores')
        np.savez_compressed(publish / 'scores.npz', ticks=np.array(ticks[:j+1]), **{m: v[:, :j+1] for m, v in scores.items()})
        base_rgb.flush()
        publish_progress(os.environ['PS_STAGE'], j+1)
        extra = time.monotonic() - started_frame - sum(timings[p][-1] for p in POLICIES)
        for policy in POLICIES:
            timings[policy][-1] += max(0., extra) / 4
    base_rgb.flush()
    anchor = anchor_at(manifest['curves'], manifest['rate_kbps'])
    summary = {'ticks': ticks, 'policies': list(POLICIES), 'timings': timings,
               'control_errors': control_error(scores, manifest['controls'], list(range(len(ticks)))),
               'hashes': hashes, 'cuda': True, 'gpu': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'opencv': cv2.__version__,
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
               'weight_sha256': WEIGHTS, 'prepared_sha256': digest(a.prepared),
               'manifest_sha256': digest(inputs / 'manifest.json'), 'native_decoder': native,
               'clean_base_rgb_sha256': digest(publish / 'clean-base-rgb.npy'),
               'rate_kbps': manifest['rate_kbps'], 'base_kbps': base_rate, 'side_kbps': manifest['side_kbps'],
               'timing_secondary': a.timing_secondary, 'source_rgb_checks': len(ticks),
               'seconds': time.monotonic() - began,
               'assessment': assess(scores, anchor) if a.judged == 0 else None}
    write(publish / 'quality.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'quality-summary.json', summary)


def validate() -> None:
    from experiments.tennis.k6b_clean_background_prep import read
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage / 'quality-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    execution = json.loads((stage / 'execution.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    timing_secondary = spec['arguments'].count('--timing-secondary') == 1
    projection = separated_projection(execution['seconds'], s['timings'], staging['publication_seconds'],
                                      require_stability=not timing_secondary)
    checks = resource_checks(projection, spec['full']['seconds'])
    checks.update({'timing_policy': timing_secondary and s['timing_secondary'],
                   'ticks': s['ticks'] == [0, 3, 6], 'policies': s['policies'] == list(POLICIES),
                   'source_rgb': s['source_rgb_checks'] == 3,
                   'cuda': s['cuda'] and 'A6000' in s['gpu'],
                   'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < 8000,
                   'controls': set(s['control_errors']) == set(WEIGHTS) and all(0 <= v <= 1e-4 for v in s['control_errors'].values()),
                   'weights': s['weight_sha256'] == WEIGHTS,
                   'input_identity': s['prepared_sha256'] == next(x['sha256'] for x in spec['staged_inputs'] if x['name'] == 'prepared'),
                   'hash_coverage': set(s['hashes']) == {'0', '3', '6'} and all(len(v) == 5 for v in s['hashes'].values()),
                   'scale': spec['scale']['judged'] == {'smoke': 3, 'full': 0},
                   'allocation': spec['budget_seconds'] == 800 and spec['full']['seconds'] == 350,
                   'native_decoder': s['native_decoder']['sha256'] == TOOLS['dav1d'],
                   'fixed_side': abs(s['side_kbps'] - 5.607128888888889) < 1e-12})
    with tarfile.open(stage / 'published.tar') as tar:
        with np.load(read(tar, 'publish/scores.npz')) as z:
            checks['scores'] = z['ticks'].tolist() == [0, 3, 6] and all(z[m].shape == (4, 3) and np.isfinite(z[m]).all() for m in WEIGHTS)
        checks['saved_summary'] = json.load(read(tar, 'publish/quality.json')) == s
        raw = read(tar, 'publish/clean-base-rgb.npy')
        checks['saved_pixels'] = hashlib.sha256(raw.getbuffer()).hexdigest() == s['clean_base_rgb_sha256'] and np.load(raw).shape == (3, 1080, 1920, 3)
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks, 'projection': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--prepared', type=Path)
    p.add_argument('--lpips', type=Path)
    p.add_argument('--dists', type=Path)
    p.add_argument('--judged', type=int, choices=(0, 3), default=3)
    p.add_argument('--timing-secondary', action='store_true')
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('prepared', 'lpips', 'dists')):
        p.error('prepared inputs and both frozen metric weights required')
    else:
        run(a)


if __name__ == '__main__':
    main()
