"""Finite source-pixel appearance oracle; no delivered-codec claim or generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_real_base_cost import resource_checks, separated_projection

POLICIES = ('base', 'actual', 'ideal_R', 'ideal_Fhat')
WEIGHTS = {'lpips': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
           'dists': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}
TICKS = list(range(0, 240, 3))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def image_sha(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_images(images: np.ndarray, expected: list[str]) -> None:
    if images.shape != (5, 1080, 1920, 3) or images.dtype != np.uint8:
        raise ValueError('five original full-resolution uint8 images required')
    if len(expected) != 5 or [image_sha(x) for x in images] != expected:
        raise ValueError('source/base/actual/oracle image identity changed')


def control_error(scores: dict[str, np.ndarray], controls: dict[str, Any], indices: list[int]) -> dict[str, float]:
    errors = {}
    for metric in WEIGHTS:
        expected = np.asarray(controls[metric], dtype=float)[:, indices]
        observed = scores[metric][:2]
        if expected.shape != observed.shape or not np.isfinite(observed).all() or not np.isfinite(expected).all():
            raise ValueError('incomplete or nonfinite control arrays')
        errors[metric] = float(np.max(np.abs(expected - observed)))
        if errors[metric] > 1e-4:
            raise ValueError('historical base/actual metric control failed: ' + metric)
    return errors


def decision(scores: dict[str, np.ndarray], anchor: dict[str, float]) -> dict[str, Any]:
    means = {m: values.mean(axis=1).tolist() for m, values in scores.items()}
    if any(np.asarray(v).shape != (4,) or not np.isfinite(v).all() for v in means.values()):
        raise ValueError('four finite policies required')
    wins = {p: {m: means[m][i] < anchor[m] - 0.001 for m in WEIGHTS} for i, p in enumerate(POLICIES)}
    losses = {p: {m: means[m][i] > anchor[m] + 0.001 for m in WEIGHTS} for i, p in enumerate(POLICIES)}
    verdict = 'MIXED/UNRESOLVED'
    if any(all(wins[p].values()) for p in POLICIES[2:]):
        verdict = 'HEADROOM'
    elif all(all(losses[p].values()) for p in POLICIES[2:]):
        verdict = 'NO-GO'
    return {'verdict': verdict, 'means': means, 'robust_wins': wins, 'robust_losses': losses,
            'anchor': anchor, 'sensitivity': 0.001,
            'scope': 'Al scale4 finite oracle at original total rate; no BD-rate, renderer authority or codec win'}


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs.quality import load_lpips

    began = time.monotonic()
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('qualified A6000 CUDA required')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(8000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    for name in WEIGHTS:
        if digest(getattr(a, name)) != WEIGHTS[name]:
            raise ValueError('metric weight changed: ' + name)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    inputs = scratch / 'oracle-input'
    inputs.mkdir()
    with tarfile.open(a.images) as tar:
        tar.extractall(inputs, filter='data')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    if manifest['ticks'] != TICKS or manifest['policies'] != list(POLICIES) or manifest['window'] != 'alcaraz_hurkacz/031@129':
        raise ValueError('fixed cohort or policies changed')
    if abs(manifest['rate_kbps'] - 22.737962222222222) > 1e-10:
        raise ValueError('original total rate changed')
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('metric network not on CUDA')
    ticks = TICKS[:a.judged or None]
    scores = {m: np.full((4, len(ticks)), np.nan) for m in WEIGHTS}
    timings: dict[str, list[float]] = {p: [] for p in POLICIES}
    hashes = {}
    all_mask = np.ones((1080, 1920), bool)
    publish = scratch / 'publish'
    publish.mkdir()
    for j, t in enumerate(ticks):
        frame_started = time.monotonic()
        path = inputs / 'frames' / f'{t:03d}.npz'
        if digest(path) != manifest['files'][path.relative_to(inputs).as_posix()]:
            raise ValueError('packed frame file changed')
        with np.load(path) as z:
            images = z['images']
        check_images(images, manifest['hashes'][str(t)])
        hashes[str(t)] = manifest['hashes'][str(t)]
        for i, policy in enumerate(POLICIES):
            torch.cuda.synchronize()
            started = time.monotonic()
            scores['lpips'][i, j] = lpips_reduce(nets['lpips'], [images[i + 1]], images[0], [[all_mask]], 'cuda')[0][0]
            scores['dists'][i, j] = dists_crops(nets['dists'], [images[i + 1]], images[0], all_mask, 'cuda')[0]
            torch.cuda.synchronize()
            timings[policy].append(time.monotonic() - started)
            if i == 1:
                control_error({m: s[:, :j + 1] for m, s in scores.items()}, manifest['controls'], list(range(j + 1)))
        if not all(np.isfinite(s[:, :j + 1]).all() for s in scores.values()):
            raise ValueError('nonfinite metric')
        np.savez_compressed(publish / 'scores.npz', ticks=np.asarray(ticks[:j + 1]), **{m: s[:, :j + 1] for m, s in scores.items()})
        publish_progress(os.environ['PS_STAGE'], j + 1)
        # Charge decompression, identity checks, controls and checkpoint I/O on
        # every frame, rather than treating their pilot cost as one-time setup.
        untraced = time.monotonic() - frame_started - sum(timings[p][-1] for p in POLICIES)
        for policy in POLICIES:
            timings[policy][-1] += max(0.0, untraced) / 4
    errors = control_error(scores, manifest['controls'], list(range(len(ticks))))
    summary = {'ticks': ticks, 'policies': POLICIES, 'timings': timings, 'control_errors': errors,
               'hashes': hashes, 'cuda': True, 'gpu': torch.cuda.get_device_name(0),
               'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
               'input_sha256': digest(a.images), 'manifest_sha256': digest(inputs / 'manifest.json'),
               'weight_sha256': WEIGHTS, 'prior_job': manifest['prior_job'],
               'rate_kbps': manifest['rate_kbps'], 'timing_secondary': a.timing_secondary, 'seconds': time.monotonic() - began,
               'assessment': decision(scores, manifest['anchor']) if a.judged == 0 else None}
    write(publish / 'oracle.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'oracle-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage / 'oracle-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    execution = json.loads((stage / 'execution.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    timing_secondary = spec.get('arguments', []).count('--timing-secondary') == 1
    projection = separated_projection(execution['seconds'], s['timings'], staging['publication_seconds'],
                                      require_stability=not timing_secondary)
    checks = resource_checks(projection, spec['full']['seconds'])
    checks.update({'saved_timing_policy': s.get('timing_secondary', False) == timing_secondary,
                   'ticks': s['ticks'] == [0, 3, 6], 'policies': s['policies'] == list(POLICIES),
                   'cuda': s['cuda'] and 'A6000' in s['gpu'],
                   'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < 8000,
                   'controls': all(0 <= v <= 1e-4 for v in s['control_errors'].values()) and set(s['control_errors']) == set(WEIGHTS),
                   'weights': s['weight_sha256'] == WEIGHTS,
                   'input_identity': s['input_sha256'] == next(x['sha256'] for x in spec['staged_inputs'] if x['name'] == 'images'),
                   'hash_coverage': set(s['hashes']) == {'0', '3', '6'} and all(len(v) == 5 for v in s['hashes'].values()),
                   'scale': spec['scale']['judged'] == {'smoke': 3, 'full': 0},
                   'allocation': spec['budget_seconds'] == 1100 and spec['full']['seconds'] == 550})
    with tarfile.open(stage / 'published.tar') as tar:
        score_file = tar.extractfile('publish/scores.npz')
        assert score_file is not None
        with np.load(score_file) as z:
            checks['saved_scores'] = z['ticks'].tolist() == [0, 3, 6] and all(z[m].shape == (4, 3) and np.isfinite(z[m]).all() for m in WEIGHTS)
        summary_file = tar.extractfile('publish/oracle.json')
        assert summary_file is not None
        saved = json.load(summary_file)
        checks['published_summary'] = saved == s
    record = {'passed': all(checks.values()), 'checks': checks, 'projection': projection,
              'projected_full_seconds': projection['projected_worker_seconds'],
              'projected_with_publication': projection['projected_full_seconds']}
    write(Path(os.environ['PS_VALIDATION_PATH']), record)
    if not record['passed']:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--timing-secondary', action='store_true',
                   help='Authorized variable-timing policy; retain slower-warm projection and hard budgets')
    p.add_argument('--images', type=Path)
    p.add_argument('--lpips', type=Path)
    p.add_argument('--dists', type=Path)
    p.add_argument('--judged', type=int, choices=(0, 3), default=3)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('images', 'lpips', 'dists')):
        p.error('all frozen image and weight inputs required')
    else:
        run(a)


if __name__ == '__main__':
    main()
