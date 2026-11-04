"""Frozen W6 production-path temporal control; no smoothing or pose inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis import w6
from experiments.tennis.w6_specs import ENVIRONMENT, FILES

POST_SHA = '48c352ae1b008cbb6cba82fa6a79d5b3b46e63c0b1123516ec25750ebf150eaf'
JUDGES_SHA = '4d71f6cc6ef5307a491099116e55f62bbe30266d9c2f50b2c3088f16d2c353f1'
GEN_SHA = '031ea5d7b744a0c52f9ee769fb38d5b9d30340f567ca042c6079d4f9240f354c'
FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
DAV1D_SHA = 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6'
INPUTS = {**{n: FILES[n][1] for n in ('al', 'transport', 'anchor', 'base', 'gpu_base')},
          'generated': GEN_SHA, 'post_archive': POST_SHA}


def compare(new: np.ndarray, old: np.ndarray) -> dict[str, Any]:
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    if new.shape != old.shape:
        raise ValueError('comparison shape mismatch')
    same_mask = bool(np.array_equal(np.isnan(new), np.isnan(old)))
    finite = np.isfinite(new) & np.isfinite(old)
    valid = bool(same_mask and np.array_equal(np.isfinite(new), ~np.isnan(new))
                 and np.array_equal(np.isfinite(old), ~np.isnan(old)) and finite.any())
    worst = float(np.max(np.abs(new[finite] - old[finite]))) if finite.any() else None
    return {'valid_mask_matches': same_mask, 'valid_pairs': int(finite.sum()), 'max_abs_diff': worst,
            'passed': bool(valid and worst is not None and worst <= 1e-6)}


def comparison(got: dict[str, np.ndarray], saved: Any, frames: int) -> dict[str, Any]:
    names = [str(x) for x in saved['variants']]
    row = names.index('hybrid')
    pairs = slice(1, frames)
    values = {'flicker': compare(got['flicker'][pairs], saved['flicker'][row, pairs]),
              'energy': compare(got['energy'][pairs], saved['energy'][row, pairs]),
              'source_energy': compare(got['source_energy'][pairs], saved['energy_src'][pairs])}
    return {'passed': all(x['passed'] for x in values.values()), 'tolerance': 1e-6, 'arrays': values}


def inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('device') != 'gpu' or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or len(dispatch.get('cpu_affinity', [])) != 1
            or spec.get('budget_seconds', 0) > 1200 or spec.get('environment') != ENVIRONMENT
            or staging['environment']['sha256'] != ENVIRONMENT['sha256']):
        raise ValueError('one-CPU qualified GPU campaign and pinned environment required')
    declared = {x['name']: x for x in spec['staged_inputs']}
    actual = {x['name']: x for x in staging['inputs']}
    if (len(declared) != len(spec['staged_inputs']) or len(actual) != len(staging['inputs'])
            or set(declared) != set(INPUTS) or set(actual) != set(INPUTS)):
        raise ValueError('wrong staged input set')
    root = Path(staging['local_root']).resolve(strict=True)
    paths = {}
    for name, sha in INPUTS.items():
        for item in (declared[name], actual[name]):
            if item.get('sha256') != sha or item.get('extract', False):
                raise ValueError('staged input identity mismatch: ' + name)
        path = Path(actual[name]['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('all inputs must be host-local')
        paths[name] = path
    return paths, spec, dispatch


def saved_judges(post: Path) -> tuple[bytes, Any]:
    with tarfile.open(post) as tar:
        member = tar.getmember('publish/judges.npz')
        if not member.isfile() or member.size > 10_000_000:
            raise ValueError('saved judge archive member invalid')
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError('saved judge archive member unavailable')
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != JUDGES_SHA:
        raise ValueError('saved judges differ from historical pinned digest')
    return data, np.load(io.BytesIO(data), allow_pickle=False)


def run(stage: Path, scratch: Path, frames: int) -> int:
    import cv2
    import torch
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6b import load_generated
    from experiments.tennis.k6b_real_base_compose import compose
    from experiments.tennis.k6b_receiver_sam import project
    from experiments.visor.j import residual_energies
    from src.codecs import svtav1

    started = time.monotonic()
    paths, _, dispatch = inputs(stage)
    if frames != {'smoke': 8, 'full': 240}.get(stage.name):
        raise ValueError('incorrect stage frame count')
    if (len(os.sched_getaffinity(0)) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability(0) != (8, 9) or torch.__version__ != '2.10.0+cu128'
            or torch.version.cuda != '12.8' or torch.backends.cudnn.version() != 91002):
        raise ValueError('historical Ada/CUDA/Torch/cuDNN environment not reproduced')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    bin_dir = Path(json.loads((stage / 'staging.json').read_text())['environment']['bin'])
    ffmpeg, dav1d = bin_dir / 'ffmpeg', bin_dir / 'dav1d'
    if w6.digest(ffmpeg) != FFMPEG_SHA or w6.digest(dav1d) != DAV1D_SHA:
        raise ValueError('historical ffmpeg/dav1d binaries changed')
    _, saved = saved_judges(paths['post_archive'])
    work = scratch / 'w6-control'
    work.mkdir()
    post = w6.extract_all(paths['post_archive'], work / 'post') / 'publish'
    if w6.digest(post / 'judges.npz') != JUDGES_SHA:
        raise ValueError('extracted judges digest differs')
    a = argparse.Namespace(**paths)
    wire = w6.load_wire(a, work)
    players, order, boxes, region = (wire[k] for k in ('players', 'order', 'boxes', 'region'))
    names = [u['name'] for u in wire['units']]
    if len(names) != len(players) or len(set(names)) != len(names) or len(set(players)) != len(players):
        raise ValueError('player and unit ordering is ambiguous')
    source_yuv = work / 'source.yuv'
    subprocess.run(convert_command(str(ffmpeg), paths['al'], source_yuv, w6.SOURCE_START, w6.FRAMES),
                   check=True, capture_output=True, timeout=900)
    if w6.digest(source_yuv) != w6.SOURCE_YUV_SHA:
        raise ValueError('source YUV identity failed')
    source = np.memmap(source_yuv, np.uint8, 'r', shape=(w6.FRAMES, 1620, 1920))
    base_ivf = work / 'base.ivf'
    with tarfile.open(paths['base']) as outer:
        bundle = outer.extractfile('publish/bundle.tar')
        if bundle is None:
            raise ValueError('paid base bundle absent')
        with tarfile.open(fileobj=bundle, mode='r:') as inner:
            manifest_stream = inner.extractfile('base/manifest.json')
            data = inner.extractfile('base/al-s4.ivf')
            if manifest_stream is None or data is None:
                raise ValueError('paid base stream absent')
            manifest = json.load(manifest_stream)
            expected = next(x for x in manifest['rows'] if x['window'] == 'al' and x['scale'] == 4)
            with base_ivf.open('xb') as out:
                shutil.copyfileobj(data, out)
    if w6.digest(base_ivf) != expected['ivf']['sha256']:
        raise ValueError('paid base stream identity failed')
    base_yuv = work / 'base.yuv'
    subprocess.run(svtav1.decode_command(str(dav1d), base_ivf, base_yuv, threads=2),
                   check=True, capture_output=True, timeout=900)
    raw_base = np.memmap(base_yuv, np.uint8, 'r', shape=(w6.FRAMES, 408, 480))
    saved_base = np.load(w6.extract_member(paths['gpu_base'], 'publish/al-s4-gpu.npy', work / 'gpu-base.npy'), mmap_mode='r')
    def base_frame(t: int) -> np.ndarray:
        return cv2.resize(w6.plane_rgb(raw_base, t, 480, 272, torch, 'cuda'), (1920, 1080), interpolation=cv2.INTER_CUBIC)
    max_diff = max(int(np.abs(base_frame(t).astype(np.int16) - np.asarray(saved_base[t // 3]).astype(np.int16)).max())
                   for t in range(0, w6.FRAMES, 3))
    generated = load_generated([str(paths['generated'])], work / 'generated')
    crops = {unit: value for (unit, _), value in generated.items()}
    if len(crops) != len(generated) or set(crops) != set(names):
        raise ValueError('generated units do not match player order')
    for name in names:
        if not np.array_equal(crops[name]['t'], np.arange(w6.FRAMES)):
            raise ValueError('generated frame indices differ from frozen timeline')
    predicted = {}
    for i, name in enumerate(names):
        with np.load(post / f'{name}-masks.npz') as z:
            if not np.array_equal(z['t'], np.arange(w6.FRAMES)) or len(z['masks']) != w6.FRAMES:
                raise ValueError('receiver masks differ from frozen timeline')
            predicted[i] = {int(t): m for t, m in zip(z['t'], z['masks'])}
    empty = np.zeros((1080, 1920), bool)
    union = [np.logical_or.reduce([*r.values(), empty]) for r in region]
    arrays = {name: np.full(frames, np.nan) for name in ('flicker', 'energy', 'source_energy')}
    previous_source = previous_hybrid = None
    diagnostics = {}
    loop_started = time.monotonic()
    for t in range(frames):
        rgb = w6.plane_rgb(source, t, 1920, 1080, torch, 'cuda')
        player_crops, player_boxes, masks = {}, {}, {}
        for i, player in enumerate(players):
            row = crops[names[i]]
            index = np.flatnonzero(row['t'] == t)
            if len(index) == 1:
                player_crops[player] = row['crops'][int(index[0])]
            if t in boxes[i]:
                player_boxes[player] = boxes[i][t]
            masks[player] = project(predicted[i][t], boxes[i][t])
        base_rgb = base_frame(t)
        hybrid, _ = compose(base_rgb, player_crops, player_boxes, masks, order)
        if t in (0, frames - 1):
            diagnostics[str(t)] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in
                                   (('source_rgb', rgb), ('base_rgb', base_rgb), ('hybrid_rgb', hybrid))}
        if t and previous_source is not None and previous_hybrid is not None:
            result = residual_energies(hybrid, previous_hybrid, rgb, previous_source, union[t], union[t - 1])
            if result is not None:
                arrays['flicker'][t], arrays['energy'][t], arrays['source_energy'][t] = result
        previous_source, previous_hybrid = rgb, hybrid
    loop_seconds = time.monotonic() - loop_started
    check = comparison(arrays, saved, frames)
    if torch.cuda.max_memory_allocated() <= 0:
        raise ValueError('CUDA conversion did not allocate on the intended device')
    publish = scratch / 'publish'
    publish.mkdir()
    np.savez_compressed(publish / 'control.npz', **arrays)
    report = {'passed': check['passed'], 'control': check, 'frames': frames, 'inputs': INPUTS,
              'source_yuv_sha256': w6.SOURCE_YUV_SHA, 'base_yuv_sha256': w6.digest(base_yuv),
              'judges_sha256': JUDGES_SHA, 'base_max_abs_diff': max_diff, 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'cv2': cv2.__version__, 'ffmpeg_sha256': FFMPEG_SHA, 'dav1d_sha256': DAV1D_SHA,
              'torch_threads': 4, 'opencv_threads': 1, 'production_torch_threads': 4,
              'base_exact_match': max_diff == 0, 'base_historical_post_max_abs_diff': 1,
              'loop_seconds': loop_seconds, 'cuda_peak_bytes': torch.cuda.max_memory_allocated(),
              'diagnostics': diagnostics, 'seconds': time.monotonic() - started,
              'code_identity': dispatch['code'].get('source_sha256'), 'scope': 'control only, no smoothing'}
    (publish / 'control.json').write_text(json.dumps(report, sort_keys=True) + '\n')
    return 0 if check['passed'] else 2


def validate(stage: Path, output: Path) -> int:
    try:
        paths, spec, dispatch = inputs(stage)
        _, saved = saved_judges(paths['post_archive'])
        with tarfile.open(stage / 'published.tar') as archive:
            def read(name: str) -> bytes:
                file = archive.extractfile('publish/' + name)
                if file is None:
                    raise ValueError('publication missing ' + name)
                return file.read()
            report = json.loads(read('control.json'))
            with np.load(io.BytesIO(read('control.npz')), allow_pickle=False) as arrays:
                check = comparison({k: arrays[k] for k in ('flicker', 'energy', 'source_energy')}, saved, 8)
        if (report['frames'] != 8 or report['inputs'] != INPUTS or report['control'] != check
                or not check['passed'] or report['base_exact_match'] != (report['base_max_abs_diff'] == 0)
                or report['source_yuv_sha256'] != w6.SOURCE_YUV_SHA
                or report['judges_sha256'] != JUDGES_SHA or report['torch_threads'] != 4
                or report['cuda_peak_bytes'] <= 0
                or report['code_identity'] != dispatch['code']['source_sha256']):
            raise ValueError('smoke control or provenance failed')
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        measured = now - clock['phase_started']
        loop = report['loop_seconds']
        if not 0 < loop <= measured:
            raise ValueError('invalid measured control loop duration')
        # Recurring setup is held fixed and measured per-frame work is scaled.
        projected = 1.5 * (measured - loop + loop * 240 / 8)
        reserve = max(60.0, 1.25 * (clock['phase_started'] - clock['started']))
        remaining = clock['budget_end'] - now
        if projected > spec['full']['seconds'] or remaining < projected + reserve:
            raise ValueError(f'full control forecast does not fit: projected={projected:.1f}, remaining={remaining:.1f}')
        result = {'passed': True, 'checks': {'three_literal_series': True, 'identity': True,
                  'saved_judges_digest': True, 'forecast': True},
                  'projection': {'smoke_seconds': measured, 'full_seconds': projected,
                                 'remaining_seconds': remaining, 'reserve_seconds': reserve}}
        output.write_text(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'control': False},
                                      'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))
        return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        return validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    if args.frames is None:
        p.error('--frames required')
    return run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.frames)


if __name__ == '__main__':
    raise SystemExit(main())
