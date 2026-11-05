"""Bounded W6 fixed-profile smoothing experiment; frozen prospective draft."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis import w6, w6_control_job as ctrl
from experiments.tennis.w1_roi import WEIGHTS

FFMPEG_SHA = ctrl.FFMPEG_SHA
DAV1D_SHA = ctrl.DAV1D_SHA
JUDGES_SHA = ctrl.JUDGES_SHA
MASK_ARCHIVE_SHA = 'fbb39eec8b9d823b8d57acd0efc0d8b6f051739886ce94592dfffbfc3494142d'
MASK_NPZ_SHA = '052d31b58dbb5aa4ab7fd68ad14da19c1b6647b0f778281373056ab1fbd559fb'
INPUTS = {**ctrl.INPUTS, 'masks': MASK_ARCHIVE_SHA, **WEIGHTS}
WEIGHTS_FIXED = (0.0, 0.25, 0.5)


def inputs(stage: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (spec.get('device') != 'gpu' or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or len(dispatch.get('cpu_affinity', [])) != 1
            or not 0 < spec.get('budget_seconds', 0) <= 1100
            or not 0 < spec.get('smoke', {}).get('seconds', 0) <= 200
            or not 0 < spec.get('full', {}).get('seconds', 0) <= 800
            or spec.get('environment', {}).get('sha256') != ctrl.ENVIRONMENT['sha256']
            or staging['environment']['sha256'] != ctrl.ENVIRONMENT['sha256']
            or spec.get('external_cpu_reserve_seconds') != 400
            or spec.get('external_gpu_reserve_seconds') != 100
            or spec.get('post_full_audit_cpu_limit_seconds') != 180
            or spec.get('phase_cpu_limit_seconds') != 1500
            or spec.get('phase_gpu_limit_seconds') != 1200
            or spec.get('new_phase_preparation_consumed_cpu_seconds') != 0
            or spec.get('new_phase_post_audit_consumed_cpu_seconds') != 0):
        raise ValueError('bounded one-CPU GPU campaign and original environment required')
    declared = {x['name']: x for x in spec['staged_inputs']}
    actual = {x['name']: x for x in staging['inputs']}
    if (set(declared) != set(INPUTS) or set(actual) != set(INPUTS)
            or len(declared) != len(spec['staged_inputs']) or len(actual) != len(staging['inputs'])):
        raise ValueError('wrong staged input set')
    local = Path(staging['local_root']).resolve(strict=True)
    paths = {}
    for name, sha in INPUTS.items():
        if any(item.get('sha256') != sha or item.get('extract', False) for item in (declared[name], actual[name])):
            raise ValueError('input identity changed: ' + name)
        path = Path(actual[name]['path']).resolve(strict=True)
        if not path.is_relative_to(local):
            raise ValueError('input must be host local')
        paths[name] = path
    return paths, spec, dispatch


def player_scores(image: np.ndarray, truth: np.ndarray, mask: np.ndarray, nets: dict[str, Any]) -> tuple[float, float]:
    from experiments.tennis.k6 import DISTS_PAD
    from experiments.visor.h3b import dists_crops, lpips_reduce

    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError('empty W1 scoring mask')
    crop = (slice(max(0, ys.min() - DISTS_PAD), min(1080, ys.max() + 1 + DISTS_PAD)),
            slice(max(0, xs.min() - DISTS_PAD), min(1920, xs.max() + 1 + DISTS_PAD)))
    lp = lpips_reduce(nets['lpips'], [image], truth, [[mask]], 'cuda')[0][0]
    di = dists_crops(nets['dists'], [image[crop]], truth[crop], mask[crop], 'cuda')[0]
    if not np.isfinite([lp, di]).all():
        raise ValueError('nonfinite player score')
    return float(lp), float(di)


def require_complete_timeline(names: list[str], crops: dict[str, Any], boxes: list[dict[int, Any]],
                              predicted: dict[int, dict[int, Any]]) -> None:
    """This frozen worker accepts only complete 240-frame crop/mask/box tracks."""
    ticks = set(range(240))
    if len(boxes) != len(names) or set(predicted) != set(range(len(names))):
        raise ValueError('incomplete player timeline set')
    for i, name in enumerate(names):
        if (name not in crops or not np.array_equal(crops[name]['t'], np.arange(240))
                or len(crops[name]['crops']) != 240 or set(boxes[i]) != ticks
                or set(predicted[i]) != ticks):
            raise ValueError('frozen worker requires complete crop, box and receiver-mask timelines')


def profile_step(current: np.ndarray | None, t: int, previous: tuple[int, np.ndarray] | None) -> tuple[list[np.ndarray], tuple[int, np.ndarray] | None]:
    """Return fixed profiles and remember only the unsmoothed current crop."""
    from experiments.tennis.w6b_smoothing import smooth_crops

    if current is None:
        return [], None
    predecessor = previous[1] if previous is not None and previous[0] == t - 1 else None
    return smooth_crops(current, predecessor, WEIGHTS_FIXED), (t, current.copy())


def checkpoint(path: Path, arrays: dict[str, np.ndarray], done: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / '.partial.npz'
    np.savez_compressed(temporary, **arrays, done=np.asarray(done), weights=np.asarray(WEIGHTS_FIXED))
    temporary.replace(path / 'series.npz')


def run(stage: Path, scratch: Path, frames: int) -> int:
    import cv2
    import torch
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis import w1_score
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips
    from experiments.tennis.k6b import load_generated
    from experiments.tennis.k6b_real_base_compose import compose
    from experiments.tennis.k6b_receiver_sam import project
    from experiments.visor.j import residual_energies
    from src.codecs import svtav1

    started = time.monotonic()
    paths, spec, dispatch = inputs(stage)
    if frames != {'smoke': 16, 'full': 240}.get(stage.name):
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
    _, saved = ctrl.saved_judges(paths['post_archive'])
    work = scratch / 'w6-smoothing'
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
    require_complete_timeline(names, crops, boxes, predicted)
    generation_dir = work / 'generated' / 'generated' / 'g0' / 'publish'
    crop_checksums = {}
    for name in names:
        metadata = json.loads((generation_dir / f'{name}@match5.json').read_text())
        crop_checksums[name] = metadata['npz_sha256']
    empty = np.zeros((1080, 1920), bool)
    union = [np.logical_or.reduce([*r.values(), empty]) for r in region]
    # The fixed S masks are the exact original W1 inputs, never filter inputs.
    with tarfile.open(paths['masks']) as archive:
        mf = archive.extractfile('publish/masks.npz')
        if mf is None:
            raise ValueError('W1 mask archive lacks masks.npz')
        mask_bytes = mf.read()
    if hashlib.sha256(mask_bytes).hexdigest() != MASK_NPZ_SHA:
        raise ValueError('W1 fixed masks changed')
    with np.load(io.BytesIO(mask_bytes), allow_pickle=False) as z:
        masks = {k: z[k] for k in ('ticks', 'r', 's')}
    if masks['ticks'].tolist() != list(range(0, 240, 3)) or masks['s'].shape != (80, 1080 * 1920 // 8):
        raise ValueError('W1 tick or mask geometry mismatch')
    for key in ('lpips', 'dists'):
        if w6.digest(paths[key]) != WEIGHTS[key]:
            raise ValueError('metric model changed')
    nets = {'lpips': load_lpips(paths['lpips'], 'cuda'), 'dists': load_dists(paths['dists'], 'cuda')}
    if any(next(net.parameters()).device.type != 'cuda' for net in nets.values()):
        raise ValueError('metric networks are not on GPU')
    # Compare the new scoring path with W1's production score_point, on the
    # *same* independently decoded paid base and exact current source images.
    control_ticks = [0, 3]
    control_truth = [w6.plane_rgb(source, t, 1920, 1080, torch, 'cuda') for t in control_ticks]
    point = {'name': 'metric-control-base', 'path': base_ivf, 'size': (480, 272),
             'decoded_sha256': w6.digest(base_yuv)}
    w1 = w1_score.score_point(point, work, control_ticks, control_truth, masks, nets, torch)['values']
    metric_differences = []
    metric_controls = []
    for i, t in enumerate(control_ticks):
        mask = np.unpackbits(masks['s'][i])[:1080 * 1920].reshape(1080, 1920).astype(bool)
        own = player_scores(base_frame(t), control_truth[i], mask, nets)
        differences = [abs(own[0] - float(w1[1, i])), abs(own[1] - float(w1[3, i]))]
        if max(differences) > 1e-6:
            raise ValueError('GPU metric path does not reproduce W1 production score')
        metric_differences.append(differences)
        metric_controls.append({'tick': t, 'w1': [float(w1[1, i]), float(w1[3, i])], 'new': list(own)})
    scores_count = len(range(0, frames, 3))
    arrays = {'flicker': np.full((3, frames), np.nan), 'energy': np.full((3, frames), np.nan),
              'source_energy': np.full(frames, np.nan), 'lpips': np.full((3, scores_count), np.nan),
              'dists': np.full((3, scores_count), np.nan), 'ticks': np.arange(0, frames, 3)}
    prev_raw: dict[str, tuple[int, np.ndarray]] = {}
    prev_source = None
    prev_hybrid: list[np.ndarray] | None = None
    loop_started = time.monotonic()
    for t in range(frames):
        rgb = w6.plane_rgb(source, t, 1920, 1080, torch, 'cuda')
        crop_variants: list[dict[str, np.ndarray]] = [{}, {}, {}]
        player_boxes, recv = {}, {}
        for i, player in enumerate(players):
            item = crops[names[i]]
            index = np.flatnonzero(item['t'] == t)
            if len(index) == 1:
                current = item['crops'][int(index[0])]
                outputs, latest = profile_step(current, t, prev_raw.get(player))
                for k in range(3):
                    crop_variants[k][player] = outputs[k]
                assert latest is not None
                prev_raw[player] = latest
            else:
                prev_raw.pop(player, None)
            if t in boxes[i]:
                player_boxes[player] = boxes[i][t]
            recv[player] = project(predicted[i][t], boxes[i][t])
        base = base_frame(t)
        hybrids = [compose(base, cv, player_boxes, recv, order)[0] for cv in crop_variants]
        if prev_source is not None and prev_hybrid is not None:
            for k in range(3):
                residual = residual_energies(hybrids[k], prev_hybrid[k], rgb, prev_source, union[t], union[t - 1])
                if residual is not None:
                    arrays['flicker'][k, t], arrays['energy'][k, t], source_energy = residual
                    if k == 0:
                        arrays['source_energy'][t] = source_energy
                    elif abs(source_energy - arrays['source_energy'][t]) > 1e-6:
                        raise ValueError('source energy differed by variant')
        if t % 3 == 0:
            j = t // 3
            scoring_mask = np.unpackbits(masks['s'][j])[:1080 * 1920].reshape(1080, 1920).astype(bool)
            for k, image in enumerate(hybrids):
                arrays['lpips'][k, j], arrays['dists'][k, j] = player_scores(image, rgb, scoring_mask, nets)
        prev_source, prev_hybrid = rgb, hybrids
        if (t + 1) % 8 == 0 or t + 1 == frames:
            checkpoint(scratch / 'checkpoint', arrays, t + 1)
    loop_seconds = time.monotonic() - loop_started
    baseline = ctrl.comparison({'flicker': arrays['flicker'][0], 'energy': arrays['energy'][0],
                                'source_energy': arrays['source_energy']}, saved, frames)
    if not baseline['passed']:
        (scratch / 'checkpoint' / 'baseline-failure.json').write_text(json.dumps(baseline, sort_keys=True) + '\n')
        raise ValueError('literal historical baseline failed; variants remain uninterpreted')
    if not all(np.isfinite(a[:, 1:]).all() for a in (arrays['flicker'], arrays['energy'])) or not np.isfinite(arrays['source_energy'][1:]).all():
        raise ValueError('invalid adjacent-pair data')
    if not np.isfinite(arrays['lpips']).all() or not np.isfinite(arrays['dists']).all():
        raise ValueError('missing player quality scores')
    meta = json.loads((post / 'w6-judge.json').read_text())
    saved_names = [str(x) for x in saved['variants']]
    svt = [x for x in saved_names if x.startswith('svt_')]
    if len(svt) != 2:
        raise ValueError('historical SVT interpolation bracket changed')
    anchor = w6.at_rate(meta['target_kbps'], (meta['anchor_rates_kbps'][svt[0]], saved['flicker'][saved_names.index(svt[0]), :frames]),
                        (meta['anchor_rates_kbps'][svt[1]], saved['flicker'][saved_names.index(svt[1]), :frames]))
    means = {key: np.mean(value[:, 1:], axis=1).tolist() if key in ('flicker', 'energy') else np.mean(value, axis=1).tolist()
             for key, value in arrays.items() if key in ('flicker', 'energy', 'lpips', 'dists')}
    source_mean = float(np.mean(arrays['source_energy'][1:]))
    anchor_mean = float(np.mean(anchor[1:]))
    time_rule = [w6.time_check(means['flicker'][k], anchor_mean, means['energy'][k], source_mean) for k in range(3)]
    promising = [False] + [bool(means['flicker'][k] < means['flicker'][0] and
                                means['lpips'][k] <= means['lpips'][0] and
                                means['dists'][k] <= means['dists'][0]) for k in (1, 2)]
    publish = scratch / 'publish'
    publish.mkdir()
    np.savez_compressed(publish / 'series.npz', **arrays, weights=np.asarray(WEIGHTS_FIXED), anchor_flicker=anchor)
    report = {'frames': frames, 'weights': WEIGHTS_FIXED, 'baseline': baseline, 'means': means,
              'source_energy_mean': source_mean, 'anchor_flicker_mean': anchor_mean, 'original_time_rule': time_rule,
              'promising': promising, 'scope': 'one-window descriptive fixed decoder profiles; no new per-frame bits, perceptual realism or complete receiver claim',
              'inputs': INPUTS, 'source_yuv_sha256': w6.SOURCE_YUV_SHA, 'judges_sha256': JUDGES_SHA,
              'mask_npz_sha256': MASK_NPZ_SHA, 'metric_control_max_abs_differences': metric_differences,
              'metric_controls': metric_controls, 'ffmpeg_sha256': FFMPEG_SHA, 'dav1d_sha256': DAV1D_SHA,
              'crop_tracks': {name: {'frames': len(crops[name]['t']), 'first': int(crops[name]['t'][0]),
                                      'last': int(crops[name]['t'][-1]), 'npz_sha256': crop_checksums[name]}
                              for name in names},
              'code_identity': dispatch['code']['source_sha256'], 'base_max_abs_diff': max_diff,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cv2': cv2.__version__,
              'loop_seconds': loop_seconds, 'seconds': time.monotonic() - started,
              'series_sha256': w6.digest(publish / 'series.npz'), 'tick_count': scores_count}
    (publish / 'smoothing.json').write_text(json.dumps(report, sort_keys=True) + '\n')
    return 0



def audit_stage(stage: Path, *, full_external: bool) -> dict[str, Any]:
    """Audit saved arrays and identities without rerunning flow, models or codecs."""
    paths, spec, dispatch = inputs(stage)
    n = {'smoke': 16, 'full': 240}.get(stage.name)
    if n is None or full_external != (stage.name == 'full'):
        raise ValueError('audit scope must match stage')
    staging = json.loads((stage / 'staging.json').read_text())
    ready = json.loads((stage.parent / 'ready.json').read_text())
    code = dispatch['code']
    if any(code.get(k) != ready.get(k) for k in ('git_head', 'source_sha256', 'snapshot_sha256', 'spec_sha256')):
        raise ValueError('fleet dispatch and ready code identities differ')
    archive_sha = w6.digest(stage / 'published.tar')
    if staging.get('published', {}).get('sha256') != archive_sha:
        raise ValueError('published archive differs from fleet receipt')
    if full_external:
        for name, expected in INPUTS.items():
            if w6.digest(paths[name]) != expected:
                raise ValueError('full audit input digest mismatch: ' + name)
    with tarfile.open(paths['masks']) as mask_archive:
        member = mask_archive.extractfile('publish/masks.npz')
        if member is None or hashlib.sha256(member.read()).hexdigest() != MASK_NPZ_SHA:
            raise ValueError('fixed W1 scoring-mask digest mismatch')
    bin_dir = Path(staging['environment']['bin'])
    if w6.digest(bin_dir / 'ffmpeg') != FFMPEG_SHA or w6.digest(bin_dir / 'dav1d') != DAV1D_SHA:
        raise ValueError('native tool identity mismatch')
    _, saved = ctrl.saved_judges(paths['post_archive'])
    with tarfile.open(paths['post_archive']) as original:
        mf = original.extractfile('publish/w6-judge.json')
        if mf is None:
            raise ValueError('historical judge metadata absent')
        original_meta = json.load(mf)
    with tarfile.open(stage / 'published.tar') as archive:
        summary_file = archive.extractfile('publish/smoothing.json')
        series_file = archive.extractfile('publish/series.npz')
        if summary_file is None or series_file is None:
            raise ValueError('published smoothing report missing')
        report_bytes, data = summary_file.read(), series_file.read()
    report = json.loads(report_bytes)
    if hashlib.sha256(data).hexdigest() != report['series_sha256']:
        raise ValueError('series identity mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {name: z[name] for name in ('flicker', 'energy', 'source_energy', 'lpips', 'dists',
                                              'ticks', 'weights', 'anchor_flicker')}
    expected_ticks = list(range(0, n, 3))
    if (arrays['flicker'].shape != (3, n) or arrays['energy'].shape != (3, n)
            or arrays['source_energy'].shape != (n,) or arrays['anchor_flicker'].shape != (n,)
            or arrays['lpips'].shape != (3, len(expected_ticks)) or arrays['dists'].shape != arrays['lpips'].shape
            or arrays['weights'].tolist() != list(WEIGHTS_FIXED) or arrays['ticks'].tolist() != expected_ticks):
        raise ValueError('paired series geometry, profile or tick coverage differs')
    baseline = ctrl.comparison({'flicker': arrays['flicker'][0], 'energy': arrays['energy'][0],
                                'source_energy': arrays['source_energy']}, saved, n)
    if (report['frames'] != n or report['tick_count'] != len(expected_ticks) or report['inputs'] != INPUTS
            or not baseline['passed'] or report['baseline'] != baseline
            or report['source_yuv_sha256'] != w6.SOURCE_YUV_SHA or report['judges_sha256'] != JUDGES_SHA
            or report['mask_npz_sha256'] != MASK_NPZ_SHA or report['ffmpeg_sha256'] != FFMPEG_SHA
            or report['dav1d_sha256'] != DAV1D_SHA or report['code_identity'] != dispatch['code']['source_sha256']
            or report['weights'] != list(WEIGHTS_FIXED)):
        raise ValueError('baseline or source/tool/code provenance failed')
    controls = report['metric_controls']
    if len(controls) != 2 or [x['tick'] for x in controls] != [0, 3]:
        raise ValueError('W1 same-image metric controls absent')
    diffs = []
    for item in controls:
        old = np.asarray(item['w1'], np.float64)
        new = np.asarray(item['new'], np.float64)
        if old.shape != (2,) or new.shape != (2,) or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('invalid W1 metric control')
        diff = np.abs(old - new)
        if np.any(diff > 1e-6):
            raise ValueError('W1 GPU metric control failed')
        diffs.append(diff.tolist())
    if not np.allclose(diffs, report['metric_control_max_abs_differences'], rtol=0, atol=1e-12):
        raise ValueError('W1 metric control differences disagree')
    tracks = report['crop_tracks']
    if len(tracks) != 2 or any(x['frames'] != 240 or x['first'] != 0 or x['last'] != 239
                                or len(x['npz_sha256']) != 64 for x in tracks.values()):
        raise ValueError('frozen complete crop timeline record differs')
    if full_external:
        with tarfile.open(paths['generated']) as archive:
            for name, record in tracks.items():
                member = archive.extractfile('publish/' + name + '@match5.npz')
                if member is None:
                    raise ValueError('generated crop archive member absent')
                h = hashlib.sha256()
                for chunk in iter(lambda: member.read(1024 * 1024), b''):
                    h.update(chunk)
                if h.hexdigest() != record['npz_sha256']:
                    raise ValueError('generated crop member checksum mismatch')
    means = {}
    for key in ('flicker', 'energy', 'lpips', 'dists'):
        values = arrays[key][:, 1:] if key in ('flicker', 'energy') else arrays[key]
        if not np.isfinite(values).all():
            raise ValueError('missing paired measurement: ' + key)
        means[key] = np.mean(values, axis=1).tolist()
        if not np.allclose(means[key], report['means'][key], atol=1e-12, rtol=0):
            raise ValueError('reported means differ from paired series')
    if not np.isfinite(arrays['source_energy'][1:]).all():
        raise ValueError('source energy pairs missing')
    saved_names = [str(x) for x in saved['variants']]
    svt = [x for x in saved_names if x.startswith('svt_')]
    if len(svt) != 2:
        raise ValueError('historical SVT bracket changed')
    expected_anchor = w6.at_rate(original_meta['target_kbps'],
        (original_meta['anchor_rates_kbps'][svt[0]], saved['flicker'][saved_names.index(svt[0]), :n]),
        (original_meta['anchor_rates_kbps'][svt[1]], saved['flicker'][saved_names.index(svt[1]), :n]))
    if not np.array_equal(arrays['anchor_flicker'], expected_anchor, equal_nan=True):
        raise ValueError('historical SVT anchor series mismatch')
    anchor_mean = float(np.mean(expected_anchor[1:]))
    source_mean = float(np.mean(arrays['source_energy'][1:]))
    rules = [w6.time_check(means['flicker'][k], anchor_mean, means['energy'][k], source_mean) for k in range(3)]
    promising = [False] + [bool(means['flicker'][k] < means['flicker'][0] and
                               means['lpips'][k] <= means['lpips'][0] and
                               means['dists'][k] <= means['dists'][0]) for k in (1, 2)]
    if (not np.isclose(anchor_mean, report['anchor_flicker_mean'], rtol=0, atol=1e-12)
            or not np.isclose(source_mean, report['source_energy_mean'], rtol=0, atol=1e-12)
            or rules != report['original_time_rule'] or promising != report['promising']):
        raise ValueError('reported time or promising decision differs from paired series')
    result: dict[str, Any] = {'passed': True, 'checks': {'literal_baseline': True, 'W1_gpu_metric_parity': True,
              'fixed_mask_identity': True, 'source_tools_and_code': True, 'crop_tracks': True,
              'paired_series_and_decisions': True, 'all_inputs_rehashed': full_external},
              'scope': 'independent saved-array audit; no flow, model or codec execution',
              'frames': n, 'pairs': n - 1, 'ticks': len(expected_ticks), 'means': means,
              'original_time_rule': rules, 'promising': promising, 'archive_sha256': archive_sha,
              'report_sha256': hashlib.sha256(report_bytes).hexdigest(), 'series_sha256': hashlib.sha256(data).hexdigest()}
    if not full_external:
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        measured = now - clock['phase_started']
        loop = report['loop_seconds']
        if not 0 < loop <= measured:
            raise ValueError('invalid measured duration')
        projected = 1.5 * (measured - loop + loop * 240 / n)
        reserve = max(60.0, 1.25 * (clock['phase_started'] - clock['started']))
        remaining = clock['budget_end'] - now
        if projected > spec['full']['seconds'] or remaining < projected + reserve:
            raise ValueError(f'full smoothing forecast exceeds remaining budget: {projected:.1f}s')
        result['checks']['full_cost_forecast'] = True
        result['projection'] = {'smoke_seconds': measured, 'full_seconds': projected,
                                'remaining_seconds': remaining, 'reserve_seconds': reserve}
    return result


def validate(stage: Path, output: Path) -> int:
    try:
        result = audit_stage(stage, full_external=False)
        output.write_text(json.dumps(result, sort_keys=True) + '\n')
        return 0
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'smoothing_validation': False},
                                      'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True) + '\n')
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--audit-full', action='store_true')
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.audit_full:
        if args.stage is None or args.output is None:
            parser.error('--audit-full requires --stage and a new --output')
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
        result = audit_stage(args.stage, full_external=True)
        result['audit_cpu_seconds'] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
        result['audit_cpu_limit_seconds'] = 180
        with args.output.open('x') as f:
            json.dump(result, f, sort_keys=True)
            f.write('\n')
        return 0
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        return validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    if args.frames is None:
        parser.error('--frames required')
    return run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.frames)


if __name__ == '__main__':
    raise SystemExit(main())
