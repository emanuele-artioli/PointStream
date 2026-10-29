"""Paid rectangular AV1 player tiles over a native scale-four base.

Decoder inputs are the base IVF, one IVF per player and a complete compressed
rectangle header. Source R is used only to select encoder rectangles. The
receiver neither knows nor reconstructs R. CRF is the sole pilot axis.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
import shutil
from pathlib import Path
import struct
import subprocess
import tarfile
import time
from typing import Any
import zlib

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, TOOLS, read
from experiments.tennis.k6b_ideal_player import WEIGHTS, digest, image_sha, write
from experiments.tennis.k6b_native_base import SOURCES, ivf
from experiments.tennis.k6b_side_feasibility import hull

CRFS = (32, 45, 58)
WINDOWS = ('alcaraz_hurkacz/031@129', 'medvedev_struff/014@194')


def rectangles(regions: np.ndarray, width: int = 1920, height: int = 1080) -> tuple[list[list[int]], np.ndarray]:
    """Fixed even tile dimensions; per-frame positions; no decoder mask."""
    if regions.ndim != 4 or regions.shape[0] == 0 or regions.dtype != bool:
        raise ValueError('Boolean frame/player masks required')
    frames, players = regions.shape[:2]
    boxes = np.zeros((frames, players, 4), np.int32)
    for t in range(frames):
        for p in range(players):
            mask = cv2.resize(regions[t, p].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            y, x = np.nonzero(mask)
            if len(x):
                boxes[t, p] = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    sizes = []
    positions = np.zeros((frames, players, 2), np.int32)
    for p in range(players):
        b = boxes[:, p]
        w = max(64, int(np.max(b[:, 2] - b[:, 0])) + 4)
        h = max(64, int(np.max(b[:, 3] - b[:, 1])) + 4)
        w, h = min(width, (w + 1) // 2 * 2), min(height, (h + 1) // 2 * 2)
        sizes.append([w, h])
        for t in range(frames):
            x, y, xx, yy = boxes[t, p]
            # Absence is explicit: all-zero mask gets the preceding rectangle.
            if xx == 0 and t:
                positions[t, p] = positions[t - 1, p]
            else:
                positions[t, p] = [max(0, min(width - w, ((int(x + xx) - w) // 2) // 2 * 2)),
                                   max(0, min(height - h, ((int(y + yy) - h) // 2) // 2 * 2))]
    return sizes, positions


def header_encode(sizes: list[list[int]], positions: np.ndarray, fps: int) -> bytes:
    data = {'version': 1, 'canvas': [1920, 1080], 'fps': fps, 'base': [480, 272],
            'sizes': sizes, 'positions': positions.tolist(),
            'order': list(range(len(sizes))), 'color': 'limited-yuv420-cubic',
            'streams': ['base.ivf'] + [f'player-{p}.ivf' for p in range(len(sizes))]}
    raw = json.dumps(data, separators=(',', ':'), sort_keys=True).encode()
    return b'PXT1' + struct.pack('<I', len(raw)) + zlib.compress(raw, 9)


def header_decode(packet: bytes) -> dict[str, Any]:
    if packet[:4] != b'PXT1' or len(packet) < 9:
        raise ValueError('Invalid player tile header')
    raw = zlib.decompress(packet[8:])
    if len(raw) != struct.unpack('<I', packet[4:8])[0]:
        raise ValueError('Header size mismatch')
    d = json.loads(raw)
    positions = np.asarray(d['positions'])
    if (d['version'] != 1 or d['canvas'] != [1920, 1080] or d['base'] != [480, 272]
            or d['color'] != 'limited-yuv420-cubic' or d['fps'] not in (25, 50)
            or positions.ndim != 3 or positions.shape[1:] != (len(d['sizes']), 2)
            or positions.dtype.kind not in 'iu' or np.any(positions % 2)
            or d['order'] != list(range(len(d['sizes'])))
            or d['streams'] != ['base.ivf'] + [f'player-{p}.ivf' for p in range(len(d['sizes']))]):
        raise ValueError('Invalid decoder tile contract')
    for p, (w, h) in enumerate(d['sizes']):
        if (w < 2 or h < 2 or w % 2 or h % 2 or w > 1920 or h > 1080
                or np.any(positions[:, p] < 0) or np.any(positions[:, p, 0] + w > 1920)
                or np.any(positions[:, p, 1] + h > 1080)):
            raise ValueError('Invalid decoder rectangle')
    return d


def crop_yuv(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if frame.shape != (1620, 1920) or frame.dtype != np.uint8 or any(v % 2 for v in (x, y, w, h)):
        raise ValueError('Native even YUV420 crop required')
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1920 or y + h > 1080:
        raise ValueError('Crop outside canvas')
    chroma = frame[1080:].reshape(2, 540, 960)
    return np.concatenate([frame[y:y+h, x:x+w].ravel(),
                           chroma[:, y//2:(y+h)//2, x//2:(x+w)//2].ravel()]).reshape(h*3//2, w)


def composite(base: np.ndarray, tiles: list[np.ndarray], header: dict[str, Any], t: int) -> np.ndarray:
    if base.shape != (1080, 1920, 3) or base.dtype != np.uint8 or len(tiles) != len(header['sizes']):
        raise ValueError('Decoder canvas/tiles mismatch')
    result = base.copy()
    for p in header['order']:
        w, h = header['sizes'][p]
        x, y = header['positions'][t][p]
        if tiles[p].shape != (h, w, 3) or tiles[p].dtype != np.uint8:
            raise ValueError('Decoded tile geometry mismatch')
        result[y:y+h, x:x+w] = tiles[p]
    return result


def run(a: argparse.Namespace) -> None:
    import torch
    from experiments.jobs.monitor import publish_progress
    from experiments.tennis.k4b_features import convert_command
    from experiments.tennis.k6 import downscaled_source
    from experiments.tennis.k6b_rate_streams import decode_masks
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, yuv420_to_rgb

    began = time.monotonic()
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualified A6000 required')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.cuda.set_per_process_memory_fraction(5000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    identities = {k: digest(getattr(a, k)) for k in ('al', 'med', 'transport', 'prior', 'images', 'lpips', 'dists')}
    expected = {'al': SOURCES[0][3], 'med': SOURCES[1][3], 'transport': INPUTS['transport'],
                'prior': INPUTS['prior'], 'images': INPUTS['images'], **WEIGHTS}
    if identities != expected:
        raise ValueError('Frozen input identity changed')
    tools = {name: svtav1.tool(name) for name in TOOLS}
    if any(tools[n]['sha256'] != s for n, s in TOOLS.items()):
        raise ValueError('Native tool identity changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
    if not all(next(n.parameters()).is_cuda for n in nets.values()):
        raise ValueError('Metric networks not on CUDA')
    with tarfile.open(a.prior) as tar:
        old_row = json.load(read(tar, 'publish/quality.json'))['rows'][0]
        with np.load(read(tar, 'publish/alcaraz_hurkacz__031@129/scores.npz')) as z:
            old_scores = {m: z['full_' + m].copy() for m in WEIGHTS}
        curves = {m: hull([(r['base_kbps'], -float(old_scores[m][2*i].mean()))
                           for i, r in enumerate(old_row['rates'])]) for m in WEIGHTS}
    with tarfile.open(a.images) as tar:
        golden = json.load(read(tar, 'manifest.json'))
    control_errors: dict[str, float] = {}
    with tarfile.open(a.images) as tar:
        for t in (0, 3, 6):
            with np.load(read(tar, f'frames/{t:03d}.npz')) as z:
                original = z['images'].copy()
            if [image_sha(im) for im in original] != golden['hashes'][str(t)]:
                raise ValueError('Saved control image identity mismatch')
            mask = np.ones((1080, 1920), bool)
            for mi, metric in enumerate(WEIGHTS):
                observed = []
                for im in original[1:3]:
                    if metric == 'lpips':
                        value = lpips_reduce(nets[metric], [im], original[0], [[mask]], 'cuda')[0][0]
                    else:
                        value = dists_crops(nets[metric], [im], original[0], mask, 'cuda')[0]
                    observed.append(value)
                error = float(np.max(np.abs(np.asarray(observed)-old_scores[metric][2:4, t//3])))
                control_errors[metric] = max(control_errors.get(metric, 0.), error)
                if not np.isfinite(error) or error > 1e-4:
                    raise ValueError('Frozen base/actual neural metric control mismatch')
    fixed_seconds = time.monotonic() - began
    command_log: list[dict[str, Any]] = []
    rows = []
    ticks = list(range(0, a.frames, 3))
    full_mask = np.ones((1080, 1920), bool)
    cpu_affinity = min(os.sched_getaffinity(0))

    def command(cmd: list[str], timeout: int = 300) -> None:
        # Bound encoder subprocess pools without constraining CUDA metric threads.
        native_command = ['taskset', '-c', str(cpu_affinity), *cmd]
        started = time.monotonic()
        subprocess.run(native_command, check=True, capture_output=True, timeout=timeout)
        command_log.append({'argv': native_command, 'seconds': time.monotonic()-started})

    def rgb(plane: np.ndarray, w: int, h: int) -> np.ndarray:
        gpu = torch.from_numpy(np.array(plane, copy=True)[None]).to('cuda')
        return yuv420_to_rgb(gpu, w, h, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    for wi, (name, start, fps, _, source_sha) in enumerate(SOURCES):
        work = scratch / name
        work.mkdir()
        out = publish / name
        out.mkdir()
        fixed_began = time.monotonic()
        raw_path = work / 'source.yuv'
        command(convert_command('ffmpeg', getattr(a, name), raw_path, start, 240))
        if digest(raw_path) != source_sha:
            raise ValueError('Full source YUV identity mismatch')
        raw = np.memmap(raw_path, np.uint8, 'r', shape=(240, 1620, 1920))
        with tarfile.open(a.transport) as tar:
            _, r = decode_masks(read(tar, f'packets/{WINDOWS[wi].replace("/", "__")}/exact-rle.matte').getvalue())
        if len(r) != 240:
            raise ValueError('Incomplete source regions')
        # Fix sizes from the full window in both smoke/full, avoiding a misleading smoke geometry.
        sizes, positions = rectangles(r)
        packet = header_encode(sizes, positions[:a.frames], fps)
        header = header_decode(packet)
        if header['positions'] != positions[:a.frames].tolist():
            raise ValueError('Independent geometry decode failed')
        (out / 'geometry.pxt').write_bytes(packet)
        source_fixed_seconds = time.monotonic()-fixed_began
        prep_began = time.monotonic()
        down = downscaled_source(raw_path, a.frames, 4, work / 'base-source.yuv')
        base_stream, base_yuv = out / 'base.ivf', work / 'base.yuv'
        command(svtav1.encode_command('SvtAv1EncApp', down, base_stream, width=480, height=272,
                                     fps=Fraction(fps), frames=a.frames, crf=63, preset=4, full_range=False))
        command(svtav1.decode_command('dav1d', base_stream, base_yuv, threads=1))
        base_info = ivf(base_stream, a.frames, 480, 272, fps)
        base_repeat = work / 'base-repeat.yuv'
        command(svtav1.decode_command('dav1d', base_stream, base_repeat, threads=1))
        if base_yuv.stat().st_size != a.frames*480*272*3//2 or digest(base_yuv) != digest(base_repeat):
            raise ValueError('Independent native base decode failed')
        base = np.memmap(base_yuv, np.uint8, 'r', shape=(a.frames, 408, 480))
        tile_sources = []
        for p, (w, h) in enumerate(sizes):
            path = work / f'tile-{p}.yuv'
            with path.open('xb') as f:
                for t in range(a.frames):
                    x, y = positions[t, p]
                    f.write(crop_yuv(raw[t], int(x), int(y), w, h).tobytes())
            tile_sources.append(path)
        scores = {m: np.full((4, len(ticks)), np.nan) for m in WEIGHTS}
        decoded_policies = []
        rates = []
        for crf in CRFS:
            policy = out / f'q{crf}'
            policy.mkdir()
            # Each operating point is a complete independently installable payload.
            (policy / 'geometry.pxt').write_bytes(packet)
            shutil.copyfile(base_stream, policy / 'base.ivf')
            streams = []
            decs = []
            for p, (w, h) in enumerate(sizes):
                stream = policy / f'player-{p}.ivf'
                decoded = work / f'q{crf}-{p}.yuv'
                command(svtav1.encode_command('SvtAv1EncApp', tile_sources[p], stream, width=w, height=h,
                                             fps=Fraction(fps), frames=a.frames, crf=crf, preset=4, full_range=False))
                command(svtav1.decode_command('dav1d', stream, decoded, threads=1))
                info = ivf(stream, a.frames, w, h, fps)
                again = work / f'q{crf}-{p}-repeat.yuv'
                command(svtav1.decode_command('dav1d', stream, again, threads=1))
                if digest(decoded) != digest(again) or decoded.stat().st_size != a.frames*w*h*3//2:
                    raise ValueError('Independent native tile decode failed')
                streams.append(info)
                decs.append(np.memmap(decoded, np.uint8, 'r', shape=(a.frames, h*3//2, w)))
            decoded_policies.append(decs)
            complete_bytes = len(packet) + base_info['bytes'] + sum(s['bytes'] for s in streams)
            rates.append({'crf': crf, 'bytes': complete_bytes, 'base_bytes': base_info['bytes'],
                          'header_bytes': len(packet), 'tile_streams': streams,
                          'kbps': complete_bytes*8/(a.frames/fps)/1000,
                          'free_tile_payload_bound_kbps': (len(packet)+base_info['bytes'])*8/(a.frames/fps)/1000})
        prep_seconds = time.monotonic()-prep_began
        per_tick = []
        hashes = {}
        for j, t in enumerate(ticks):
            tick_began = time.monotonic()
            truth = rgb(raw[t], 1920, 1080)
            if name == 'al' and image_sha(truth) != golden['hashes'][str(t)][0]:
                raise ValueError('Original CUDA source RGB identity mismatch')
            background = cv2.resize(rgb(base[t], 480, 272), (1920, 1080), interpolation=cv2.INTER_CUBIC)
            images = [background]
            for decs in decoded_policies:
                tiles = [rgb(d[t], *sizes[p]) for p, d in enumerate(decs)]
                images.append(composite(background, tiles, header, t))
            hashes[str(t)] = [image_sha(truth)] + [image_sha(im) for im in images]
            if t in (0, 3, 6):
                np.savez_compressed(out / f'replay-{t:03d}.npz', source_yuv=np.asarray(raw[t]),
                                    base_yuv=np.asarray(base[t]), images=np.stack([truth, *images]))
            for pi, image in enumerate(images):
                scores['lpips'][pi, j] = lpips_reduce(nets['lpips'], [image], truth, [[full_mask]], 'cuda')[0][0]
                scores['dists'][pi, j] = dists_crops(nets['dists'], [image], truth, full_mask, 'cuda')[0]
            if not all(np.isfinite(v[:, :j+1]).all() for v in scores.values()):
                raise ValueError('Nonfinite scores')
            np.savez_compressed(out / 'scores.npz', ticks=np.array(ticks[:j+1]), **{m: v[:, :j+1] for m, v in scores.items()})
            torch.cuda.synchronize()
            per_tick.append(time.monotonic()-tick_began)
            publish_progress(os.environ['PS_STAGE'], wi*len(ticks)+j+1)
        control_error = None
        if name == 'al' and a.frames == 240:
            control_error = {m: float(np.max(np.abs(scores[m][0]-old_scores[m][2]))) for m in WEIGHTS}
            if any(v > 1e-4 for v in control_error.values()):
                raise ValueError('Historical scale-four base scores do not reproduce')
        row = {'window': WINDOWS[wi], 'source_yuv_sha256': source_sha, 'frames': a.frames, 'fps': fps,
               'source_fixed_seconds': source_fixed_seconds, 'native_base_repeat_matches': True,
               'sizes': sizes, 'header_sha256': digest(out/'geometry.pxt'), 'rates': rates,
               'means': {m: v.mean(1).tolist() for m, v in scores.items()}, 'ticks': ticks,
               'hashes': hashes, 'base_ivf': base_info, 'scalable_prep_seconds': prep_seconds,
               'per_tick_seconds': per_tick, 'historical_base_control_error': control_error,
               'svt_curves': curves if name == 'al' else None}
        write(out / 'row.json', row)
        rows.append(row)
    summary = {'control_errors': control_errors, 'fixed_setup_seconds': fixed_seconds,
               'commands': command_log, 'metric_entrypoints': ['lpips_reduce/full-mask', 'dists_crops/full-mask'],
               'rows': rows, 'frames': a.frames, 'crfs': list(CRFS), 'inputs': identities,
               'tools': tools, 'seconds': time.monotonic()-began, 'cuda': True,
               'gpu': torch.cuda.get_device_name(0), 'timing_secondary': True,
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
               'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
               'decoder_mask_required': False, 'unbilled_side_components': [],
               'scope': 'Two familiar development windows; paid native rectangular player tiles, no training.'}
    write(publish / 'pixels.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'pixels-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    s = json.loads((stage/'pixels-summary.json').read_text())
    spec = json.loads((stage.parent/'spec.json').read_text())
    execution = json.loads((stage/'execution.json').read_text())
    staging = json.loads((stage/'staging.json').read_text())
    # Full source decode, full-window sizing, imports and historical controls are fixed.
    # Scale native tile/base preparation10x and use the slower warm metric ticks.
    measured_fixed = s['fixed_setup_seconds'] + sum(r['source_fixed_seconds'] for r in s['rows'])
    measured_variable = sum(r['scalable_prep_seconds'] + sum(r['per_tick_seconds']) for r in s['rows'])
    untraced = max(0., execution['seconds'] - measured_fixed - measured_variable)
    projected_work = measured_fixed + untraced + sum(
        r['scalable_prep_seconds']*10 + max(r['per_tick_seconds'][1:])*80 for r in s['rows'])
    projection = (projected_work + staging['publication_seconds']*10)*1.35
    checks = {'cuda': s['cuda'] and 'A6000' in s['gpu'], 'frames': s['frames'] == 24,
              'two_windows': [r['window'] for r in s['rows']] == list(WINDOWS),
              'crfs': s['crfs'] == list(CRFS), 'memory': 0 < s['peak_allocated_mib'] <= s['peak_reserved_mib'] < 5000,
              'timing_secondary': s['timing_secondary'],
              'controls': set(s['control_errors']) == set(WEIGHTS) and all(0 <= e <= 1e-4 for e in s['control_errors'].values()),
              'fresh_native_bases': all(r['native_base_repeat_matches'] for r in s['rows']),
              'projection': projection < spec['full']['seconds'],
              'no_unpaid_side': s['unbilled_side_components'] == [] and s['decoder_mask_required'] is False,
              'scales': spec['scale']['frames'] == {'smoke': 24, 'full': 240},
              'allocation': spec['budget_seconds'] <= 2700,
              'weights': all(s['inputs'][m] == v for m, v in WEIGHTS.items()),
              'input_identity': all(s['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs']
                                    if x['name'] in s['inputs'])}

    with tarfile.open(stage/'published.tar') as tar:
        checks['saved_summary'] = json.load(read(tar, 'publish/pixels.json')) == s
        for wi, name in enumerate(('al', 'med')):
            r = s['rows'][wi]
            with np.load(read(tar, f'publish/{name}/scores.npz')) as z:
                checks[name+':scores'] = z['ticks'].tolist() == list(range(0, 24, 3)) and all(z[m].shape == (4, 8) and np.isfinite(z[m]).all() for m in WEIGHTS)
            for t in (0, 3, 6):
                with np.load(read(tar, f'publish/{name}/replay-{t:03d}.npz')) as z:
                    checks[f'{name}:{t}:saved_pixels'] = (
                        z['images'].shape == (5, 1080, 1920, 3)
                        and [image_sha(im) for im in z['images']] == r['hashes'][str(t)]
                        and z['source_yuv'].shape == (1620, 1920) and z['base_yuv'].shape == (408, 480))
            h = read(tar, f'publish/{name}/geometry.pxt').getvalue()
            checks[name+':header'] = header_decode(h)['sizes'] == r['sizes']
            for rate in r['rates']:
                paths = [f'publish/{name}/q{rate["crf"]}/base.ivf', f'publish/{name}/q{rate["crf"]}/geometry.pxt'] + [f'publish/{name}/q{rate["crf"]}/player-{p}.ivf' for p in range(len(r['sizes']))]
                checks[f'{name}:{rate["crf"]}:bytes'] = sum(tar.getmember(p).size for p in paths) == rate['bytes']
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate', action='store_true')
    for name in ('al', 'med', 'transport', 'prior', 'images', 'lpips', 'dists'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--frames', type=int, choices=(24, 240), default=24)
    a = p.parse_args()
    if a.validate:
        validate()
    elif any(getattr(a, k) is None for k in ('al', 'med', 'transport', 'prior', 'images', 'lpips', 'dists')):
        p.error('All pinned input files required')
    else:
        run(a)


if __name__ == '__main__':
    main()
