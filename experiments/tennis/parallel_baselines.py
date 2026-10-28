"""Paid whole-frame DCVC-UF and encoder-ROI SVT development comparators.

Original limited-range source samples are coded directly. ROI uses frozen R
only at the encoder; AV1 carries its quantizer decisions in the paid stream.
These are development curves, not untouched confirmation or timing evidence.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from typing import Any

import cv2
import numpy as np

from experiments.tennis.k6b_clean_background_prep import INPUTS, TOOLS, read
from experiments.tennis.k6b_ideal_player import WEIGHTS, digest, image_sha, write
from experiments.tennis.k6b_native_base import SOURCES

WINDOWS = {'al': 'alcaraz_hurkacz/031@129', 'med': 'medvedev_struff/014@194'}
CHECKPOINTS = {'image_ckpt': 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302',
               'video_ckpt': '934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3'}
PREPARED_SHA = '56f54ccdc9db75ce98796597d2f2de0c849ad8041ab57c2def28dc5a4cd25ae0'
# Three coarse points, fixed before viewing their outcomes.
POINTS = [('dcvc', 0, 4), ('dcvc', 21, 2), ('dcvc', 42, 1),
          ('svt_roi', 43, 4), ('svt_roi', 53, 4), ('svt_roi', 63, 4)]


def configured_points(roi_only: bool) -> list[tuple[str, int, int]]:
    # Independent unfinished comparator, never a retry of failed DCVC pixels.
    # The paid native-resolution encode adds coverage above the old33.685kbps
    # ceiling; its setting is frozen before seeing its new quality outcomes.
    return ([p for p in POINTS if p[0] == 'svt_roi'] + [('svt_plain', 63, 1)]
            if roi_only else POINTS)


def union_regions(regions: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if (regions.shape != (240, 2, 540, 960) or regions.dtype != bool
            or metadata['output_shape'] != [1080, 1920] or metadata['native_grid_factor'] != 1):
        raise ValueError('Original two-player half-resolution R transport required')
    return regions.any(axis=1)


def roi_text(regions: np.ndarray, width: int, height: int) -> str:
    """Positive32 outside every64x64block touched by frozen source R."""
    if regions.ndim != 3 or regions.dtype != bool or width % 2 or height % 2:
        raise ValueError('Boolean native regions and even coded geometry required')
    rows, cols = (height + 63) // 64, (width + 63) // 64
    lines = []
    for t, region in enumerate(regions):
        resized = cv2.resize(region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        padded = np.zeros((rows * 64, cols * 64), bool)
        padded[:height, :width] = resized
        inside = padded.reshape(rows, 64, cols, 64).any(axis=(1, 3))
        lines.append(str(t) + ' ' + ' '.join('0' if x else '32' for x in inside.ravel()))
    return '\n'.join(lines) + '\n'


def cost_projection(setup: float, native: float, scoring: float, publication: float) -> float:
    values = np.asarray([setup, native, scoring, publication], float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Finite nonnegative complete timings required')
    #24/240 native frames,3/80 judged frames; fixed setup charged once.
    return float(1.35 * (setup + native * 10 + scoring * 80 / 3 + publication * 4))


def native_dcvc(a: argparse.Namespace, raw: Path, work: Path, n: int, w: int, h: int, qp: int) -> dict[str, Any]:
    from experiments.tennis.k5 import luma_psnr
    from src.codecs.dcvc_uf_worker import dcvc_command, unpack_container
    plan = {'structure': 'htl', 'qp': qp, 'frame_count': n, 'height': h, 'width': w,
            'src_type': 'yuv420', 'frames_file': str(raw), 'container': str(work / 'stream.psdc'),
            'out_file': str(work / 'decoded.yuv'), 'image_ckpt': str(a.image_ckpt),
            'image_sha256': CHECKPOINTS['image_ckpt'], 'video_ckpt': str(a.video_ckpt),
            'video_sha256': CHECKPOINTS['video_ckpt'], 'profile': False}
    write(work / 'plan.json', plan)
    reports = {}
    commands = {}
    startup_seconds = 0.
    for action in ('encode', 'decode'):
        command, env, cwd = dcvc_command(Path(sys.prefix), action, work / 'plan.json', work / f'{action}.json')
        process_started = time.monotonic()
        subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, check=True, timeout=600)
        process_seconds = time.monotonic() - process_started
        commands[action] = command
        report = json.loads((work / f'{action}.json').read_text())
        reports[action] = report
        measured_work = (report['encode_seconds'] if action == 'encode' else
                         sum(p['decode_seconds'] for p in report['passes']))
        startup_seconds += max(0., process_seconds - report['load_seconds'] - measured_work)
    encoded, decoded = reports['encode'], reports['decode']
    header, _ = unpack_container((work / 'stream.psdc').read_bytes())
    if (header != {'structure': 'htl', 'frame_count': n} or not decoded['deterministic']
            or decoded['passes'][0]['frame_sha256'][0] != encoded['i_recon_sha256']
            or len(decoded['passes'][0]['frame_sha256']) != n
            or digest(work / 'stream.psdc') != encoded['container_sha256']
            or not 0 < encoded['peak_allocated_mib'] < 5000):
        raise ValueError('Native DCVC bytes/count/determinism/intra/memory gate failed')
    for report in reports.values():
        runtime = report['environment']
        if ('A6000' not in runtime['device_name'] or runtime['capability'] != [8, 6]
                or runtime['extension']['variant'] != 'sm80'
                or not report['checkpoints']['strict_load']
                or report['checkpoints']['precision'] != 'float16'):
            raise ValueError('Native DCVC execution provider/kernel/load contract changed')
    psnr = luma_psnr(work / 'decoded.yuv', raw, w, h, n)
    if not np.isfinite(psnr) or psnr < 12:
        raise ValueError('Native DCVC source sanity floor failed')
    frame_bytes = w * h * 3 // 2
    with (work / 'decoded.yuv').open('rb') as stream:
        observed = [__import__('hashlib').sha256(stream.read(frame_bytes)).hexdigest() for _ in range(n)]
        if stream.read(1) or observed != decoded['passes'][0]['frame_sha256']:
            raise ValueError('Saved DCVC pixels differ from independent decoder reports')
    return {'stream': work / 'stream.psdc', 'decoded': work / 'decoded.yuv', 'reports': reports,
            'commands': commands, 'startup_seconds': startup_seconds,
            'peak_allocated_mib': encoded['peak_allocated_mib'], 'native_luma_psnr': psnr}


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
    for key, expected in {**CHECKPOINTS, **WEIGHTS, 'transport': INPUTS['transport'], 'prepared': PREPARED_SHA}.items():
        if digest(getattr(a, key)) != expected:
            raise ValueError('Pinned input changed: ' + key)
    tools = {key: svtav1.tool(key) for key in TOOLS}
    if any(tools[key]['sha256'] != expected for key, expected in TOOLS.items()):
        raise ValueError('Pinned native tool changed')
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir()
    prepared = scratch / 'prepared'
    prepared.mkdir()
    with tarfile.open(a.prepared) as outer:
        with tarfile.open(fileobj=read(outer, 'publish/input.tar')) as inner:
            inner.extractall(prepared, filter='data')
    golden = json.loads((prepared / 'manifest.json').read_text())
    if golden['frames'] != 240 or golden['source_yuv_sha256'] != SOURCES[0][4]:
        raise ValueError('Original golden cohort changed')
    for name, expected in golden['files'].items():
        if digest(prepared / name) != expected:
            raise ValueError('Original golden member changed')
    inputs_checked = {key: digest(getattr(a, key)) for key in ('al', 'med', 'prepared', 'transport', 'image_ckpt', 'video_ckpt', 'lpips', 'dists')}
    n, judged = a.frames, 3 if a.frames == 24 else 80
    ticks = list(range(0, 240, 3))[:judged]
    rows: list[dict[str, Any]] = []
    native_seconds = scoring_seconds = 0.
    control_errors: dict[str, list[float]] = {metric: [] for metric in WEIGHTS}
    source_hashes: dict[str, list[str]] = {}
    peaks = []
    all_mask = np.ones((1080, 1920), bool)
    points = configured_points(a.roi_only)
    for name, start, fps, source_hash, yuv_hash in SOURCES:
        if inputs_checked[name] != source_hash:
            raise ValueError('Source video changed: ' + name)
        work = scratch / name
        work.mkdir()
        full = work / 'source-full.yuv'
        conversion = convert_command('ffmpeg', getattr(a, name), full, start, 240)
        subprocess.run(conversion, check=True, capture_output=True, timeout=180)
        if digest(full) != yuv_hash:
            raise ValueError('Native source YUV identity changed')
        native_source = np.memmap(full, np.uint8, 'r', shape=(240, 1620, 1920))
        with tarfile.open(a.transport) as tar:
            mask_meta, masks = decode_masks(read(tar, 'packets/' + WINDOWS[name].replace('/', '__') + '/exact-rle.matte').getvalue())
        regions = union_regions(masks, mask_meta)
        del masks
        source = work / 'source.yuv'
        with source.open('wb') as output:
            output.write(native_source[:n].tobytes())
        del native_source
        recons = []
        for codec, qp, scale in points:
            point = work / f'{codec}-{qp}-{scale}'
            point.mkdir()
            width, height = {1: (1920, 1080), 2: (960, 544), 4: (480, 272)}[scale]
            raw = downscaled_source(source, n, scale, point / 'input.yuv') if scale != 1 else source
            started = time.monotonic()
            if codec == 'dcvc':
                native = native_dcvc(a, raw, point, n, width, height, qp)
                peaks.append(native['peak_allocated_mib'])
            else:
                roi = point / 'roi.txt'
                if codec == 'svt_roi':
                    roi.write_text(roi_text(regions[:n], width, height))
                cpus = sorted(os.sched_getaffinity(0))[:4]
                rec = svtav1.code(raw, point, width=width, height=height, fps=Fraction(fps), frames=n,
                                  crf=qp, preset=4, threads=4, full_range=False, encoder='SvtAv1EncApp',
                                  decoder='dav1d', timeout=500, cpus=cpus,
                                  extra=['--roi-map-file', str(roi)] if codec == 'svt_roi' else [])
                again = point / 'fresh.yuv'
                subprocess.run(svtav1.decode_command('dav1d', Path(rec['stream']), again, threads=4), check=True, capture_output=True, timeout=60)
                if digest(again) != digest(Path(rec['decoded'])) or len(svtav1.ivf_frames(Path(rec['stream']).read_bytes())) != n:
                    raise ValueError('Independent native SVT decode/count failed')
                native = {'stream': Path(rec['stream']), 'decoded': Path(rec['decoded']), 'reports': rec,
                          'roi_sha256': digest(roi) if codec == 'svt_roi' else None,
                          'roi_grant': 'encoder-only frozen source R; no decoder mask' if codec == 'svt_roi' else None}
            elapsed = time.monotonic() - started
            # Checkpoint setup is paid once per invocation in both stages; do
            # not expand that fixed cost tenfold with the frame count.
            fixed_load = (sum(report['load_seconds'] for report in native['reports'].values())
                          + native['startup_seconds'] if codec == 'dcvc' else 0.)
            native_seconds += max(0., elapsed - fixed_load)
            if native['decoded'].stat().st_size != n * width * height * 3 // 2:
                raise ValueError('Native decoded frame count failed')
            stream = publish / f'{name}-{codec}-{qp}-{scale}{native["stream"].suffix}'
            stream.write_bytes(native['stream'].read_bytes())
            row = {'window': name, 'codec': codec, 'qp': qp, 'scale': scale, 'frames': n, 'size': [width, height],
                   'source_sha256': source_hash, 'source_yuv_sha256': yuv_hash, 'source_command': conversion,
                   'stream': stream.name, 'stream_sha256': digest(stream), 'bytes': stream.stat().st_size,
                   'rate_kbps': stream.stat().st_size * 8 * fps / n / 1000, 'native_seconds': elapsed,
                   'native': {k: v for k, v in native.items() if k not in ('stream', 'decoded')}, 'decoded_sha256': digest(native['decoded'])}
            recons.append((row, native['decoded']))
            write(publish / 'partial-native.json', [x[0] for x in recons])
            publish_progress(os.environ['PS_STAGE'], len(rows) + len(recons))
        # All codec subprocesses finish before loading metric networks.
        nets = {'lpips': load_lpips(a.lpips, 'cuda'), 'dists': load_dists(a.dists, 'cuda')}
        if not all(next(net.parameters()).is_cuda for net in nets.values()):
            raise ValueError('CUDA metric path required')
        raw_source = np.memmap(source, np.uint8, 'r', shape=(n, 1620, 1920))
        source_hashes[name] = []
        started = time.monotonic()

        def rgb(plane: np.ndarray, width: int, height: int) -> np.ndarray:
            value = torch.from_numpy(np.array(plane, copy=True)[None]).cuda()
            return yuv420_to_rgb(value, width, height, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()

        sources = [rgb(raw_source[t], 1920, 1080) for t in ticks]
        for j, tick in enumerate(ticks):
            source_hashes[name].append(image_sha(sources[j]))
            if name == 'al':
                if image_sha(sources[j]) != golden['hashes'][str(tick)]['images'][0]:
                    raise ValueError('Exact source CUDA RGB control failed')
                if j < 3:
                    with np.load(prepared / 'frames' / f'{tick:03d}.npz') as data:
                        base = data['images'][1]
                    observations = {'lpips': lpips_reduce(nets['lpips'], [base], sources[j], [[all_mask]], 'cuda')[0][0],
                                    'dists': dists_crops(nets['dists'], [base], sources[j], all_mask, 'cuda')[0]}
                    for metric, observed in observations.items():
                        error = abs(float(observed) - golden['controls'][metric][0][j])
                        control_errors[metric].append(error)
                        if error > 1e-4:
                            raise ValueError('Exact original metric control failed')
        for row, decoded in recons:
            width, height = row['size']
            pixels = np.memmap(decoded, np.uint8, 'r', shape=(n, height * 3 // 2, width))
            scores: dict[str, list[float]] = {metric: [] for metric in WEIGHTS}
            hashes = []
            for j, tick in enumerate(ticks):
                image = cv2.resize(rgb(pixels[tick], width, height), (1920, 1080), interpolation=cv2.INTER_CUBIC)
                hashes.append(image_sha(image))
                scores['lpips'].append(float(lpips_reduce(nets['lpips'], [image], sources[j], [[all_mask]], 'cuda')[0][0]))
                scores['dists'].append(float(dists_crops(nets['dists'], [image], sources[j], all_mask, 'cuda')[0]))
            if not all(np.isfinite(v).all() for v in scores.values()):
                raise ValueError('Nonfinite score')
            row.update({'ticks': ticks, 'scores': scores, 'mean': {m: float(np.mean(v)) for m, v in scores.items()}, 'rgb_sha256': hashes})
            rows.append(row)
            write(publish / 'curves.json', rows)
            publish_progress(os.environ['PS_STAGE'], len(rows))
        scoring_seconds += time.monotonic() - started
        del nets, sources
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    peaks.append(torch.cuda.max_memory_allocated() / 2**20)
    seconds = time.monotonic() - began
    summary = {'roi_only': a.roi_only, 'points': points,
               'frames': n, 'ticks': ticks, 'rows': rows, 'inputs': inputs_checked, 'tools': tools,
               'source_rgb_sha256': source_hashes, 'control_errors': control_errors,
               'cuda': True, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'peak_allocated_mib': max(peaks), 'native_seconds': native_seconds, 'scoring_seconds': scoring_seconds,
               'setup_seconds': max(0., seconds - native_seconds - scoring_seconds), 'seconds': seconds,
               'scope': 'Two familiar development windows; complete native stream bytes; encoder R oracle for ROI only; no timing claim.'}
    write(publish / 'summary.json', summary)
    write(Path(os.environ['PS_STAGE_DIR']) / 'baseline-summary.json', summary)


def validate() -> None:
    stage = Path(os.environ['PS_STAGE_DIR'])
    summary = json.loads((stage / 'baseline-summary.json').read_text())
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staging = json.loads((stage / 'staging.json').read_text())
    roi_only = '--roi-only' in spec['arguments']
    expected_points = configured_points(roi_only)
    projection = cost_projection(summary['setup_seconds'], summary['native_seconds'], summary['scoring_seconds'], staging['publication_seconds'])
    checks = {'cohort': summary['frames'] == 24 and summary['ticks'] == [0, 3, 6],
              'rows': len(summary['rows']) == 2 * len(expected_points) and {r['window'] for r in summary['rows']} == {'al', 'med'},
              'point_grid': summary['roi_only'] == roi_only and all([(r['codec'], r['qp'], r['scale']) for r in summary['rows'] if r['window'] == w] == expected_points for w in WINDOWS),
              'scores': all(set(r['scores']) == set(WEIGHTS) and all(len(v) == 3 and np.isfinite(v).all() for v in r['scores'].values()) for r in summary['rows']),
              'controls': all(len(v) == 3 and max(v) <= 1e-4 for v in summary['control_errors'].values()),
              'memory': 0 < summary['peak_allocated_mib'] < 5000,
              'cuda': summary['cuda'] and 'A6000' in summary['gpu'],
              'projection': projection < spec['full']['seconds'],
              'budget': spec['budget_seconds'] <= 1800,
              'scale': spec['scale']['frames'] == {'smoke': 24, 'full': 240},
              'inputs': all(summary['inputs'][x['name']] == x['sha256'] for x in spec['staged_inputs'])}
    with tarfile.open(stage / 'published.tar') as tar:
        checks['published_summary'] = json.load(read(tar, 'publish/summary.json')) == summary
        checks['published_streams'] = all(__import__('hashlib').sha256(read(tar, 'publish/' + r['stream']).getvalue()).hexdigest() == r['stream_sha256'] for r in summary['rows'])
    write(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks, 'projected_full_seconds': projection})
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--roi-only', action='store_true', help='Complete independent SVT comparators; never invokes DCVC')
    for key in ('al', 'med', 'prepared', 'transport', 'image_ckpt', 'video_ckpt', 'lpips', 'dists'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path)
    parser.add_argument('--frames', type=int, choices=(24, 240), default=24)
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        if any(getattr(args, key) is None for key in ('al', 'med', 'prepared', 'transport', 'image_ckpt', 'video_ckpt', 'lpips', 'dists')):
            parser.error('All frozen source, checkpoint, control and transport inputs required')
        run(args)


if __name__ == '__main__':
    main()
