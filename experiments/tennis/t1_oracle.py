"""T1 rate-only oracle: actual split reconstructions with foreground payload cost relaxed to zero."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tennis import k4, k6
from experiments.tennis.k4b_features import convert
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.pointstream import bitstream as bs, composite as cp, mask as mk
from src.segmentation.masks import ClipMasks

# Same three coarse points on the background and whole-frame anchors.
GRID = [('svt', 62, 1), ('svt', 62, 2), ('svt', 55, 4),
        ('dcvc', 42, 1), ('dcvc', 21, 2), ('dcvc', 0, 4)]
FG = [('dcvc', 0, 4), ('dcvc', 21, 2), ('dcvc', 42, 1)]
METRICS = {'wlpips': -1, 'wdists': -1, 'frame_lpips': -1, 'frame_dists': -1, 'wpsnr': 1}


def container_bytes(bg: bytes, fg: bytes, mask: bytes, bg_codec: str, bg_size: tuple[int, int],
                    fg_size: tuple[int, int], scale: int, fps: Fraction, frames: int) -> bytes:
    container = bs.Container(1920, 1080, fps, frames)
    codec = 'av1' if bg_codec == 'svt' else 'dcvc-uf'
    container.add_coded(bs.Coded('background', bs.ROUTES[bg_codec], codec, *bg_size, bg))
    container.add_coded(bs.Coded('foreground', scale, 'dcvc-uf', *fg_size, fg, ((cp.HAND, 0), (cp.OBJECT, 0))))
    container.add(bs.Tag.MASK, mask)
    packed = container.pack()
    if bs.Container.unpack(packed).pack() != packed:
        raise ValueError('container roundtrip mismatch')
    return packed


def relaxed_bytes(actual: bytes, foreground_payload: bytes) -> int:
    """Keep actual container framing, including its length fields; grant only FG codec payload."""
    return len(actual) - len(foreground_payload)


def reference(args: argparse.Namespace, wid: str, source: Path, work: Path) -> dict[str, Any]:
    import cv2
    from experiments.background.g5 import rgb_to_yuv420
    from experiments.visor.h3 import rgb_memmap

    entry = json.loads((args.inputs / 'inputs.json').read_text())['windows'][wid]
    w = entry['window']
    n = min(args.frames, int(w['frames']))
    if file_sha256(source) != w['sha256']:
        raise ValueError(f'{wid}: changed source')
    root = args.inputs / 'windows' / k4.safe(wid)
    for name, digest in entry['files'].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f'{wid}: changed K6 feature {name}')
    ball_info = next(r for r in json.loads(args.ball_report.read_text())['windows'] if r['id'] == wid)
    ball_path = args.balls / k4.safe(wid) / 'ball-masks.rle'
    if ball_info['source_sha256'] != w['sha256'] or file_sha256(ball_path) != ball_info['sha256']:
        raise ValueError(f'{wid}: ball reference identity mismatch')
    balls = ClipMasks.load(ball_path)
    clip = ClipMasks.load(root / 'sam-masks.rle')
    if len(balls.frames) < n:
        raise ValueError(f'{wid}: incomplete ball sequence')
    features = json.loads((root / 'features.json').read_text())
    fits = {int(t): np.asarray(h, np.float64) for t, h in entry['fits'].items()}
    players = k6.window_players(features, clip, int(w['start']), fits)['players']
    classes = np.lib.format.open_memmap(work / 'classes.npy', mode='w+', dtype=np.uint8, shape=(n, 1080, 1920))
    for t in range(n):
        regions = k6.frame_regions(clip.frames[t], features['aliases'], players)['regions']
        c = np.zeros((1080, 1920), np.uint8)
        for m in regions.values():
            c[m] = cp.HAND
        for inst in balls.frames[t]:
            c[k4.upsample(inst.mask(), k6.FRAME)] = cp.OBJECT
        classes[t] = c
    reduced = np.stack([mk.reduce(classes[t:t + 1], 8)[0] for t in range(n)])
    payload = mk.encode(reduced, 8)
    decoded, scale = mk.decode(payload)
    if scale != 8 or not np.array_equal(decoded, reduced):
        raise ValueError('mask roundtrip failed')
    shown = np.lib.format.open_memmap(work / 'shown.npy', mode='w+', dtype=np.uint8, shape=classes.shape)
    for t in range(n):
        shown[t] = mk.upsample(decoded[t:t + 1], k6.FRAME)[0]
    raw = work / 'original.yuv'
    conversion = convert(args.ffmpeg, {**w, 'frames': n}, source, raw)
    original = rgb_memmap(raw, work / 'original.npy', n, 1920, 1080, False, 'cuda')
    full = work / 'reference.yuv'
    # A common full-range 4:2:0 normalization is applied to every arm and to the reference.
    with full.open('wb') as target:
        for t in range(n):
            target.write(rgb_to_yuv420(original[t:t + 1]).tobytes())
    truth = rgb_memmap(full, work / 'truth.npy', n, 1920, 1080, True, 'cuda')
    filled = np.lib.format.open_memmap(work / 'filled.npy', mode='w+', dtype=np.uint8, shape=truth.shape)
    alpha = np.lib.format.open_memmap(work / 'alpha.npy', mode='w+', dtype=np.float32, shape=classes.shape)
    for t in range(n):
        filled[t] = cv2.inpaint(np.asarray(truth[t]), (shown[t] > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        alpha[t] = cp.alpha_and_classes(np.asarray(shown[t]))[0]
    judged = [t for t in range(n) if (int(w['start']) + t in fits) and (classes[t] > 0).any() and t % 3 == 0]
    if not judged:
        raise ValueError(f'{wid}: no judged frames')
    return {'id': wid, 'frames': n, 'fps': Fraction(w['fps']), 'truth': truth, 'classes': classes,
            'shown': shown, 'filled': filled, 'alpha': alpha, 'mask': payload, 'judged': judged,
            'source_sha256': w['sha256'], 'ball_sha256': ball_info['sha256'], 'conversion': conversion,
            'reference_yuv_sha256': file_sha256(full)}


def encode(args: argparse.Namespace, point: tuple[str, int, int], rgb: np.ndarray, w: dict[str, Any],
           work: Path, stream_dir: Path, name: str, foreground: bool = False,
           cpus: list[int] | None = None, defer_rgb: bool = False) -> dict[str, Any]:
    import cv2
    import torch
    from experiments.background import g5
    from experiments.visor import h5
    from src.codecs import svtav1

    codec, value, scale = point
    width, height = k6.scaled_size(scale)
    work.mkdir(parents=True, exist_ok=True)
    source = work / 'input.yuv'
    with source.open('wb') as target:
        for t in range(w['frames']):
            if foreground:
                small = cp.foreground_input(rgb[t:t + 1], w['shown'][t:t + 1] > 0, scale, (width, height))
            else:
                image = cv2.resize(np.asarray(rgb[t]), (width, height), interpolation=cv2.INTER_AREA) if scale != 1 else np.asarray(rgb[t])
                small = image[None]
            target.write(g5.rgb_to_yuv420(small).tobytes())
    if codec != 'svt':
        torch.cuda.empty_cache()
    start = time.monotonic()
    if codec == 'svt':
        cpus = cpus or sorted(os.sched_getaffinity(0))[:args.threads]
        rec = svtav1.code(source, work, width=width, height=height, fps=w['fps'], frames=w['frames'], crf=value,
                          preset=1, threads=len(cpus), full_range=True, encoder=args.encoder, decoder=args.decoder,
                          timeout=900, cpus=cpus)
        stream, count = bs.ivf_to_obus(Path(rec['stream']).read_bytes())
        if count != w['frames']:
            raise ValueError('SVT frame count mismatch')
        native_stream = Path(rec['stream'])
    else:
        coder = argparse.Namespace(image_ckpt=str(args.image_ckpt), video_ckpt=str(args.video_ckpt))
        rec = h5.code_dcvc(coder, source, w['frames'], value, (width, height), work / 'dcvc')
        native_stream = work / 'dcvc' / 'stream.psdc'
        stream = native_stream.read_bytes()
        from experiments.tennis.k5 import luma_psnr
        rec['native_luma_psnr'] = luma_psnr(Path(rec['decoded']), source, width, height, w['frames'])
        if rec['native_luma_psnr'] < 12:
            raise ValueError('DCVC decode failed the native luma sanity floor')
        for report_name in ['plan.json', 'encode.json', 'decode.json']:
            shutil.copyfile(work / 'dcvc' / report_name, stream_dir / (name + '-' + report_name))
    decoded = Path(rec['decoded'])
    if decoded.stat().st_size != w['frames'] * width * height * 3 // 2:
        raise ValueError('decoded size mismatch')
    shown_path = work / 'shown.npy'
    dest = stream_dir / (name + ('.ivf' if codec == 'svt' else '.psdc'))
    shutil.copyfile(native_stream, dest)
    result = {'name': name, 'codec': codec, 'value': value, 'scale': scale, 'size': [width, height],
              'payload_bytes': len(stream), 'stream_sha256': hashlib.sha256(stream).hexdigest(), 'native_stream': dest.name,
              'decoded': str(shown_path), 'stream': stream, 'encode_decode_seconds': time.monotonic() - start,
              'input_sha256': file_sha256(source), 'checks': {k: rec[k] for k in ('deterministic', 'decoder_matches_encoder_intra', 'attempts', 'native_luma_psnr', 'encode_command', 'decode_command', 'cpus') if k in rec}}
    result['_native_yuv'] = str(decoded)
    result['_input_yuv'] = str(source)
    if not defer_rgb:
        finish_rgb(result, w['frames'])
    return result


def finish_rgb(record: dict[str, Any], frames: int) -> None:
    """All color conversion stays on the GPU, after native work; CPU SVT threads never touch CUDA."""
    import cv2
    from experiments.visor.h3 import rgb_memmap

    width, height = record['size']
    shown_path = Path(record['decoded'])
    work = shown_path.parent
    decoded = Path(record.pop('_native_yuv'))
    source = Path(record.pop('_input_yuv'))
    native_rgb = rgb_memmap(decoded, work / 'native.npy', frames, width, height, True, 'cuda')
    shown = np.lib.format.open_memmap(shown_path, mode='w+', dtype=np.uint8, shape=(frames, 1080, 1920, 3))
    for t in range(frames):
        shown[t] = cv2.resize(np.asarray(native_rgb[t]), k6.FRAME, interpolation=cv2.INTER_CUBIC) if (width, height) != k6.FRAME else native_rgb[t]
    del native_rgb, shown
    for path in (source, decoded, work / 'native.npy'):
        path.unlink(missing_ok=True)


def score_outputs(nets: dict[str, Any], outputs: list[np.ndarray], truth: np.ndarray, classes: np.ndarray) -> list[dict[str, float]]:
    import torch
    from experiments.visor.h3b import lpips_reduce
    from experiments.visor.h5 import dists_regions
    from src.codecs.quality import psnr

    fg = classes > 0
    masks = [fg, ~fg, np.ones_like(fg)]
    lpips = lpips_reduce(nets['lpips'], outputs, truth, [masks for _ in outputs], 'cuda')
    ref = torch.from_numpy(np.ascontiguousarray(truth)).cuda().permute(2, 0, 1)[None].float() / 255
    torch_masks = [torch.from_numpy(m.copy()).cuda()[None, None] for m in masks]
    rows = []
    for image, lp in zip(outputs, lpips):
        x = torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2, 0, 1)[None].float() / 255
        ds = [float(v[0]) for v in dists_regions(nets['dists'], x, ref, torch_masks)]
        error = (image.astype(np.float32) - truth.astype(np.float32)) ** 2
        p = [psnr(float(error[m].mean())) for m in masks]
        rows.append({'wlpips': .7 * lp[0] + .3 * lp[1], 'wdists': .7 * ds[0] + .3 * ds[1],
                     'frame_lpips': lp[2], 'frame_dists': ds[2], 'wpsnr': .7 * p[0] + .3 * p[1]})
    return rows


def process(args: argparse.Namespace, wid: str, source: Path, scratch: Path) -> dict[str, Any]:
    import cv2
    import torch
    from src.codecs.quality import load_lpips
    from experiments.visor.h3b import load_dists

    start = time.monotonic()
    work = scratch / k4.safe(wid)
    work.mkdir(exist_ok=True)
    streams = scratch / 'publish' / k4.safe(wid)
    streams.mkdir(parents=True, exist_ok=True)
    w = reference(args, wid, source, work)
    foreground: list[dict[str, Any]] = []
    backgrounds: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    groups = [('foreground', FG, w['truth'], foreground), ('background', GRID, w['filled'], backgrounds), ('anchor', GRID, w['truth'], anchors)]
    cpus = sorted(os.sched_getaffinity(0))[:args.threads]
    svt_cpus = cpus[:max(1, len(cpus) - 8)]
    # One bounded CPU encoder overlaps the GPU coder, as K6 already does.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = []
        for kind, points, rgb, records in groups:
            for point in points:
                if point[0] == 'svt':
                    name = f'{kind}-{point[0]}-{point[1]}-s{point[2]}'
                    future = pool.submit(encode, args, point, rgb, w, work / name, streams, name, False, svt_cpus, True)
                    pending.append((records, future))
        for kind, points, rgb, records in groups:
            for point in points:
                if point[0] == 'svt':
                    continue
                name = f'{kind}-{point[0]}-{point[1]}-s{point[2]}'
                records.append(encode(args, point, rgb, w, work / name, streams, name, kind == 'foreground'))
                print(f'phase=codec window={wid} point={name} bytes={records[-1]["payload_bytes"]}', flush=True)
                publish_progress('codecs', sum(len(r) for r in (foreground, backgrounds, anchors)))
        for records, future in pending:
            record = future.result()
            finish_rgb(record, w['frames'])
            records.append(record)
            print(f'phase=codec window={wid} point={record["name"]} bytes={record["payload_bytes"]}', flush=True)
            publish_progress('codecs', sum(len(r) for r in (foreground, backgrounds, anchors)))
    for records in (backgrounds, anchors):
        records.sort(key=lambda r: GRID.index((r['codec'], r['value'], r['scale'])))
    write_json(stage_dir() / (k4.safe(wid) + '-native.json'), {
        'id': wid, 'frames': w['frames'], 'source_sha256': w['source_sha256'],
        'ball_sha256': w['ball_sha256'], 'mask_bytes': len(w['mask']),
        'streams': [{k: v for k, v in r.items() if k != 'stream'} for r in anchors + backgrounds + foreground]})
    nets = {'lpips': load_lpips(args.lpips_backbone, 'cuda'), 'dists': load_dists(args.dists_backbone, 'cuda')}
    rows = []
    seconds = w['frames'] / float(w['fps'])
    for anchor in anchors:
        rows.append({'kind': 'anchor', 'codec': anchor['codec'], 'name': anchor['name'],
                     'kbps': 8 * anchor['payload_bytes'] / seconds / 1000, 'samples': []})
    pairs = [(b, f) for b in backgrounds for f in foreground]
    for bg, fg in pairs:
        packed = container_bytes(bg['stream'], fg['stream'], w['mask'], bg['codec'], tuple(bg['size']), tuple(fg['size']), fg['scale'], w['fps'], w['frames'])
        rows.append({'kind': 'split', 'name': bg['name'] + '+' + fg['name'],
                     'kbps': relaxed_bytes(packed, fg['stream']) * 8 / seconds / 1000,
                     'actual_kbps': len(packed) * 8 / seconds / 1000,
                     'background_kbps': len(bg['stream']) * 8 / seconds / 1000,
                     'foreground_kbps': len(fg['stream']) * 8 / seconds / 1000,
                     'mask_kbps': len(w['mask']) * 8 / seconds / 1000, 'samples': []})
    decodes = {r['name']: np.load(r['decoded'], mmap_mode='r') for r in anchors + backgrounds + foreground}
    for j, t in enumerate(w['judged']):
        images = [np.asarray(decodes[a['name']][t]) for a in anchors]
        images += [cp.composite(decodes[f['name']][t:t + 1], decodes[b['name']][t:t + 1], w['alpha'][t:t + 1])[0] for b, f in pairs]
        scores = score_outputs(nets, images, np.asarray(w['truth'][t]), np.asarray(w['classes'][t]))
        for row, values in zip(rows, scores):
            row['samples'].append({'t': t, **values})
        if j in [0, len(w['judged']) // 2, len(w['judged']) - 1]:
            # One source, anchor and split review; selection fixed before quality is known.
            strip = np.concatenate([cv2.resize(im, (640, 360)) for im in [w['truth'][t], images[0], images[len(anchors)]]], axis=1)
            cv2.imwrite(str(stage_dir() / f'{k4.safe(wid)}-t{t:03d}.jpg'), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        if (j + 1) % 10 == 0:
            write_json(stage_dir() / (k4.safe(wid) + '-score-progress.json'), {'id': wid, 'complete_frames': j + 1, 'rows': rows})
            publish_progress('scores', j + 1)
            print(f'phase=score window={wid} frame={j + 1}/{len(w["judged"])}', flush=True)
    for row in rows:
        row['scores'] = {m: float(np.mean([s[m] for s in row['samples']])) for m in METRICS}
    write_json(streams / 'per-frame.json', rows)
    result = {'id': wid, 'frames': w['frames'], 'judged': w['judged'], 'seconds': seconds,
              'source_sha256': w['source_sha256'], 'ball_sha256': w['ball_sha256'], 'reference_yuv_sha256': w['reference_yuv_sha256'],
              'mask_bytes': len(w['mask']),
              'ball_pixels_retained': float(((w['classes'] == cp.OBJECT) & (w['shown'] > 0)).sum() / max(int((w['classes'] == cp.OBJECT).sum()), 1)),
              'conversion': w['conversion'], 'rows': [{k: v for k, v in r.items() if k != 'samples'} for r in rows],
              'streams': [{k: v for k, v in r.items() if k not in ('stream', 'decoded')} for r in anchors + backgrounds + foreground],
              'gpu_peak_gib': torch.cuda.max_memory_allocated() / 2**30, 'wall_seconds': time.monotonic() - start,
              'relaxation_check': all(r['kbps'] <= r['actual_kbps'] for r in rows if r['kind'] == 'split')}
    write_json(streams / 'window.json', result)
    del nets, decodes, w
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(work)
    return result


def validate(root: Path) -> dict[str, Any]:
    report = json.loads((root / 't1-oracle.json').read_text())
    checks = {'cuda': report['device'] == 'cuda', 'requested': report['requested'] == [w['id'] for w in report['windows']],
              'window_complete': all(len(w['rows']) == 24 and len(w['streams']) == 15 for w in report['windows']),
              'relaxation': all(w['relaxation_check'] for w in report['windows']),
              'finite': all(np.isfinite(r['kbps']) and r['kbps'] > 0 and all(np.isfinite(v) for v in r['scores'].values()) for w in report['windows'] for r in w['rows']),
              'score_frames': all(w['judged'] for w in report['windows'])}
    return {'passed': all(checks.values()), 'checks': checks}


def report(paths: list[Path], out: Path) -> int:
    from experiments.visor.h5 import compare

    windows = [w for p in paths for w in json.loads(p.read_text())['windows']]
    result: dict[str, Any] = {'windows': {}, 'summary': {}}
    for w in windows:
        comparisons: dict[str, Any] = {}
        for metric, sign in METRICS.items():
            test = [(r['kbps'], sign * r['scores'][metric]) for r in w['rows'] if r['kind'] == 'split']
            comparisons[metric] = {}
            for codec in ['pooled', 'svt', 'dcvc']:
                anchor = [(r['kbps'], sign * r['scores'][metric]) for r in w['rows'] if r['kind'] == 'anchor' and (codec == 'pooled' or r['codec'] == codec)]
                comparisons[metric][codec] = compare(test, anchor, 1e-6)
        result['windows'][w['id']] = comparisons
    for metric in METRICS:
        values = [v[metric]['pooled']['bd'] for v in result['windows'].values()]
        mean = float(np.mean(values)) if len(values) == 8 and all(v is not None for v in values) else None
        wins = sum(v[metric]['pooled']['bd'] is not None and v[metric]['pooled']['bd'] < 0 for v in result['windows'].values())
        result['summary'][metric] = {'mean_bd': mean, 'wins': wins, 'decision': 'unclear' if mean is None else 'pass' if mean <= -.1 and wins >= 6 else 'fail'}
    write_json(out, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for name in ['inputs', 'balls', 'ball-report', 'image-ckpt', 'video-ckpt', 'lpips-backbone', 'dists-backbone']:
        run.add_argument('--' + name, type=Path, required=True)
    run.add_argument('--windows', nargs='+', required=True)
    run.add_argument('--sources', type=Path, nargs='+', required=True)
    run.add_argument('--frames', type=int, default=240)
    run.add_argument('--max-windows', type=int, default=1)
    run.add_argument('--threads', type=int, default=32)
    run.add_argument('--ffmpeg', default='ffmpeg')
    run.add_argument('--encoder', default='SvtAv1EncApp')
    run.add_argument('--decoder', default='dav1d')
    sub.add_parser('validate')
    rep = sub.add_parser('report')
    rep.add_argument('--stage', nargs='+', type=Path, required=True)
    rep.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'report':
        return report(args.stage, args.out)
    if args.command == 'validate':
        result = validate(stage_dir())
        write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        return 0 if result['passed'] else 1
    import cv2
    import torch
    from src.codecs import svtav1
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if len(args.windows) != len(args.sources):
        parser.error('one source per window required')
    torch.cuda.set_per_process_memory_fraction(min(1., int(os.environ.get('PS_GPU_MEMORY_LIMIT_MIB', '40000')) * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    result = {'requested': args.windows[:args.max_windows], 'device': 'cuda', 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'tools': {t: svtav1.tool(t) for t in [args.encoder, args.decoder, args.ffmpeg]},
              'command': __import__('sys').argv, 'cpu_threads': args.threads, 'windows': []}
    for wid, source in zip(args.windows[:args.max_windows], args.sources):
        result['windows'].append(process(args, wid, source, scratch))
        write_json(stage_dir() / 't1-oracle.json', result)
        with tarfile.open(stage_dir() / 'published.tar', 'w') as archive:
            archive.add(scratch / 'publish', arcname='publish')
        publish_progress('windows', len(result['windows']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
