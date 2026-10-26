"""Native fixed-region judging of delivered foreground video or changed references."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k4 import safe
from experiments.tennis.k6b_foreground_video import PILOTS, SOURCE_SHA
from experiments.tennis.k6b_matte_cpu import box_maps
from experiments.tennis.k6b_matte_judge import control_error, extract, gpu_memory_limit
from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_reference_video import command, ffmpeg_args


def unpack_carriers(archives: list[Path], scratch: Path, crfs: list[int], windows: list[str], settings: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {w: [] for w in PILOTS}
    for i, archive in enumerate(archives):
        root = scratch/str(i)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
        report = json.loads((root/'publish/foreground-video.json').read_text())
        if report['frames'] != 240:
            raise ValueError('Only complete240-frame delivered carriers can be judged')
        for row in report['rows']:
            if row['window'] not in windows or row['crf'] not in crfs:
                continue
            rec = row['container']
            name = f"{row['kind']}-floor{row['floor']}-p{rec['contract'].get('preset', 8)}-q{row['crf']}"
            if settings and name not in settings:
                continue
            path = root/'publish'/rec['file']
            if sha(path.read_bytes()) != rec['sha256'] or path.stat().st_size != rec['bytes']:
                raise ValueError('Delivered dense packet changed')
            # Independently decode each delivered track and compare native CPU receipts.
            decoded: list[Any] = []
            for j, expected in enumerate(rec['track_decoded_sha256']):
                raw = root/f'{path.stem}-{j}.yuv'
                command([*ffmpeg_args(), '-i', str(path), '-map', f'0:v:{j}', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', str(raw)])
                if sha(raw.read_bytes()) != expected:
                    raise ValueError('Delivered track differs from saved native dav1d decode')
                if j == 0:
                    rgb_path = root/f'{path.stem}.rgb'
                    command([*ffmpeg_args(), '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', '960x540',
                        '-i', str(raw), '-frames:v', '240', '-vf', 'scale=in_range=full:out_range=full:in_color_matrix=bt709',
                        '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(rgb_path)])
                    decoded.append(np.memmap(rgb_path, dtype=np.uint8, mode='r', shape=(240, 540, 960, 3)))
                else:
                    yuv = np.memmap(raw, dtype=np.uint8, mode='r', shape=(240, 540*960*3//2))
                    decoded.append(yuv[:, :540*960].reshape(240, 540, 960) >= 128)
            rgb = decoded[0]
            mask = decoded[1] if row['kind'] == 'alpha' else rgb.max(-1) >= (row['floor']+1)//2
            out[row['window']].append({**row, 'rgb': rgb, 'mask': mask,
                'name': name, 'archive_sha256': sha(archive.read_bytes())})
    for rows in out.values():
        if len({r['name'] for r in rows}) != len(rows):
            raise ValueError('Duplicate carrier setting')
    return out


def judge_window(wid: str, anchor: Path, old_control: Path, packets: Path,
                 generated: dict[str, dict[Any, Any]], carriers: list[dict[str, Any]],
                 source: Path, nets: dict[str, Any], scratch: Path, publish: Path,
                 max_judged: int) -> dict[str, Any]:
    import cv2
    from experiments.tennis import k6, k6b
    from experiments.tennis.k4b_features import convert
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import lpips_reduce, dists_crops
    from src.segmentation.masks import decompress
    from experiments.tennis.k6b_reference_affine import unlift
    from src.codecs.quality import read_frames, yuv420_to_rgb
    import torch

    began = time.monotonic()
    native_window = k6b.k6_window(anchor)
    native = native_window['row']
    requested = native['judged'][:max_judged or None]
    kmeta, units = decode_keypoints((packets/safe(wid)/'keypoints.kps').read_bytes())
    _, source_masks = decode_masks((packets/safe(wid)/'exact-rle.matte').read_bytes())
    if kmeta['window'] != wid or source_masks.shape != (240, len(native['players']), 540, 960):
        raise ValueError('Frozen frame/player identity differs')
    boxes = box_maps(units)
    crops: dict[str, dict[int, dict[int, np.ndarray]]] = {}
    for label, generated_values in generated.items():
        crops[label] = {}
        for i, unit in enumerate(units):
            g = generated_values[(unit['name'], 'match5')]
            expected = sorted(set(unit['t'].tolist()) & set(native['judged']))
            if g['t'].tolist() != expected or g['crops'].shape != (len(expected), 784, 512, 3):
                raise ValueError('Full-window generated crop coverage changed')
            crops[label][i] = dict(zip(g['t'].tolist(), g['crops']))
    names = list(crops) + [c['name'] for c in carriers]
    if not names or names[0] != 'old-control' or len(names) != len(set(names)):
        raise ValueError('Original control must be first; no duplicate variants')
    work = scratch/safe(wid)
    work.mkdir()
    convert_record = convert('ffmpeg', {'id': wid, 'start': native['start'], 'frames': 240}, source, work/'source.yuv')
    truth = rgb_memmap(work/'source.yuv', work/'truth.npy', 240, 1920, 1080, False, 'cuda')
    regions = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        for i, p in enumerate(native['players']) if source_masks[t, i].any()} for t in range(240)]
    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0])+1) if p in r and r[p].any() else 1.
        for t, r in enumerate(regions) for p in native['players']}
    decoded_all = {key: {(n.rsplit('@', 1)[0], int(n.rsplit('@', 1)[1])): np.asarray(v, np.float64)
        for n, v in d.items()} for key, d in json.loads(decompress((anchor/'keypoints.json.zst').read_bytes()))['decoded'].items()}
    old_row = json.loads((old_control/'window.json').read_text())
    old_index = old_row['variants'].index('gen-match5')
    with np.load(old_control/'scores.npz') as a:
        expected_control = np.stack([a[m][old_index] for m in ('lpips', 'dists')])
    dest = publish/safe(wid)
    dest.mkdir()
    arrays = {m: np.full((len(names), 240), np.nan) for m in ('lpips', 'dists', 'full_sse', 'outside_r_sse', 's_sse', 'mask_fp', 'mask_fn')}
    row: dict[str, Any] = {'id': wid, 'variants': names, 'judged': [], 'expected_judged': requested,
        'original_judged': native['judged'], 'scoring_masks': {}, 'source_sha256': sha(source.read_bytes()),
        'dists_effective_batch': min(4 if len(names) > 5 else 6, len(names)), 'source_conversion': convert_record, 'source_cpu_vs_native_gpu': {}, 'composite_sha256': {}, 'carrier_receipts': [
            {k: c[k] for k in ('name', 'container', 'kbps_whole_window', 'archive_sha256')} for c in carriers]}
    for n, t in enumerate(requested):
        truth_t = np.array(truth[t])
        cpu_source = None
        if carriers:
            planes = torch.from_numpy(read_frames(work/'source.yuv', 1920, 1080, t, 1))
            cpu_source = yuv420_to_rgb(planes, 1920, 1080, full_range=False)[0].permute(1, 2, 0).to(torch.uint8).numpy()
            delta = np.abs(cpu_source.astype(np.int16)-truth_t.astype(np.int16))
            row['source_cpu_vs_native_gpu'][str(t)] = {'cpu_sha256': array_sha(cpu_source), 'native_gpu_sha256': array_sha(truth_t), 'max_abs_difference': int(delta.max()), 'different_channels': int(np.count_nonzero(delta))}
            if delta.max() > 1:
                raise ValueError('CPU/carrier and nativeGPU source conversion differ beyond1level')
        r = np.logical_or.reduce([*regions[t].values(), np.zeros((1080, 1920), bool)])
        fill = k6.telea_fill(truth_t, r)
        _, drawn = k6b.k6_oracle(t, native_window, k6.variant_names()[-9:], truth, regions, fill,
            kmeta['compositing_order'], native['scored'], heights, decoded_all,
            max(1, round(k6.REF_GAP_S*float(Fraction(native['fps'])))))
        s = r.copy()
        for drawn_mask in drawn.values():
            s |= drawn_mask
        hashes = {'r': sha(np.packbits(r).tobytes()), 's': sha(np.packbits(s).tobytes())}
        if hashes != native['scoring_masks'][str(t)] or hashes != old_row['scoring_masks'][str(t)]:
            raise ValueError('Original R/S changed')
        row['scoring_masks'][str(t)] = hashes
        frames, supports = [], []
        for label in crops:
            canvas, written = fill.copy(), np.zeros_like(r)
            for player in kmeta['compositing_order']:
                i = native['players'].index(player)
                if t in crops[label][i]:
                    written |= k6b.paste(canvas, crops[label][i][t], boxes[i][t], regions[t].get(player, np.zeros_like(r)))
            frames.append(canvas)
            supports.append(written)
        for c in carriers:
            rgb = np.asarray(c['rgb'][t])
            alpha = cv2.resize(c['mask'][t].astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR) >= .5
            if c['kind'] == 'black-affine':
                rgb = unlift(rgb)
            image = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            canvas = fill.copy()
            canvas[alpha] = image[alpha]
            frames.append(canvas)
            supports.append(alpha)
            diag = c['diagnostic_only']
            if cpu_source is None:
                raise ValueError('CPU source pixel identity unavailable')
            pixel_checks = {'source_cpu': array_sha(cpu_source) == diag['original_rgb_frame_sha256'][t],
                'decoded_rgb': array_sha(image) == diag['decoded_rgb_frame_sha256'][t],
                'decoded_alpha': array_sha(alpha) == diag['decoded_alpha_frame_sha256'][t]}
            if not all(pixel_checks.values()):
                write(Path(os.environ['PS_STAGE_DIR'])/'pixel-mismatch.json', {'window': wid, 't': t, 'variant': c['name'], 'checks': pixel_checks, 'source_roundoff': row['source_cpu_vs_native_gpu'][str(t)]})
                raise ValueError('Actual dense decoded pixels differ from CPU evidence: '+json.dumps(pixel_checks))
        lp = lpips_reduce(nets['lpips'], frames, truth_t, [[s, support] for support in supports], 'cuda')
        ys, xs = np.nonzero(s)
        crop = (slice(max(0, ys.min()-k6.DISTS_PAD), min(1080, ys.max()+1+k6.DISTS_PAD)),
                slice(max(0, xs.min()-k6.DISTS_PAD), min(1920, xs.max()+1+k6.DISTS_PAD)))
        # DISTS' default batch6 exceeds12GB for the eight-variant historical curve.
        # Four is bounded by the full-window five-variant memory evidence.
        if len(frames) > 5:
            di = [value for first in range(0, len(frames), 4) for value in
                dists_crops(nets['dists'], [image[crop] for image in frames[first:first+4]], truth_t[crop], s[crop], 'cuda')]
        else:
            di = dists_crops(nets['dists'], [image[crop] for image in frames], truth_t[crop], s[crop], 'cuda')
        values = np.array([[v[0] for v in lp], di])
        control_error(values[:, 0], expected_control[:, t])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite actual neural score')
        arrays['lpips'][:, t], arrays['dists'][:, t] = values
        for i, (image, support) in enumerate(zip(frames, supports)):
            error = (image.astype(np.float64)-truth_t.astype(np.float64))**2
            arrays['full_sse'][i, t], arrays['outside_r_sse'][i, t], arrays['s_sse'][i, t] = error.sum(), error[~r].sum(), error[s].sum()
            arrays['mask_fp'][i, t], arrays['mask_fn'][i, t] = (support & ~r).sum(), (~support & r).sum()
        row['composite_sha256'][str(t)] = [array_sha(image) for image in frames]
        row['judged'].append(t)
        if n % 10 == 0 or n+1 == len(requested):
            np.savez_compressed(dest/'scores.npz', **arrays, judged=np.array(row['judged']))
            write(dest/'partial.json', {'id': wid, 'judged': row['judged'], 'expected': requested})
            from experiments.jobs.monitor import publish_progress
            publish_progress('carrier-frame', n+1)
    row['control_max_abs_difference'] = control_error(np.stack([arrays[m][0, requested] for m in ('lpips', 'dists')]), expected_control[:, requested])
    row['means'] = {name: {m: float(a[i, requested].mean()) for m, a in arrays.items()} for i, name in enumerate(names)}
    if 'png-control' in names:
        i = names.index('png-control')
        row['new_png_control_max_score_difference'] = control_error(np.stack([arrays[m][i, requested] for m in ('lpips', 'dists')]), expected_control[:, requested], tolerance=1e-3)
        row['new_png_control_bitexact_composites'] = all(v[0] == v[i] for v in row['composite_sha256'].values())
    row['seconds'] = time.monotonic()-began
    row['files'] = {q.name: sha(q.read_bytes()) for q in dest.iterdir() if q.is_file()}
    write(dest/'window.json', row)
    del truth
    shutil.rmtree(work)
    return row


def validate(archive: Path, full: bool, *, allow_completed_prefix: bool = False) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='carrier-check-', dir='/tmp') as tmp:
        root = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
        r = json.loads((root/'publish/carrier-judge.json').read_text())
        completed = [w['id'] for w in r['rows']]
        collection_complete = completed == r['requested']
        collection_valid = collection_complete or (allow_completed_prefix and bool(completed) and completed == r['requested'][:len(completed)])
        checks = {'cuda': all(r['on_cuda'].values()), 'class': r['device']['class_matches'],
            'memory': r['gpu_peak_gib']*1024 <= r['memory_limit_mib'], 'windows': collection_valid}
        for row in r['rows']:
            label = row['id']
            checks[label+':variants'] = row['variants'] == r['expected_variants'][label]
            checks[label+':source'] = row['source_sha256'] == SOURCE_SHA[PILOTS.index(label)]
            checks[label+':coverage'] = row['judged'] == row['expected_judged'] and (not full or row['judged'] == row['original_judged'])
            if row.get('carrier_receipts'):
                checks[label+':source_roundoff'] = set(row['source_cpu_vs_native_gpu']) == {str(t) for t in row['judged']} and all(v['max_abs_difference'] <= 1 for v in row['source_cpu_vs_native_gpu'].values())
            checks[label+':control'] = row['control_max_abs_difference'] <= 1e-4 and row.get('new_png_control_max_score_difference', 0) <= (1e-3 if r.get('regenerated_png_control_tolerance') == 1e-3 else 1e-4)
            dest = root/'publish'/safe(label)
            checks[label+':files'] = all(sha((dest/name).read_bytes()) == digest for name, digest in row['files'].items())
            with np.load(dest/'scores.npz') as arrays:
                checks[label+':scores'] = all(np.isfinite(arrays[m][:, row['judged']]).all() and
                    np.allclose(arrays[m][:, row['judged']].mean(1), [row['means'][v][m] for v in row['variants']], rtol=0, atol=1e-10)
                    for m in row['means'][row['variants'][0]])
        return {'passed': all(checks.values()), 'checks': checks, 'complete_requested_collection': collection_complete}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor', 'control', 'source', 'backbone', 'carrier'):
        p.add_argument('--'+name, type=Path, action='append', default=[])
    p.add_argument('--generated', nargs=2, action='append', default=[])
    p.add_argument('--transport', type=Path)
    p.add_argument('--windows', nargs='+', default=list(PILOTS))
    p.add_argument('--crfs', default='63')
    p.add_argument('--carrier-setting', action='append', default=[])
    p.add_argument('--max-judged', type=int, default=0)
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if a.validate:
        checked = validate(stage/'published.tar', os.environ.get('PS_STAGE') == 'full')
        write(Path(os.environ['PS_VALIDATION_PATH']), checked)
        return 0 if checked['passed'] else 1
    import cv2
    import torch
    from experiments.audit.env_smoke import device_record
    from experiments.tennis.k6b import load_generated
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record
    torch.set_num_threads(8)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    device = device_record('RTX 6000 Ada')
    if not device['class_matches']:
        raise ValueError('Compatible Ada GPU required')
    limit = gpu_memory_limit()
    torch.cuda.set_per_process_memory_fraction(min(1., limit*2**20/torch.cuda.get_device_properties(0).total_memory))
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch/'publish'
    publish.mkdir()
    anchors, controls = extract(a.anchor, scratch/'anchors'), extract(a.control, scratch/'controls')
    with tarfile.open(a.transport) as tar:
        tar.extractall(scratch/'transport', filter='data')
    labels: dict[str, list[str]] = {}
    for label, path in a.generated:
        labels.setdefault(label, []).append(path)
    if set(labels) & {'full', 'alpha', 'black-affine'} and 'png-control' not in labels:
        raise ValueError('Actual reference comparison requires the newly rendered PNG control')
    generated = {label: load_generated(paths, scratch/label) for label, paths in labels.items()}
    carriers = unpack_carriers(a.carrier, scratch/'carriers', [int(q) for q in a.crfs.split(',')], a.windows, a.carrier_setting)
    if a.carrier_setting and any(set(c['name'] for c in carriers[wid]) != set(a.carrier_setting) for wid in a.windows):
        raise ValueError('Declared delivered carrier setting missing')
    nets = {'lpips': load_lpips(a.backbone[0], 'cuda'), 'dists': load_dists(a.backbone[1], 'cuda')}
    sources = {sha(source.read_bytes()): source for source in a.source}
    result: dict[str, Any] = {'rows': [], 'requested': a.windows, 'device': device, 'memory_limit_mib': limit,
        'on_cuda': {name: next(net.parameters()).is_cuda for name, net in nets.items()},
        'saved_rgb_control_tolerance': 1e-4, 'regenerated_png_control_tolerance': 1e-3,
        'expected_variants': {wid: list(labels)+[c['name'] for c in carriers[wid]] for wid in a.windows},
        'inputs': {str(q): sha(q.read_bytes()) for q in [a.transport, *a.anchor, *a.control, *a.source, *a.carrier, *[Path(v) for _, v in a.generated]]},
        'code_sha256': sha(Path(__file__).read_bytes()), 'torch': torch.__version__,
        'lpips': lpips_record(a.backbone[0]), 'dists_backbone_sha256': sha(a.backbone[1].read_bytes())}
    if not all(result['on_cuda'].values()):
        raise ValueError('Metric CPU fallback forbidden')
    began = time.monotonic()
    for wid in a.windows:
        if wid not in PILOTS:
            raise ValueError('Frozen two-window scope only')
        source_sha = SOURCE_SHA[PILOTS.index(wid)]
        result['rows'].append(judge_window(wid, anchors[wid], controls[wid], scratch/'transport/packets', generated,
            carriers[wid], sources[source_sha], nets, scratch, publish, a.max_judged))
        result['gpu_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        result['seconds'] = time.monotonic()-began
        write(publish/'carrier-judge.json', result)
        write(stage/'carrier-judge.json', result)
        with tarfile.open(stage/'published.tar', 'w') as tar:
            tar.add(publish, arcname='publish')
    print(json.dumps({'windows': len(result['rows']), 'seconds': result['seconds'], 'peak_gib': result['gpu_peak_gib']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
