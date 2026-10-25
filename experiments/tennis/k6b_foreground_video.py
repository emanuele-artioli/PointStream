"""CPU dense foreground-video carrier: explicit alpha versus RGB black key.

A separate representation from the sparse-reference generator. No neural quality
claim is made by its CPU MSE/IoU diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import tarfile
import tempfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_streams import decode_masks, decode_keypoints
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_reference_affine import lift, unlift
from experiments.tennis.k6b_reference_video import (
    CRFS, black_key, recover_key, overlap, command, ffmpeg_args,
    rgb_yuv, code_yuv, back_rgb, container,
)
from src.codecs import svtav1

PILOTS = ('alcaraz_hurkacz/031@129', 'medvedev_struff/014@194')
SOURCE_SHA = ('054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76',
              'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be')


def accuracy(truth: np.ndarray, target_mask: np.ndarray, rgb: np.ndarray,
             mask: np.ndarray) -> dict[str, Any]:
    import cv2
    totals = {k: 0 for k in ('intersection', 'union', 'false_positive', 'false_negative')}
    numerator = denominator = 0.
    rgb_hashes, mask_hashes, original_hashes = [], [], []
    for i, (image, alpha) in enumerate(zip(rgb, mask)):
        # native 960x540 SAM is an exact 2x repeated mask in the original study.
        source_alpha = np.repeat(np.repeat(target_mask[i], 2, 0), 2, 1)
        full_rgb = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        full_alpha = cv2.resize(alpha.astype(np.float32), (1920, 1080), interpolation=cv2.INTER_LINEAR) >= .5
        counts = overlap(source_alpha, full_alpha)
        for k in totals:
            totals[k] += counts[k]
        # MSE is defined only on foreground: avoid allocating three full-frame float arrays.
        diff = (full_rgb[source_alpha].astype(np.float32)-np.asarray(truth[i])[source_alpha].astype(np.float32))**2
        numerator += float(diff.sum())
        denominator += int(source_alpha.sum())*3
        rgb_hashes.append(array_sha(full_rgb))
        mask_hashes.append(array_sha(full_alpha))
        original_hashes.append(array_sha(np.asarray(truth[i])))
    return {**totals, 'iou': totals['intersection']/max(1, totals['union']),
            'foreground_rgb_mse': numerator/max(1., denominator),
            'decoded_rgb_frame_sha256': rgb_hashes, 'decoded_alpha_frame_sha256': mask_hashes,
            'original_rgb_frame_sha256': original_hashes}


def validate(root: Path) -> dict[str, Any]:
    result = json.loads((root/'foreground-video.json').read_text())
    n = result['frames']
    checks = {'cpu': result['cuda_visible'] == '' and result['gpu_hours'] == 0,
              'cohort': result['windows'] == list(PILOTS), 'row_count': len(result['rows']) == len(result['expected_points']),
              'settings': {(r['window'], r['crf'], r['kind'], r['floor']) for r in result['rows']} == {tuple(x) for x in result['expected_points']},
              'sources': [r['source_sha256'] for r in result['sources']] == list(SOURCE_SHA),
              'frames': n in (8, 240)}
    if n == 8 and result.get('preset', 8) == 8 and all(row['crf'] == 63 for row in result['rows']):
        # Historical full-window pilot provides the measured scale basis, not a smoke-only guess.
        projected = 818.7586409798823/7*4 + 150
        checks['historical_full_scale_projection_under_660s'] = projected < 660
        checks['real_smoke_under_55s'] = result['wall_seconds'] < 55
    for i, row in enumerate(result['rows']):
        rec = row['container']
        path = root/rec['file']
        checks[f'{i}:packet'] = len(path.read_bytes()) == rec['bytes'] and sha(path.read_bytes()) == rec['sha256']
        checks[f'{i}:coverage'] = row['codec']['decoded_frames'] == n and len(row['diagnostic_only']['original_rgb_frame_sha256']) == n
        checks[f'{i}:finite'] = all(np.isfinite(row['diagnostic_only'][k]) for k in ('iou', 'foreground_rgb_mse'))
        for j, expected in enumerate(rec['track_decoded_sha256']):
            decoded = root/f'check-{i}-{j}.yuv'
            command([*ffmpeg_args(), '-i', str(path), '-map', f'0:v:{j}', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', str(decoded)])
            checks[f'{i}:track{j}'] = sha(decoded.read_bytes()) == expected
            decoded.unlink()
    return {'passed': all(checks.values()), 'checks': checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--frames', type=int, choices=(8, 240), default=8)
    parser.add_argument('--crfs', default='20,35,50')
    parser.add_argument('--formats', choices=('all', 'robust', 'affine'), default='all')
    parser.add_argument('--preset', type=int, choices=(1, 4, 8), default=8)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        with tempfile.TemporaryDirectory(prefix='k6b-dense-validate-', dir='/tmp') as tmp:
            with tarfile.open(stage/'published.tar') as tar:
                tar.extractall(tmp, filter='data')
            validated = validate(Path(tmp)/'publish')
        write(Path(os.environ['PS_VALIDATION_PATH']), validated)
        return 0 if validated['passed'] else 1
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(args.source) != 2:
        raise ValueError('CPU-only, two frozen sources required')
    import cv2
    import torch
    from experiments.tennis.k4b_features import convert
    from experiments.visor.h3 import rgb_memmap
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    crfs = tuple(int(q) for q in args.crfs.split(','))
    if crfs not in (CRFS, (63,)):
        raise ValueError('Undeclared CRF refinement')
    formats = (('alpha', 0), ('black-affine', 32)) if args.formats == 'affine' else (('alpha', 0), ('black', 32)) if args.formats == 'robust' else (('alpha', 0), ('black', 1), ('black', 32))
    n, w, h = args.frames, 960, 540
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch/'publish'
    publish.mkdir()
    with tarfile.open(args.transport) as tar:
        tar.extractall(scratch/'packets', filter='data')
    result: dict[str, Any] = {'windows': list(PILOTS), 'rows': [], 'sources': [], 'frames': n,
        'preset': args.preset, 'expected_points': [[wid, q, kind, floor] for wid in PILOTS for q in crfs for kind, floor in formats],
        'gpu_hours': 0, 'cuda_visible': os.environ['CUDA_VISIBLE_DEVICES'],
        'code_sha256': sha(Path(__file__).read_bytes()), 'packet_archive_sha256': sha(args.transport.read_bytes()),
        'tools': {tool: svtav1.tool(tool) for tool in ('ffmpeg', 'dav1d', 'SvtAv1EncApp')},
        'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__}
    began = time.monotonic()
    def save() -> None:
        result['wall_seconds'] = time.monotonic()-began
        result['cpu_seconds'] = sum(r.ru_utime+r.ru_stime for r in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
        write(publish/'foreground-video.json', result)
        from experiments.jobs.monitor import publish_progress
        publish_progress('dense-foreground-video', len(result['rows']))
    for wid, source, expected in zip(PILOTS, args.source, SOURCE_SHA):
        if sha(source.read_bytes()) != expected:
            raise ValueError('Frozen source identity changed')
        label = wid.replace('/', '__')
        packet_root = scratch/'packets/packets'/label
        kmeta, _ = decode_keypoints((packet_root/'keypoints.kps').read_bytes())
        _, masks = decode_masks((packet_root/'exact-rle.matte').read_bytes())
        if masks.shape != (240, 2, h, w):
            raise ValueError('Source matte contract changed')
        alpha = masks[:n].any(1)
        del masks
        conversion = convert('ffmpeg', {'id': wid, 'frames': n, 'start': kmeta['clip_local_source_start']}, source, scratch/'source.yuv')
        truth = rgb_memmap(scratch/'source.yuv', scratch/f'{label}-truth.npy', n, 1920, 1080, False, 'cpu')
        (scratch/'source.yuv').unlink()
        original = np.stack([cv2.resize(np.asarray(im), (w, h), interpolation=cv2.INTER_AREA) for im in truth])
        result['sources'].append({'window': wid, 'source_sha256': expected, 'conversion': conversion,
                                 'source_alpha_sha256': array_sha(alpha), 'original_half_rgb_sha256': array_sha(original),
                                 'fps': kmeta['fps'], 'source_start': kmeta['clip_local_source_start']})
        alpha_raw = scratch/f'{label}-alpha.yuv'
        y = np.where(alpha, 255, 0).astype(np.uint8).reshape(n, -1)
        np.concatenate([y, np.full((n, h*w//2), 128, np.uint8)], 1).tofile(alpha_raw)
        source_preparation_seconds = time.monotonic()-began-sum(row.get('point_wall_seconds', 0) for row in result['rows'])
        for crf in crfs:
            alpha_rec = code_yuv(alpha_raw, scratch/f'{label}-alpha-q{crf}', n, w, h, crf, kmeta['fps'][0], preset=args.preset, timeout=240)
            back_alpha = np.fromfile(alpha_rec['decoded'], np.uint8).reshape(n, -1)[:, :w*h].reshape(n, h, w) >= 128
            for kind, floor in formats:
                point_began = time.monotonic()
                tag = f'{label}-{kind}-{floor}-q{crf}'
                root = scratch/tag
                root.mkdir()
                rgb = lift(original, alpha) if kind == 'black-affine' else black_key(original, alpha, floor) if kind == 'black' else np.where(alpha[..., None], original, 0).astype(np.uint8)
                raw, forward = rgb_yuv(rgb, root)
                coded = code_yuv(raw, root/'codec', n, w, h, crf, kmeta['fps'][0], preset=args.preset, timeout=240)
                restored, inverse = back_rgb(coded, n, w, h, root)
                decoded_alpha = recover_key(restored, floor) if kind in ('black', 'black-affine') else back_alpha
                if kind == 'black-affine':
                    restored = unlift(restored)
                contract = {'window': wid, 'frames': n, 'size': [w, h], 'output_size': [1920, 1080],
                            'source_start': kmeta['clip_local_source_start'], 'fps': kmeta['fps'],
                            'kind': kind, 'floor': floor, 'preset': args.preset, 'key_threshold': (floor+1)//2 if floor else None,
                            'rgb_restore': 'bicubic', 'alpha_restore': 'bilinear>=0.5',
                            'matrix': 'bt709', 'range': 'full', 'alpha': 'saved-SAM-union-both-players',
                            'tracks': ['color', 'alpha'] if kind == 'alpha' else ['color'],
                            'pixels': 'dense-original-foreground-no-generator'}
                delivered = container([coded, alpha_rec] if kind == 'alpha' else [coded], publish/f'{tag}.mkv', contract)
                diag = accuracy(truth, alpha, restored, decoded_alpha)
                result['rows'].append({'window': wid, 'kind': kind, 'floor': floor, 'crf': crf,
                    'codec': coded, 'alpha_codec': alpha_rec if kind == 'alpha' else None, 'container': delivered,
                    'diagnostic_only': diag, 'forward': forward, 'inverse': inverse,
                    'kbps_whole_window': delivered['bytes']*8/(n/kmeta['fps'][0])/1000,
                    'point_wall_seconds': time.monotonic()-point_began,
                    'source_preparation_seconds_cumulative': source_preparation_seconds,
                    'saved_kps_for_time_only_not_delivered': True})
                raw.unlink()
                # Release intermediates only; delivered containers and source/decoded identities persist.
                Path(coded['decoded']).unlink()
                (root/'back.rgb').unlink()
                save()
            Path(alpha_rec['decoded']).unlink()
        alpha_raw.unlink()
        del truth, original, rgb, restored
        (scratch/f'{label}-truth.npy').unlink()
    save()
    checked = validate(publish)
    write(publish/'validation.json', checked)
    if not checked['passed']:
        raise ValueError('Dense delivered-codec validation failed')
    write(stage/'foreground-video.json', result)
    print(json.dumps({'rows': len(result['rows']), 'frames': n, 'cpu_seconds': result['cpu_seconds'], 'wall_seconds': result['wall_seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
