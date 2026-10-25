"""Bounded reversible guard-band lift for sparse black-key reference video."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_reference_video import (
    reference_banks, rgb_yuv, code_yuv, back_rgb, container, ref_diagnostics,
    recover_key, command, ffmpeg_args,
)
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from src.codecs import svtav1


def lift(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    values = np.round(32+rgb.astype(np.float32)*223/255).astype(np.uint8)
    return np.where(alpha[..., None], values, 0).astype(np.uint8)


def unlift(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip((rgb.astype(np.float32)-32)*255/223, 0, 255)).astype(np.uint8)


def validate(root: Path) -> dict[str, Any]:
    result = json.loads((root/'affine.json').read_text())
    checks = {'count': len(result['rows']) == (1 if result['mode'] == 'smoke' else 6),
              'cpu': result['cuda_visible'] == '', 'gpu': result['gpu_hours'] == 0,
              'inverse_precodec_error': result['precodec_max_rgb_error'] <= 1}
    for i, row in enumerate(result['rows']):
        rec = row['container']
        path = root/rec['file']
        checks[f'{i}:packet'] = path.stat().st_size == rec['bytes'] and sha(path.read_bytes()) == rec['sha256']
        decoded = root/f'validate-{i}.yuv'
        command([*ffmpeg_args(), '-i', str(path), '-map', '0:v:0', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', str(decoded)])
        checks[f'{i}:track'] = sha(decoded.read_bytes()) == rec['track_decoded_sha256'][0]
        decoded.unlink()
        with np.load(root/(path.stem+'-conditioning.npz')) as arrays:
            checks[f'{i}:inverse_pixels'] = array_sha(arrays['rgb']) == row['diagnostic_only']['decoded_rgb_sha256']
    return {'passed': all(checks.values()), 'checks': checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--transport', type=Path)
    p.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if a.validate:
        with tempfile.TemporaryDirectory(prefix='k6b-affine-', dir='/tmp') as tmp:
            with tarfile.open(stage/'published.tar') as archive:
                archive.extractall(tmp, filter='data')
            checked = validate(Path(tmp)/'publish')
        write(Path(os.environ['PS_VALIDATION_PATH']), checked)
        return 0 if checked['passed'] else 1
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only required')
    import cv2
    cv2.setNumThreads(1)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch/'publish'
    publish.mkdir()
    with tarfile.open(a.transport) as archive:
        archive.extractall(scratch/'transport', filter='data')
    banks = reference_banks(scratch/'transport/packets')
    toy = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, -1)
    error = int(np.max(np.abs(unlift(lift(toy, np.ones(toy.shape[:-1], bool))).astype(int)-toy.astype(int))))
    report: dict[str, Any] = {'rows': [], 'mode': a.mode, 'cuda_visible': '', 'gpu_hours': 0,
        'input_sha256': sha(a.transport.read_bytes()), 'code_sha256': sha(Path(__file__).read_bytes()),
        'tools': {tool: svtav1.tool(tool) for tool in ('ffmpeg', 'dav1d', 'SvtAv1EncApp')},
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'precodec_max_rgb_error': error, 'rule': 'foreground round(32+223*RGB/255), background0; mask max(channel)>=16; inverse clip(round((channel-32)*255/223))'}
    for match, entries in list(banks.items())[:1 if a.mode == 'smoke' else None]:
        rgb = np.stack([e['small'] for e in entries])
        alpha = np.stack([e['small_mask'] for e in entries])
        original = np.stack([e['rgb'] for e in entries])
        source_alpha = np.stack([e['mask'] for e in entries])
        raw_root = scratch/match
        raw_root.mkdir()
        raw, conversion = rgb_yuv(lift(rgb, alpha), raw_root)
        for crf in ((35,) if a.mode == 'smoke' else (20, 35, 50)):
            label = f'{match}-affine32-q{crf}'
            root = scratch/label
            root.mkdir()
            rec = code_yuv(raw, root/'codec', 12, 256, 392, crf)
            decoded, inverse = back_rgb(rec, 12, 256, 392, root)
            mask = recover_key(decoded, 32)
            restored = unlift(decoded)
            meta = {'units': [e['meta'] for e in entries], 'size': [256, 392], 'output_size': [512, 784],
                    'order': 'unit-ID-within-match-not-source-chronology', 'rgb_restore': 'unlift-then-bicubic',
                    'alpha_restore': 'bilinear>=0.5', 'kind': 'black-affine', 'floor': 32, 'threshold': 16,
                    'matrix': 'bt709', 'range': 'full', 'inverse': report['rule']}
            delivered = container([rec], publish/f'{label}.mkv', meta)
            diag = ref_diagnostics(original, source_alpha, restored, mask)
            report['rows'].append({'kind': 'black-affine', 'floor': 32, 'match': match, 'crf': crf,
                'codec': rec, 'conversion': conversion, 'inverse_conversion': inverse, 'container': delivered,
                'diagnostic_only': diag})
            np.savez_compressed(publish/f'{label}-conditioning.npz', rgb=np.stack([cv2.resize(x, (512, 784), interpolation=cv2.INTER_CUBIC) for x in restored]), alpha=mask)
            write(publish/'affine.json', report)
            from experiments.jobs.monitor import publish_progress
            publish_progress('affine', len(report['rows']))
    checked = validate(publish)
    write(publish/'validation.json', checked)
    if not checked['passed']:
        raise ValueError('Affine publication verification failed')
    write(stage/'affine.json', report)
    print(json.dumps({'rows': len(report['rows']), 'inverse_precodec_error': error, 'gpu_hours': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
