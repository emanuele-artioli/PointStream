"""Bounded native slower-preset comparison before new generated-quality scores."""
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

from experiments.tennis.k6b_reference_affine import lift, unlift
from experiments.tennis.k6b_reference_video import (
    reference_banks, rgb_yuv, code_yuv, back_rgb, container, ref_diagnostics,
    recover_key, command, ffmpeg_args,
)
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from src.codecs import svtav1


def validate(root: Path) -> dict[str, Any]:
    r = json.loads((root/'presets.json').read_text())
    expected = 2 if r['mode'] == 'smoke' else 12
    checks = {'count': len(r['rows']) == expected, 'cpu': r['cuda_visible'] == '',
              'gpu': r['gpu_hours'] == 0, 'settings': all(x['preset'] in (1, 4) and x['crf'] == 20 for x in r['rows'])}
    # Same complete 12-image bank in smoke and full. A measured projection gates full.
    if r['mode'] == 'smoke':
        checks['full_projection_fits_600s'] = r['wall_seconds']*9+30 < 600
    identities = set()
    for i, row in enumerate(r['rows']):
        identities.add((row['match'], row['kind'], row['preset']))
        rec = row['container']
        path = root/rec['file']
        checks[f'{i}:packet'] = path.stat().st_size == rec['bytes'] and sha(path.read_bytes()) == rec['sha256']
        checks[f'{i}:frames'] = row['codec']['decoded_frames'] == 12
        for j, expected_hash in enumerate(rec['track_decoded_sha256']):
            decoded = root/f'validate-{i}-{j}.yuv'
            command([*ffmpeg_args(), '-i', str(path), '-map', f'0:v:{j}', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', str(decoded)])
            checks[f'{i}:track{j}'] = sha(decoded.read_bytes()) == expected_hash
            decoded.unlink()
        with np.load(root/(path.stem+'-conditioning.npz')) as arrays:
            checks[f'{i}:pixels'] = array_sha(arrays['rgb']) == row['diagnostic_only']['decoded_rgb_sha256']
        checks[f'{i}:finite'] = all(np.isfinite(row['diagnostic_only'][k]) for k in ('foreground_rgb_mse', 'iou'))
    checks['unique'] = len(identities) == len(r['rows'])
    return {'passed': all(checks.values()), 'checks': checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--transport', type=Path)
    p.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if a.validate:
        with tempfile.TemporaryDirectory(prefix='k6b-presets-', dir='/tmp') as tmp:
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
    began = time.monotonic()
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch/'publish'
    publish.mkdir()
    with tarfile.open(a.transport) as archive:
        archive.extractall(scratch/'transport', filter='data')
    banks = reference_banks(scratch/'transport/packets')
    report: dict[str, Any] = {'rows': [], 'mode': a.mode, 'cuda_visible': '', 'gpu_hours': 0,
        'input_sha256': sha(a.transport.read_bytes()), 'code_sha256': sha(Path(__file__).read_bytes()),
        'tools': {tool: svtav1.tool(tool) for tool in ('ffmpeg', 'dav1d', 'SvtAv1EncApp')},
        'numpy': np.__version__, 'opencv': cv2.__version__}
    for match, entries in list(banks.items())[:1 if a.mode == 'smoke' else None]:
        rgb = np.stack([e['small'] for e in entries])
        alpha = np.stack([e['small_mask'] for e in entries])
        original = np.stack([e['rgb'] for e in entries])
        source_alpha = np.stack([e['mask'] for e in entries])
        for kind in (('full',) if a.mode == 'smoke' else ('full', 'alpha', 'black-affine')):
            source_rgb = rgb if kind == 'full' else lift(rgb, alpha) if kind == 'black-affine' else np.where(alpha[..., None], rgb, 0).astype(np.uint8)
            raw_root = scratch/f'{match}-{kind}'
            raw_root.mkdir()
            raw, conversion = rgb_yuv(source_rgb, raw_root)
            for preset in (4, 1):
                label = f'{match}-{kind}-p{preset}-q20'
                root = scratch/label
                root.mkdir()
                rec = code_yuv(raw, root/'codec', 12, 256, 392, 20, preset=preset, timeout=120)
                decoded, inverse = back_rgb(rec, 12, 256, 392, root)
                records = [rec]
                if kind == 'alpha':
                    alpha_raw = root/'alpha.yuv'
                    y = np.where(alpha, 255, 0).astype(np.uint8).reshape(12, -1)
                    np.concatenate([y, np.full((12, 256*392//2), 128, np.uint8)], 1).tofile(alpha_raw)
                    ar = code_yuv(alpha_raw, root/'alpha-codec', 12, 256, 392, 20, preset=preset, timeout=120)
                    mask = np.fromfile(ar['decoded'], np.uint8).reshape(12, -1)[:, :256*392].reshape(12, 392, 256) >= 128
                    records.append(ar)
                else:
                    mask = np.ones_like(alpha) if kind == 'full' else recover_key(decoded, 32)
                restored = unlift(decoded) if kind == 'black-affine' else decoded
                meta = {'units': [e['meta'] for e in entries], 'size': [256, 392], 'output_size': [512, 784],
                    'order': 'unit-ID-within-match-not-source-chronology', 'rgb_restore': 'unlift-then-bicubic' if kind == 'black-affine' else 'bicubic',
                    'alpha_restore': 'bilinear>=0.5', 'kind': kind, 'floor': 32 if kind == 'black-affine' else 0,
                    'threshold': 16 if kind == 'black-affine' else None, 'preset': preset, 'crf': 20,
                    'matrix': 'bt709', 'range': 'full', 'tracks': ['color', 'alpha'] if kind == 'alpha' else ['color']}
                delivered = container(records, publish/f'{label}.mkv', meta)
                diag = ref_diagnostics(original, source_alpha, restored, mask)
                report['rows'].append({'kind': kind, 'floor': meta['floor'], 'match': match, 'crf': 20, 'preset': preset,
                    'codec': rec, 'conversion': conversion, 'inverse_conversion': inverse, 'container': delivered,
                    'alpha_codec': records[1] if len(records) == 2 else None,
                    'source_rgb_sha256': array_sha(source_rgb), 'source_small_mask_sha256': array_sha(alpha),
                    'diagnostic_only': diag})
                np.savez_compressed(publish/f'{label}-conditioning.npz', rgb=np.stack([cv2.resize(x, (512, 784), interpolation=cv2.INTER_CUBIC) for x in restored]), alpha=mask)
                report['wall_seconds'] = time.monotonic()-began
                report['cpu_seconds'] = sum(r.ru_utime+r.ru_stime for r in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))
                write(publish/'presets.json', report)
                from experiments.jobs.monitor import publish_progress
                publish_progress('reference-presets', len(report['rows']))
    checked = validate(publish)
    write(publish/'validation.json', checked)
    if not checked['passed']:
        raise ValueError('Preset validation/projection failed')
    write(stage/'presets.json', report)
    print(json.dumps({'rows': len(report['rows']), 'wall_seconds': report['wall_seconds'], 'cpu_seconds': report['cpu_seconds'], 'gpu_hours': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
