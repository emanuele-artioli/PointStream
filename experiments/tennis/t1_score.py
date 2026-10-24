"""Score frozen T1 native RGB decodes; never decode a stored codec stream."""
from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tennis.t1_oracle import METRICS, container_bytes, relaxed_bytes, score_outputs
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.pointstream import bitstream as bs, composite as cp, mask as mk


def score(args: argparse.Namespace) -> int:
    import torch
    from src.codecs.quality import load_lpips
    from experiments.visor.h3b import load_dists

    begin = time.monotonic()
    bundle = args.bundle
    m = json.loads((bundle / 'manifest.json').read_text())
    if len(m['arrays']) != 17:
        raise ValueError('full native decode archive is required')
    for name, identity in m['arrays'].items():
        if file_sha256(bundle / name) != identity['sha256']:
            raise ValueError(f'changed native RGB array: {name}')
    torch.set_num_threads(4)
    truth = np.load(bundle / 'truth.npy', mmap_mode='r')
    classes = np.load(bundle / 'classes.npy', mmap_mode='r')
    frames = m['frames']
    reduced = np.stack([mk.reduce(classes[t:t + 1], 8)[0] for t in range(frames)])
    mask = mk.encode(reduced, 8)
    records = m['streams']
    foreground = [r for r in records if r['kind'] == 'foreground']
    backgrounds = [r for r in records if r['kind'] == 'background']
    anchors = [r for r in records if r['kind'] == 'anchor']
    payloads = {}
    for r in records:
        native = (bundle / 'streams' / r['native_stream']).read_bytes()
        payload = bs.ivf_to_obus(native)[0] if r['codec'] == 'svt' else native
        if len(payload) != r['payload_bytes']:
            raise ValueError('native stream size changed')
        payloads[r['name']] = payload
    seconds = frames / float(Fraction(m['fps']))
    rows: list[dict[str, Any]] = []
    for a in anchors:
        rows.append({'kind': 'anchor', 'codec': a['codec'], 'name': a['name'],
                     'kbps': len(payloads[a['name']]) * 8 / seconds / 1000, 'samples': []})
    pairs = [(b, f) for b in backgrounds for f in foreground]
    for b, f in pairs:
        bg, fg = payloads[b['name']], payloads[f['name']]
        packed = container_bytes(bg, fg, mask, b['codec'], tuple(b['size']), tuple(f['size']), f['scale'], Fraction(m['fps']), frames)
        rows.append({'kind': 'split', 'name': b['name'] + '+' + f['name'],
                     'kbps': relaxed_bytes(packed, fg) * 8 / seconds / 1000,
                     'actual_kbps': len(packed) * 8 / seconds / 1000,
                     'background_kbps': len(bg) * 8 / seconds / 1000,
                     'foreground_kbps': len(fg) * 8 / seconds / 1000,
                     'mask_kbps': len(mask) * 8 / seconds / 1000, 'samples': []})
    arrays = {r['name']: np.load(bundle / (r['name'] + '.npy'), mmap_mode='r') for r in records}
    nets = {'lpips': load_lpips(args.lpips_backbone, 'cuda'), 'dists': load_dists(args.dists_backbone, 'cuda')}
    completed: list[int] = []
    if args.resume_json is not None:
        previous = json.loads(args.resume_json.read_text())
        if previous['id'] != m['id'] or len(previous['rows']) != len(rows):
            raise ValueError('wrong score checkpoint')
        completed = [s['t'] for s in previous['rows'][0]['samples']]
        if completed != m['judged'][:len(completed)]:
            raise ValueError('score checkpoint is not an exact frame prefix')
        for row, old in zip(rows, previous['rows']):
            if row['name'] != old['name'] or row['kbps'] != old['kbps'] or [s['t'] for s in old['samples']] != completed:
                raise ValueError('score checkpoint configuration mismatch')
            row['samples'] = old['samples']
    judged = [t for t in m['judged'] if t < args.max_frames and t not in completed]
    for j, t in enumerate(judged):
        shown = mk.upsample(reduced[t:t + 1], (1920, 1080))[0]
        alpha = cp.alpha_and_classes(shown)[0]
        images = [np.asarray(arrays[a['name']][t]) for a in anchors]
        images += [cp.composite(arrays[f['name']][t:t + 1], arrays[b['name']][t:t + 1], alpha[None])[0] for b, f in pairs]
        scores = score_outputs(nets, images, np.asarray(truth[t]), np.asarray(classes[t]))
        for row, values in zip(rows, scores):
            row['samples'].append({'t': t, **values})
        if (j + 1) % 10 == 0:
            write_json(stage_dir() / 'score-progress.json', {'id': m['id'], 'complete_frames': len(completed) + j + 1, 'rows': rows})
        publish_progress('saved-native-pixel-scores', j + 1)
        print(f'phase=score window={m["id"]} frame={j + 1}/{len(judged)}', flush=True)
    for row in rows:
        row['scores'] = {metric: float(np.mean([s[metric] for s in row['samples']])) for metric in METRICS}
    write_json(stage_dir() / 'per-frame.json', rows)
    window = {k: m[k] for k in ['id', 'frames', 'source_sha256', 'ball_sha256', 'reference_yuv_sha256', 'native_job', 'native_code', 'native_gpu']}
    window.update(seconds=seconds, judged=completed + judged, rows=[{k: v for k, v in r.items() if k != 'samples'} for r in rows],
                  wall_seconds=time.monotonic() - begin, streams=m['streams'],
                  relaxation_check=all(r['kbps'] < r['actual_kbps'] for r in rows if r['kind'] == 'split'))
    report = {'windows': [window], 'requested': [m['id']], 'device': 'cuda', 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'tools': m['tools'], 'recovery': True,
              'pixel_archive_manifest_sha256': file_sha256(bundle / 'manifest.json'),
              'resumed_score_sha256': file_sha256(args.resume_json) if args.resume_json is not None else None,
              'prior_metric_job': args.resume_job,
              'dists_cache_smoke_diff': nets.get('cache_verified')}
    write_json(stage_dir() / 't1-oracle.json', report)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'validate'])
    p.add_argument('--bundle', type=Path)
    p.add_argument('--lpips-backbone', type=Path)
    p.add_argument('--dists-backbone', type=Path)
    p.add_argument('--resume-json', type=Path)
    p.add_argument('--resume-job')
    p.add_argument('--max-frames', type=int, default=240)
    args = p.parse_args()
    if args.command == 'validate':
        from experiments.tennis.t1_oracle import validate
        result = validate(stage_dir())
        write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        return 0 if result['passed'] else 1
    return score(args)


if __name__ == '__main__':
    raise SystemExit(main())
