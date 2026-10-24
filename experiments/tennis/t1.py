"""T1 CPU mask accounting on K6's adopted tennis windows."""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4, k6
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.pointstream import mask
from src.pointstream.composite import HAND, OBJECT
from src.segmentation.masks import ClipMasks


def measure(inputs: Path, wid: str, frames: int) -> dict[str, Any]:
    manifest = json.loads((inputs / 'inputs.json').read_text())
    entry = manifest['windows'][wid]
    w = entry['window']
    root = inputs / 'windows' / k4.safe(wid)
    for name, digest in entry['files'].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f'Changed input: {name}')
    features = json.loads((root / 'features.json').read_text())
    clip = ClipMasks.load(root / 'sam-masks.rle')
    fits = {int(t): np.asarray(h, np.float64) for t, h in entry['fits'].items()}
    players = k6.window_players(features, clip, int(w['start']), fits)['players']
    n = min(frames, int(w['frames']))
    reduced: dict[int, list[np.ndarray]] = {8: [], 16: []}
    shares: list[dict[str, Any]] = []
    for t, instances in enumerate(clip.frames[:n]):
        regions = k6.frame_regions(instances, features['aliases'], players)['regions']
        p = np.logical_or.reduce([*regions.values(), np.zeros((1080, 1920), bool)])
        b = np.zeros_like(p)
        for inst in instances:
            if inst.class_name == 'ball':
                b |= k4.upsample(inst.mask(), k6.FRAME)
        classes = np.zeros(p.shape, np.uint8)
        classes[p] = HAND
        classes[b] = OBJECT
        for scale in reduced:
            reduced[scale].append(mask.reduce(classes[None], scale)[0])
        shares.append({'t': t, 'registered': int(w['start']) + t in fits,
                       'player': float(p.mean()), 'ball': float(b.mean()), 'foreground': float((p | b).mean())})
    seconds = n / float(Fraction(w['fps']))
    rates = {}
    for scale, sequence in reduced.items():
        r = np.stack(sequence)
        payload = mask.encode(r, scale)
        decoded, s = mask.decode(payload)
        if s != scale or not np.array_equal(decoded, r):
            raise ValueError('MASK roundtrip failed')
        rates[str(scale)] = {'bytes': len(payload), 'kbps': 8 * len(payload) / seconds / 1000,
                             'reduced_foreground_share': float((r > 0).mean())}
    selected = [s for s in shares if s['registered']]
    return {'id': wid, 'frames': n, 'seconds': seconds, 'players': players,
            'scored_frames': len(selected), 'mask': rates,
            'share_all': {k: float(np.mean([s[k] for s in shares])) for k in ('player', 'ball', 'foreground')},
            'share_registered': {k: float(np.mean([s[k] for s in selected])) for k in ('player', 'ball', 'foreground')},
            'frame_shares': shares, 'input_files': entry['files'],
            'ball_provenance': 'SAM masks in K6 features; no native ball labels on broadcast windows'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['masks', 'validate'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--windows', nargs='+')
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--max-windows', type=int, default=8)
    args = parser.parse_args()
    out = stage_dir()
    if args.command == 'validate':
        report = json.loads((out / 't1-masks.json').read_text())
        checks = {'cpu_only': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                  'windows': bool(report['windows']),
                  'two_scales': all(set(w['mask']) == {'8', '16'} for w in report['windows']),
                  'finite_rates': all(np.isfinite(m['kbps']) and m['kbps'] > 0 for w in report['windows'] for m in w['mask'].values()),
                  'nonempty_players': all(w['players'] and w['scored_frames'] > 0 and 0 < w['share_all']['foreground'] < 1 for w in report['windows'])}
        write_json(Path(os.environ['PS_VALIDATION_PATH']), {'passed': all(checks.values()), 'checks': checks})
        return 0 if all(checks.values()) else 1
    rows = []
    for wid in args.windows[:args.max_windows]:
        rows.append(measure(args.inputs, wid, args.frames))
        write_json(out / 't1-masks.json', {'windows': rows, 'input_manifest_sha256': file_sha256(args.inputs / 'inputs.json')})
        from experiments.jobs.monitor import publish_progress
        publish_progress('masks', len(rows))
        print(f'window={wid} F={rows[-1]["share_all"]["foreground"]:.5f} mask8_kbps={rows[-1]["mask"]["8"]["kbps"]:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
