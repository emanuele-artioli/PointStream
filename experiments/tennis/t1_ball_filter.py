"""T1 ball references: retain SAM at BallTrack's source position, else K4's four-pixel disk."""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tennis import k4, windows as tw
from experiments.tennis.k4b_features import convert
from experiments.tennis.k5b_trackers import BallTrack, SIZE
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.segmentation.masks import ClipMasks, Instance


def run(args: argparse.Namespace) -> int:
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    tracker = BallTrack(args.balltrack_code, args.balltrack)
    if not str(next(tracker.net.parameters()).device).startswith('cuda'):
        raise ValueError('BallTrack did not load on CUDA')
    original = json.loads((args.inputs / 'inputs.json').read_text())['windows']
    raw = json.loads(args.sam_report.read_text())
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir(exist_ok=True)
    report: dict[str, Any] = {'requested': args.windows[:args.max_windows], 'windows': [], 'device': 'cuda',
                              'method': 'SAM reaching BallTrack within K4 BALL_REACH_PX=3, else radius-4 native disk; no detection means unknown/empty reference',
                              'balltrack_sha256': file_sha256(args.balltrack), 'balltrack_meta': tracker.meta}
    for wid, source in zip(args.windows[:args.max_windows], args.sources):
        w = original[wid]['window']
        n = min(args.frames, int(w['frames']))
        if file_sha256(source) != w['sha256']:
            raise ValueError('source identity mismatch')
        raw_info = next(r for r in raw['windows'] if r['id'] == wid)
        masks = args.sam_masks / k4.safe(wid) / 'ball-masks.rle'
        if raw_info['source_sha256'] != w['sha256'] or file_sha256(masks) != raw_info['sha256']:
            raise ValueError('SAM identity mismatch')
        sam = ClipMasks.load(masks)
        work = scratch / k4.safe(wid)
        work.mkdir(exist_ok=True)
        yuv = work / 'source.yuv'
        convert(args.ffmpeg, {**w, 'frames': n}, source, yuv)
        rgb = tw.rgb_frames(yuv, 1920, 1080, n, work / 'source.npy')
        small = np.stack([cv2.resize(np.asarray(f), SIZE, interpolation=cv2.INTER_AREA) for f in rgb])
        predictions = tracker.run(small)
        clip = ClipMasks(('ball',), 1080, 1920, float(w['fps']))
        retained, fallback, unknown = 0, 0, 0
        positions = []
        for t, (seen, x, y) in enumerate(predictions):
            x, y = x * 1.5, y * 1.5
            positions.append([bool(seen), float(x), float(y)])
            selected = np.zeros((1080, 1920), bool)
            if seen:
                near = tw.disk((1080, 1920), x, y, k4.BALL_REACH_PX)
                found = [k4.upsample(i.mask(), (1920, 1080)) for i in sam.frames[t]]
                reached = [m for m in found if (m & near).any()]
                if reached:
                    selected = reached[0]
                    retained += 1
                else:
                    selected = tw.disk((1080, 1920), x, y, k4.BALL_DISK_PX)
                    fallback += 1
            else:
                unknown += 1
            instance = Instance.from_mask('ball', 1, selected, provenance='sam_at_balltrack_or_disk')
            clip.frames.append([] if instance is None else [instance])
            if t in {0, n // 2, n - 1}:
                image = cv2.resize(np.asarray(rgb[t]), (960, 540), interpolation=cv2.INTER_AREA)
                if seen:
                    cv2.circle(image, (round(x / 2), round(y / 2)), 8, (255, 0, 0), 1)
                cv2.imwrite(str(stage_dir() / f'{k4.safe(wid)}-t{t:03d}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        dest = publish / k4.safe(wid)
        dest.mkdir(exist_ok=True)
        clip.save(dest / 'ball-masks.rle')
        row = {'id': wid, 'frames': n, 'source_sha256': w['sha256'], 'sha256': file_sha256(dest / 'ball-masks.rle'),
               'sam_sha256': raw_info['sha256'], 'sam_retained': retained, 'disk_fallback': fallback, 'unknown_frames': unknown,
               'positions': positions, 'counts': [len(f) for f in clip.frames], 'classes': ['ball']}
        report['windows'].append(row)
        with tarfile.open(stage_dir() / 'balls.tar', 'w') as archive:
            for finished in report['windows']:
                archive.add(publish / k4.safe(finished['id']), arcname=k4.safe(finished['id']))
        report['archive_sha256'] = file_sha256(stage_dir() / 'balls.tar')
        write_json(stage_dir() / 't1-balls.json', report)
        publish_progress('ball-reference-filter', len(report['windows']))
        print(f'window={wid} retained={retained} disk={fallback} unknown={unknown}', flush=True)
        del rgb, small
        import shutil
        shutil.rmtree(work)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run', 'validate'])
    for name in ['inputs', 'sam-masks', 'sam-report', 'balltrack', 'balltrack-code']:
        p.add_argument('--' + name, type=Path)
    p.add_argument('--windows', nargs='+')
    p.add_argument('--sources', nargs='+', type=Path)
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--max-windows', type=int, default=8)
    p.add_argument('--ffmpeg', default='ffmpeg')
    args = p.parse_args()
    if args.command == 'validate':
        r = json.loads((stage_dir() / 't1-balls.json').read_text())
        checks = {'cuda': r['device'] == 'cuda', 'requested': r['requested'] == [w['id'] for w in r['windows']],
                  'counts': all(w['sam_retained'] + w['disk_fallback'] + w['unknown_frames'] == w['frames'] for w in r['windows']),
                  'nonempty': any(w['sam_retained'] + w['disk_fallback'] > 0 for w in r['windows']),
                  'archive': file_sha256(stage_dir() / 'balls.tar') == r['archive_sha256']}
        result = {'passed': all(checks.values()), 'checks': checks}
        write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        return 0 if result['passed'] else 1
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
