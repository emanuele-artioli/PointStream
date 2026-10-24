"""Regenerate T1's missing broadcast ball references with K4's SAM text prompt."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tennis import k4, windows as tw
from experiments.tennis.k4b_features import convert
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.segmentation.masks import ClipMasks


def validate(root: Path) -> dict[str, Any]:
    report = json.loads((root / 't1-balls.json').read_text())
    checks = {'cuda': report['device'] == 'cuda', 'requested': report['requested'] == [w['id'] for w in report['windows']],
              'positive_frames': all(w['frames'] > 0 for w in report['windows']),
              'some_ball_detections': any(any(c > 0 for c in w['counts']) for w in report['windows'])}
    with tarfile.open(root / 'balls.tar') as archive:
        for row in report['windows']:
            name = f"{k4.safe(row['id'])}/ball-masks.rle"
            member = archive.extractfile(name)
            assert member is not None
            import hashlib
            checks[row['id'] + ':identity'] = hashlib.sha256(member.read()).hexdigest() == row['sha256']
            checks[row['id'] + ':classes'] = row['classes'] == ['ball']
            checks[row['id'] + ':counts'] = len(row['counts']) == row['frames'] and all(n >= 0 for n in row['counts'])
    return {'passed': all(checks.values()), 'checks': checks}


def command_run(args: argparse.Namespace) -> int:
    import cv2
    import torch
    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get('PS_CPU_ALLOWANCE', '8')))
    manifest = json.loads((args.inputs / 'inputs.json').read_text())
    requested = args.windows[:args.max_windows]
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'balls'
    publish.mkdir(exist_ok=True)
    limit = int(os.environ.get('PS_GPU_MEMORY_LIMIT_MIB', '32000'))
    torch.cuda.set_per_process_memory_fraction(min(1., limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    sam = Sam31SequenceSegmenter(checkpoint_path=str(args.sam_checkpoint))
    report: dict[str, Any] = {'requested': requested, 'windows': [], 'device': 'cuda', 'prompt': {'ball': 'tennis ball'},
                              'runtime': runtime(), 'source_revision': os.environ.get('PS_GIT_HEAD'), 'command': __import__('sys').argv}
    began = time.monotonic()
    for wid, source in zip(args.windows, args.sources):
        if wid not in requested:
            continue
        entry = manifest['windows'][wid]
        w = entry['window']
        if file_sha256(source) != w['sha256']:
            raise ValueError(f'{wid}: changed original source')
        n = min(args.frames, int(w['frames']))
        work = scratch / k4.safe(wid)
        work.mkdir(exist_ok=True)
        yuv = work / 'source.yuv'
        converted = convert(args.ffmpeg, {**w, 'frames': n}, source, yuv)
        rgb = tw.rgb_frames(yuv, 1920, 1080, n, work / 'rgb.npy')
        jpeg = k4.write_jpegs(rgb, work / 'jpeg')
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        clip = sam.segment_frames(jpeg, {'ball': 'tennis ball'}, policy='offline_bidirectional', fps=float(w['fps']))
        if len(clip.frames) != n:
            raise ValueError(f'{wid}: incomplete ball sequence')
        out = publish / k4.safe(wid)
        out.mkdir(exist_ok=True)
        clip.save(out / 'ball-masks.rle')
        check = ClipMasks.load(out / 'ball-masks.rle')
        counts = [len(f) for f in check.frames]
        shares = [float(check.foreground(t).mean()) for t in range(n)]
        for t in sorted(set([0, n // 2, n - 1])):
            image = cv2.resize(np.asarray(rgb[t]), (960, 540), interpolation=cv2.INTER_AREA)
            overlay = image.copy()
            for inst in check.frames[t]:
                mask = inst.mask()
                overlay[mask] = (.3 * overlay[mask] + .7 * np.asarray([255, 40, 40])).astype(np.uint8)
                x0, y0, x1, y1 = [int(v) for v in inst.bbox]
                cv2.rectangle(overlay, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (255, 255, 0), 1)
            cv2.imwrite(str(stage_dir() / f'{k4.safe(wid)}-t{t:03d}.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        row = {'id': wid, 'frames': n, 'classes': list(clip.classes), 'counts': counts, 'share': shares,
               'source_sha256': w['sha256'], 'sha256': file_sha256(out / 'ball-masks.rle'),
               'sam_seconds': time.monotonic() - start, 'gpu_peak_gib': torch.cuda.max_memory_allocated() / 2**30,
               'conversion': converted}
        report['windows'].append(row)
        provenance = sam.provenance('offline_bidirectional')
        report['provenance'] = {k: getattr(provenance, k) for k in ('name', 'model_revision', 'checkpoint_sha256', 'config_sha256', 'policy')}
        report['seconds'] = time.monotonic() - began
        write_json(stage_dir() / 't1-balls.json', report)
        with tarfile.open(stage_dir() / 'balls.tar', 'w') as archive:
            for completed in report['windows']:
                archive.add(publish / k4.safe(completed['id']), arcname=k4.safe(completed['id']))
        publish_progress('ball-masks', len(report['windows']))
        print(f'window={wid} frames={n} ball_frames={sum(c > 0 for c in counts)} seconds={row["sam_seconds"]:.1f}', flush=True)
        del rgb
        shutil.rmtree(work)
        gc.collect()
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--inputs', type=Path, required=True)
    run.add_argument('--sam-checkpoint', type=Path, required=True)
    run.add_argument('--windows', nargs='+', required=True)
    run.add_argument('--sources', type=Path, nargs='+', required=True)
    run.add_argument('--frames', type=int, default=240)
    run.add_argument('--max-windows', type=int, default=8)
    run.add_argument('--ffmpeg', default='ffmpeg')
    sub.add_parser('validate')
    identify = sub.add_parser('identify')
    identify.add_argument('--source-stage', type=Path, required=True)
    identify.add_argument('--max-windows', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'identify':
        info = {n: {'path': str(args.source_stage / n), 'sha256': file_sha256(args.source_stage / n)} for n in ['balls.tar', 't1-balls.json']}
        source = json.loads((args.source_stage / 't1-balls.json').read_text())
        if len(source['windows']) != args.max_windows:
            raise ValueError('ball reference generation is incomplete')
        with tarfile.open(args.source_stage / 'balls.tar') as archive:
            for row in source['windows']:
                import hashlib
                handle = archive.extractfile(k4.safe(row['id']) + '/ball-masks.rle')
                assert handle is not None
                if hashlib.sha256(handle.read()).hexdigest() != row['sha256']:
                    raise ValueError('ball archive content identity mismatch')
        write_json(stage_dir() / 'ball-identities.json', info)
        write_json(Path(os.environ['PS_VALIDATION_PATH']), {'passed': True, 'checks': {'identified': len(info) == 2}})
        return 0
    if args.command == 'validate':
        result = validate(stage_dir())
        write_json(Path(os.environ['PS_VALIDATION_PATH']), result)
        return 0 if result['passed'] else 1
    if len(args.windows) != len(args.sources):
        parser.error('one source per window is required')
    return command_run(args)


if __name__ == '__main__':
    raise SystemExit(main())
