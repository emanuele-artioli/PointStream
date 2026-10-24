"""H5p 4d: SAM tracking on saved decodes, with billed seed and exact corrections.

The quality equals H5p 4a only after correction to its exact reduced binary
mask. Uncorrected accuracy is descriptive, never reused as a quality score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.jobs.monitor import publish_progress
from src.pointstream import mask as mk, shape
from src.segmentation.masks import ClipMasks
from src.segmentation.visor import HANDS


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--masks')
    p.add_argument('--foreground', action='append', default=[])
    p.add_argument('--item')
    p.add_argument('--checkpoint')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--points', type=int, default=0)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        r = json.loads((stage / 'decoder-mask.json').read_text())
        checks = {'on_cuda': r['runtime']['visible_devices'] not in ('', None),
                  'device_class': r['runtime']['capability'][0] >= 8,
                  'loaded_all_weights': not r['model']['missing_keys'] and not r['model']['unexpected_keys'],
                  'points': len(r['rows']) == r['expected_points'] and len(r['rows']) > 0,
                  'exact_correction': all(x['exact'] for x in r['rows']),
                  'finite_rate': all(np.isfinite(x['kbps']) and x['kbps'] > 0 for x in r['rows'])}
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}))
        return int(not all(checks.values()))
    import av
    import cv2
    import torch
    from src.segmentation.sam31_tracker import Sam31MaskTracker, runtime
    torch.set_num_threads(4)
    clip = ClipMasks.load(Path(args.masks) / 'publish/masks' / args.item)
    n = min(args.frames, len(clip))
    classes = []
    for t in range(n):
        c = np.where(clip.foreground(t), 2, 0).astype(np.uint8)
        c[clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1])] = 1
        classes.append(mk.reduce(c[None], 8)[0])
    truth = (np.stack(classes) > 0).astype(np.uint8)
    seed_bytes = shape.encode(truth[:1], 0)
    seed = shape.decode(seed_bytes)[0]
    # Receiver obtains the very same full-size initial union from these bytes.
    initial = mk.upsample(seed[None], (clip.width, clip.height))[0] > 0
    tracker = Sam31MaskTracker(args.checkpoint)
    paths = [(arm, f) for arm, root in enumerate(args.foreground)
             for f in sorted((Path(root) / 'publish/points').glob('*/decoded.mkv'))]
    if args.points:
        paths = paths[:args.points]
    rows = []
    scratch = Path(os.environ['PS_SCRATCH_DIR']) / 'decoder-mask'
    scratch.mkdir(parents=True, exist_ok=True)
    for index, (arm, path) in enumerate(paths):
        started = time.monotonic()
        frames_dir = scratch / str(index)
        frames_dir.mkdir()
        with av.open(str(path)) as video:
            for t, frame in enumerate(video.decode(video=0)):
                if t >= n:
                    break
                image = frame.to_ndarray(format='rgb24')
                image = cv2.resize(image, (clip.width, clip.height), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(frames_dir / f'{t}.png'), image[..., ::-1])
        images, height, width = tracker.load_frames(frames_dir)
        predicted = np.zeros_like(truth)
        inter = union = 0
        for t, masks in tracker.track(images, height, width, {0: {1: initial}}, frames=n):
            full = masks[1][0]
            predicted[t] = (mk.reduce(full[None].astype(np.uint8), 8)[0] > 0).astype(np.uint8)
            exact_full = mk.upsample(truth[t:t + 1], (clip.width, clip.height))[0] > 0
            inter += int((full & exact_full).sum())
            union += int((full | exact_full).sum())
        residual = predicted ^ truth
        payload = shape.encode(residual, 1)
        restored = predicted ^ shape.decode(payload)
        key = f'arm{arm}-{path.parent.name}'
        (stage / f'{key}.correction').write_bytes(payload)
        np.savez_compressed(stage / f'{key}.npz', predicted=predicted, corrected=restored)
        record = {'arm': arm, 'point': path.parent.name, 'frames': n,
                  'seed_bytes': len(seed_bytes), 'correction_bytes': len(payload),
                  'bytes': len(seed_bytes) + len(payload) + 2,
                  'kbps': (len(seed_bytes) + len(payload) + 2)*8/(n/clip.fps)/1000,
                  'exact': bool(np.array_equal(restored, truth)),
                  'uncorrected_iou_vs_s8': inter/union if union else 1.0,
                  'prediction_sha256': hashlib.sha256(predicted.tobytes()).hexdigest(),
                  'correction_sha256': hashlib.sha256(payload).hexdigest(),
                  'seconds': time.monotonic()-started}
        rows.append(record)
        result = {'item': args.item, 'frames': n, 'fps': clip.fps, 'rows': rows,
                  'expected_points': len(paths), 'runtime': runtime(), 'model': tracker.load_report,
                  'native_decoder': {'pyav': av.__version__, 'libraries': {k: list(v) for k, v in av.library_versions.items()}},
                  'peak_gpu_mib': tracker.peak_gpu_mib()}
        (stage / 'decoder-mask.json').write_text(json.dumps(result, indent=2))
        publish_progress(os.environ['PS_STAGE'], len(rows))
        print(json.dumps(record), flush=True)
        # Local scratch only; preserve stage evidence and all shared inputs.
        for frame_path in frames_dir.glob('*.png'):
            frame_path.unlink()
        frames_dir.rmdir()
        del images
        torch.cuda.empty_cache()
    (stage / 'seed.shape').write_bytes(seed_bytes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
