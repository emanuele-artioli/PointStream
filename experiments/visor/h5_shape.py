"""CPU-only H5p 4d binary shape pilot; immutable staged B1b inputs."""
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
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--masks')
    parser.add_argument('--item')
    parser.add_argument('--frames', type=int, default=240)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        result = json.loads((stage / 'shape.json').read_text())
        rows = result['rows']
        checks = {'rows': len(rows) == 3, 'frames': result['frames'] > 0,
                  'roundtrips': all(r['roundtrip'] for r in rows),
                  'accuracy_identical': all(r['accuracy_identical'] for r in rows),
                  'nonzero_rate': all(r['bytes'] > 20 and np.isfinite(r['kbps']) for r in rows),
                  'cpu_only': result['cuda_visible'] == ''}
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}))
        return int(not all(checks.values()))
    source = Path(args.masks) / 'publish/masks' / args.item / 'masks.rle'
    clip = ClipMasks.load(source.parent)
    n = min(args.frames, len(clip))
    reduced = []
    class_reduced = []
    for t in range(n):
        frame = np.where(clip.foreground(t), 2, 0).astype(np.uint8)
        frame[clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1])] = 1
        small = mk.reduce(frame[None], 8)[0]
        class_reduced.append(small)
        reduced.append((small > 0).astype(np.uint8))
    masks = np.stack(reduced)
    classes = np.stack(class_reduced)
    baseline = mk.encode(classes, 8)
    seconds = n / clip.fps
    rows = []
    for mode in range(3):
        start = time.monotonic()
        payload = shape.encode(masks, mode)
        decoded = shape.decode(payload)
        path = stage / f'mode{mode}.shape'
        path.write_bytes(payload)
        rows.append({'mode': mode, 'bytes': len(payload), 'kbps': len(payload)*8/seconds/1000,
                     'roundtrip': bool(np.array_equal(decoded, masks)),
                     'accuracy_identical': bool(np.array_equal(decoded > 0, classes > 0)),
                     'sha256': hashlib.sha256(payload).hexdigest(), 'seconds': time.monotonic()-start})
        publish_progress(os.environ['PS_STAGE'], len(rows))
        print(json.dumps(rows[-1]), flush=True)
    result = {'item': args.item, 'frames': n, 'fps': clip.fps, 'scale': 8,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'reduced_sha256': hashlib.sha256(masks.tobytes()).hexdigest(),
              'lzma_class_kbps': len(baseline)*8/seconds/1000, 'rows': rows,
              'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES')}
    (stage / 'shape.json').write_text(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
