"""CPU accuracy of saved H5p 4d decoder masks against original B1b F."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.jobs.monitor import publish_progress
from src.pointstream import mask as mk
from src.segmentation.masks import ClipMasks


def main() -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--masks')
    p.add_argument('--prediction', nargs=2, action='append', default=[])
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--validate', action='store_true')
    args = p.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        result = json.loads((stage / 'accuracy.json').read_text())
        rows = result['rows']
        checks = {'four_items': len(rows) == 4,
                  'exact_grid': all(r['exact_grid'] for r in rows),
                  'finite': all(np.isfinite(r[k][m]) for r in rows for k in ['baseline', 'predicted', 'corrected']
                                for m in ['iou', 'boundary_px', 'error_share']),
                  'cpu_only': result['cuda_visible'] == ''}
        Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}))
        return int(not all(checks.values()))
    rows = []
    for item, source in args.prediction:
        clip = ClipMasks.load(Path(args.masks) / 'publish/masks' / item)
        saved = np.load(source)
        n = min(args.frames, len(saved['predicted']))
        truth = np.stack([clip.foreground(t).astype(np.uint8) for t in range(n)])
        reduced = (mk.reduce(truth, 8) > 0).astype(np.uint8)
        row = {'item': item, 'frames': n, 'prediction_file_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
               'exact_grid': bool(np.array_equal(saved['corrected'][:n], reduced))}
        for key, grid in [('baseline', reduced), ('predicted', saved['predicted'][:n]), ('corrected', saved['corrected'][:n])]:
            decoded = mk.upsample(grid, (clip.width, clip.height))
            metrics = mk.accuracy(truth, decoded)
            metrics.pop('class_accuracy')  # union F only, G_HO in both slots
            row[key] = metrics
        rows.append(row)
        publish_progress(os.environ['PS_STAGE'], len(rows))
        print(json.dumps(row), flush=True)
        del truth, decoded
    (stage / 'accuracy.json').write_text(json.dumps({'rows': rows, 'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
