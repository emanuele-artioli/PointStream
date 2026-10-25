"""CPU-only independent sparse packet checks and explicit fixed-quality sensitivities."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_reference_video import decode_aligned
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from experiments.tennis.k6b_rate_budget import comparison


def report(archive: Path, diagnostic: dict[str, Any], transport: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    import cv2
    with tarfile.open(archive) as tar:
        def read(name: str) -> bytes:
            stream = tar.extractfile('publish/'+name)
            if stream is None:
                raise ValueError('Missing delivered archive member')
            return stream.read()
        result = json.loads(read('reference-video.json'))
        for i, row in enumerate(result['rows']):
            rec = row.get('container', row)
            blob = read(rec['file'])
            checks[f'{i}:packet'] = len(blob) == rec['bytes'] and sha(blob) == rec['sha256']
            if 'container' in row:
                with np.load(io.BytesIO(read(Path(rec['file']).stem+'-conditioning.npz'))) as arr:
                    restored_mask = np.stack([cv2.resize(m.astype(np.float32), (512, 784), interpolation=cv2.INTER_LINEAR) >= .5 for m in arr['alpha']])
                    checks[f'{i}:decoded_conditioning'] = array_sha(arr['rgb']) == row['diagnostic_only']['decoded_rgb_sha256'] and array_sha(restored_mask) == row['diagnostic_only']['decoded_alpha_sha256']
        for i, row in enumerate(result['alignments']):
            blob = read(row['file'])
            checks[f'{i}:alignment'] = sha(blob) == row['sha256'] and array_sha(decode_aligned(blob)) == row['reduced_sha256']
    if not all(checks.values()):
        raise ValueError('Independent publication checks failed')
    wire = {x['window']: x for x in transport['rows']}
    counter = []
    for kind, floor in [('full', 0), ('alpha', 0), ('black', 1), ('black', 8), ('black', 32)]:
        for crf in (20, 35, 50):
            banks = {x['match']: x['container']['bytes'] for x in result['rows'] if x['kind'] == kind and x.get('floor') == floor and x['crf'] == crf}
            cells = []
            for row in diagnostic['rows']:
                rate = wire[row['window']]['keypoints']['bytes']*8/row['registered_seconds']/1000 + (banks[row['match']]+row['adapter_guider_bytes'])*8/7200/1000
                for metric in ('lpips', 'dists'):
                    cells.append({'window': row['window'], 'metric': metric, 'rate': rate, **comparison(row, metric, rate)})
            counter.append({'kind': kind, 'floor': floor, 'crf': crf, 'banks': banks,
                'fixed_original_quality_and_free_dynamic_matte': True,
                'wins': sum(x['won'] is True for x in cells), 'unresolved': sum(x['won'] is None for x in cells), 'cells': cells})
    return {'archive_sha256': sha(archive.read_bytes()), 'checks': checks, 'passed': True,
            'reference_rows': [x for x in result['rows'] if x['kind'] != 'mask-video'],
            'mask_rows': [x for x in result['rows'] if x['kind'] == 'mask-video'],
            'alignments': result['alignments'], 'conditional_rebilling': counter,
            'not_actual_generated_quality': True}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--transport', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = report(a.archive, json.loads(a.diagnostic.read_text()), json.loads(a.transport.read_text()))
    write(a.out, result)
    print(json.dumps({'passed': True, 'independent_checks': len(result['checks']), 'out': str(a.out)}))


if __name__ == '__main__':
    main()
