"""Materialize decoded reference conditioning; preserve native pose/timing arrays."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_transport import array_sha, sha, write


def prepare(original: Path, video: Path, kind: str, target: Path) -> dict[str, Any]:
    import cv2
    if target.exists() or kind not in ('full', 'alpha', 'black', 'black-affine'):
        raise ValueError('New target and declared recipe required')
    floor, crf = (32 if kind.startswith('black') else 0), 20
    decoded = {}
    banks = {}
    with tarfile.open(video) as source:
        f = source.extractfile('publish/'+('affine.json' if kind == 'black-affine' else 'reference-video.json'))
        if f is None:
            raise ValueError('Missing sparse video report')
        report = json.load(f)
        for row in report['rows']:
            if row['kind'] != kind or row['crf'] != crf or row.get('floor') != floor:
                continue
            rec = row['container']
            stream = source.extractfile('publish/'+rec['file'])
            if stream is None or sha(stream.read()) != rec['sha256']:
                raise ValueError('Delivered bank hash differs')
            contract = rec['contract']
            file = source.extractfile('publish/'+Path(rec['file']).stem+'-conditioning.npz')
            if file is None:
                raise ValueError('Missing decoded conditioning')
            with np.load(io.BytesIO(file.read())) as arrays:
                rgb, alpha = arrays['rgb'], arrays['alpha']
                if array_sha(rgb) != row['diagnostic_only']['decoded_rgb_sha256']:
                    raise ValueError('Decoded reference pixels changed')
                for index, unit in enumerate(contract['units']):
                    mask = cv2.resize(alpha[index].astype(np.float32), (512, 784), interpolation=cv2.INTER_LINEAR) >= .5
                    # Actual client reconstruction, no source pixels are used for missing support.
                    image = rgb[index] if kind == 'full' else np.where(mask[..., None], rgb[index], 0).astype(np.uint8)
                    decoded[unit['unit']] = image
            banks[row['match']] = {k: rec[k] for k in ('bytes', 'sha256', 'file')}
    changed = {}
    with tarfile.open(original) as source, tarfile.open(target, 'w') as output:
        for member in source:
            if not member.isfile():
                continue
            file = source.extractfile(member)
            if file is None:
                raise ValueError('Missing native prepared member')
            blob = file.read()
            if member.name.startswith('publish/units/') and member.name.endswith('.npz'):
                name = Path(member.name).stem
                image = decoded[name]
                with np.load(io.BytesIO(blob)) as arrays:
                    values = {key: arrays[key] for key in arrays.files}
                if values['reference'].shape != image.shape or image.shape != (784, 512, 3):
                    raise ValueError('Reference geometry changed')
                before = {key: array_sha(value) for key, value in values.items()}
                values['reference'] = image
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **values)
                blob = buffer.getvalue()
                with np.load(io.BytesIO(blob)) as back:
                    if any(array_sha(back[key]) != digest for key, digest in before.items() if key != 'reference'):
                        raise ValueError('Reference variant changes motion/timing/geometry')
                changed[name] = {'original_arrays_sha256': before,
                                 'decoded_reference_sha256': array_sha(image), 'npz_sha256': sha(blob)}
            info = tarfile.TarInfo(member.name)
            info.size = len(blob)
            output.addfile(info, io.BytesIO(blob))
    if len(changed) != 6 or len(banks) != 2:
        raise ValueError('First original prepared group / both banks required')
    return {'original_prepared_sha256': sha(original.read_bytes()), 'video_archive_sha256': sha(video.read_bytes()),
            'kind': kind, 'crf': crf, 'floor': floor, 'banks': banks, 'units': changed,
            'target': str(target), 'target_sha256': sha(target.read_bytes()),
            'only_reference_arrays_changed': True, 'generated_quality_measured': False,
            'segmented_outside_alpha_fill': 'black-after-bicubic-RGB-and-bilinear-alpha-restore'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--affine-only', action='store_true')
    a = p.parse_args()
    a.out_dir.mkdir(exist_ok=False)
    if sha(a.original.read_bytes()) != '674c40981a7d805a2d241b12bf499ca1a6e99351234581edebe30667cb6bfa0c':
        raise ValueError('Original first-group preparation identity changed')
    kinds = ('black-affine',) if a.affine_only else ('full', 'alpha', 'black')
    records = [prepare(a.original, a.video, kind, a.out_dir/f'{kind}-q20-prepared.tar') for kind in kinds]
    write(a.out_dir/'reference-variants.json', {'rows': records})
    print(json.dumps({'variants': len(records), 'changed_units_each': [len(r['units']) for r in records], 'gpu_hours': 0, 'out': str(a.out_dir)}))


if __name__ == '__main__':
    main()
