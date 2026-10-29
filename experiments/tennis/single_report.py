"""CPU byte/score audit and overlap-only restoration comparisons."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
from typing import Any

import numpy as np

from experiments.tennis.parallel_curves import compare
from experiments.tennis.single_restoration import parse_header


def audit(path: Path) -> dict[str, Any]:
    with tarfile.open(path) as tar:
        def data(name: str) -> bytes:
            stream = tar.extractfile('publish/'+name)
            if stream is None:
                raise ValueError('Missing published member '+name)
            return stream.read()

        summary = json.loads(data('restoration.json'))
        frames = summary['frames']
        ticks = [0, 3, 6] if frames == 24 else list(range(0, 96, 3))
        checks = {'frames': frames in (24, 96), 'ticks': summary['ticks'] == ticks,
            'controls': all(v <= 1e-4 for v in summary['controls'].values()),
            'native': summary['native_double_decode'], 'decoder_only': summary['decoder_source_access'] is False}
        with np.load(io.BytesIO(data('scores.npz'))) as z:
            scores = {m: z[m].copy() for m in ('lpips', 'dists')}
            temporal = z['temporal_residual'].copy()
            checks['saved_ticks'] = z['ticks'].tolist() == ticks
        for m, value in scores.items():
            checks[m+':finite'] = value.shape == (12, 3, len(ticks)) and bool(np.isfinite(value).all())
            checks[m+':means'] = bool(np.allclose(value.mean(2), summary['means'][m], rtol=0, atol=1e-12))
        checks['temporal'] = temporal.shape == (12, 3, len(ticks)-1) and bool(np.isfinite(temporal).all())
        checks['temporal_means'] = bool(np.allclose(temporal.mean(2), summary['temporal_means'], rtol=0, atol=1e-12))
        for crf in (43, 53, 63):
            raw = data(f'anchor-{crf}.ivf')
            checks[f'{crf}:header'] = raw[:4] == b'DKIF' and struct.unpack('<HH4sHHIIII', raw[4:32])[:8] == (0, 32, b'AV01', 960, 544, 25, 1, 96)
            offset, times = 32, []
            while offset < len(raw):
                size, tick = struct.unpack('<IQ', raw[offset:offset+12])
                if size <= 0 or offset+12+size > len(raw):
                    raise ValueError('Invalid native frame extent')
                offset += 12+size
                times.append(tick)
            checks[f'{crf}:framing'] = times == list(range(96)) and offset == len(raw)
        rows = summary['configs']
        for row in rows:
            stream_bytes = len(data(f'anchor-{row["crf"]}.ivf'))
            count = stream_bytes
            if row['alpha']:
                packet = data(row['name']+'.trf')
                checks[row['name']+':profile'] = parse_header(packet) == (row['alpha'], 96)
                count += len(packet)
            checks[row['name']+':bytes'] = count == row['bytes'] and stream_bytes == row['stream_bytes']
            checks[row['name']+':rate'] = abs(count*8/(96/25)/1000-row['kbps']) < 1e-10
        with np.load(io.BytesIO(data('replay.npz'))) as z:
            images = z['images']
            checks['sample_hashes'] = [hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images] == summary['hashes']['0']
            checks['sample_shape'] = images.shape == (13, 1080, 1920, 3)
        curves: dict[str, Any] = {}
        assessments: dict[str, Any] = {}
        for ri, region in enumerate(summary['regions']):
            curves[region] = {}
            for metric, value in scores.items():
                anchor = [(rows[i]['kbps'], -float(value[i, ri].mean())) for i in (0, 4, 8)]
                for ai, alpha in enumerate((.15, .35, .60), 1):
                    indices = (ai, ai+4, ai+8)
                    curves[region][f'{alpha}:{metric}'] = compare(
                        [(rows[i]['kbps'], -float(value[i, ri].mean())) for i in indices], anchor)
                envelope = curves[region]['0.15:'+metric]['anchor_hull']
                for i, row in enumerate(rows):
                    if not row['alpha']:
                        continue
                    key = row['name']+':'+region
                    assessment = assessments.setdefault(key, {})
                    rate = row['kbps']
                    if not envelope[0][0] <= rate <= envelope[-1][0]:
                        assessment[metric] = {'status': 'UNRESOLVED/ANCHOR-COVERAGE'}
                    else:
                        target = -float(np.interp(rate, [r for r, _ in envelope], [q for _, q in envelope]))
                        distance = float(value[i, ri].mean())
                        assessment[metric] = {'distance': distance, 'anchor': target,
                            'robust_win': distance < target-.001, 'robust_loss': distance > target+.001}
        robust_whole = [key for key, value in assessments.items() if key.endswith(':whole')
                        and all(v.get('robust_win', False) for v in value.values())]
        return {'integrity_passed': all(checks.values()), 'checks': checks,
            'frames': frames, 'accepted_full': frames == 96, 'curves': curves,
            'pointwise': assessments, 'robust_whole_points': robust_whole,
            'publication_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'neural_metrics_independently_recomputed': False,
            'scope': 'Familiar Al32-tick development pilot against restricted SVT grid; no neural/full-envelope or independent confirmation claim.',
            'native_summary': summary}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.publication)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'integrity': result['integrity_passed'], 'frames': result['frames'],
                      'robust_whole_points': result['robust_whole_points']}))


if __name__ == '__main__':
    main()
