"""Independent saved-byte/score audit and overlap-only comparison of memory pilot."""
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


def audit(path: Path) -> dict[str, Any]:
    with tarfile.open(path) as tar:
        def data(name: str) -> bytes:
            stream = tar.extractfile('publish/'+name)
            if stream is None:
                raise ValueError('Missing published member: '+name)
            return stream.read()

        s = json.loads(data('summary.json'))
        frames = s['frames']
        if frames not in (24, 96):
            raise ValueError('Unknown cohort')
        ticks = [0, 3, 6] if frames == 24 else list(range(0, 96, 3))
        checks: dict[str, bool] = {'ticks': s['ticks'] == ticks, 'replay': s['wire_prediction_equal'],
                                  'native_double_decode': s['native_double_decode_equal']}
        with np.load(io.BytesIO(data('scores.npz'))) as z:
            checks['saved_ticks'] = z['ticks'].tolist() == ticks
            scores = {m: z[m].copy() for m in ('lpips', 'dists')}
        for m, v in scores.items():
            checks[m+':finite'] = v.shape == (9, 3, len(ticks)) and bool(np.isfinite(v).all())
            checks[m+':means'] = np.array_equal(v.mean(2), np.asarray(s['means'][m]))
        components: dict[str, dict[str, Any]] = {}
        for row in s['configurations']:
            for name, count in row['components'].items():
                raw = data(name)
                checks[name+':size'] = len(raw) == count
                components[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                if name.endswith('.ivf'):
                    if raw[:4] != b'DKIF' or len(raw) < 32:
                        raise ValueError('Invalid native IVF')
                    expected = 1 if name == 'reference.ivf' else frames
                    offset, timestamps = 32, []
                    while offset < len(raw):
                        length, tick = struct.unpack('<IQ', raw[offset:offset+12])
                        if length <= 0 or offset+12+length > len(raw):
                            raise ValueError('Invalid native frame extent')
                        timestamps.append(tick)
                        offset += 12+length
                    checks[name+':framing'] = timestamps == list(range(expected)) and offset == len(raw)
            checks[row['name']+':total'] = sum(row['components'].values()) == row['bytes']
            checks[row['name']+':rate'] = abs(row['bytes']*8/(frames/25)/1000-row['kbps']) < 1e-10
        with np.load(io.BytesIO(data('replay.npz'))) as z:
            images = z['images']
            checks['sample_hashes'] = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in images] == s['hashes']['0']
            checks['sample_shape'] = images.shape == (10, 1080, 1920, 3)
        curves: dict[str, Any] = {}
        pointwise: dict[str, Any] = {}
        rows = s['configurations']
        for region, label in enumerate(s['regions']):
            curves[label] = {}
            for metric, v in scores.items():
                anchor = [(rows[i]['kbps'], -float(v[i, region].mean())) for i in range(6, 9)]
                for family, indices in (('background', range(3)), ('objects', range(3, 6))):
                    for price in ('kbps', 'warm_reference_kbps', 'zero_correction_payload_kbps'):
                        key = family+':'+price+':'+metric
                        curves[label][key] = compare([(rows[i][price], -float(v[i, region].mean())) for i in indices], anchor)
            for i in range(6):
                row = rows[i]
                key = row['name']+':'+label
                pointwise[key] = {}
                for metric, v in scores.items():
                    # Equal-rate interpolation only within achievable anchor hull.
                    envelope = curves[label]['background:kbps:'+metric]['anchor_hull']
                    rates = [r for r, _ in envelope]
                    rate = row['kbps']
                    if not rates[0] <= rate <= rates[-1]:
                        pointwise[key][metric] = {'status': 'UNRESOLVED/ANCHOR-COVERAGE'}
                    else:
                        target = -float(np.interp(rate, rates, [q for _, q in envelope]))
                        value = float(v[i, region].mean())
                        pointwise[key][metric] = {'candidate': value, 'anchor': target,
                            'robust_win': value < target-.001, 'robust_loss': value > target+.001}
        return {'integrity_passed': all(checks.values()), 'checks': checks,
            'publication_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'frames': frames, 'accepted_full_pilot': frames == 96, 'components': components,
            'means': s['means'], 'configurations': rows, 'curves': curves, 'pointwise_cold': pointwise,
            'scope': 'One familiar Al prefix; restricted960x544 SVT anchor grid, not full conventional/neural envelope. '
                     'Warm/zero-payload curves are accounting diagnostics, not measured cache lifetime or free-delivery claims. '
                     'Neural scores not independently recomputed.',
            'native_summary': s}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    record = audit(a.publication)
    with a.out.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'integrity_passed': record['integrity_passed'], 'frames': record['frames'],
                      'sha256': record['publication_sha256']}))


if __name__ == '__main__':
    main()
