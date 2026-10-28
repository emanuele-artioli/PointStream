"""Independent CPU audit of frozen parallel-batch bytes, scores and controls.

No codec or production decision routine is imported. This audits result
integrity; it does not recompute neural metrics or certify a codec win.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tarfile
from typing import Any
import zlib

import numpy as np

PRIOR_SHA = 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1'
METRICS = ('lpips', 'dists')
TICKS = list(range(0, 240, 3))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(tar: tarfile.TarFile, name: str) -> bytes:
    stream = tar.extractfile(name)
    if stream is None:
        raise ValueError('Missing published member: ' + name)
    return stream.read()


def verified_tar(path: Path, expected: str) -> tarfile.TarFile:
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('Publication SHA mismatch')
    return tarfile.open(path)


def ivf(data: bytes, frames: int, width: int, height: int, fps: int) -> None:
    if (len(data) < 32 or data[:4] != b'DKIF' or
            struct.unpack('<HH4sHHIIII', data[4:32])[:8] !=
            (0, 32, b'AV01', width, height, fps, 1, frames)):
        raise ValueError('IVF geometry/header mismatch')
    offset = 32
    for tick in range(frames):
        if offset + 12 > len(data):
            raise ValueError('Truncated IVF')
        size, timestamp = struct.unpack('<IQ', data[offset:offset+12])
        offset += 12
        if timestamp != tick or size <= 0 or offset + size > len(data):
            raise ValueError('Invalid IVF frame/timestamp')
        offset += size
    if offset != len(data):
        raise ValueError('Trailing IVF bytes')


def near(observed: float, expected: float) -> None:
    if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Accounting/mean mismatch: {observed} vs {expected}')


def arrays(tar: tarfile.TarFile, name: str, policies: int) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(get(tar, name))) as z:
        if z['ticks'].tolist() != TICKS:
            raise ValueError('Wrong full scoring cohort')
        result = {m: z[m].copy() for m in METRICS}
    if any(a.shape != (policies, 80) or not np.isfinite(a).all() for a in result.values()):
        raise ValueError('Incomplete/nonfinite full scores')
    return result


def audit_background(tar: tarfile.TarFile, old: dict[str, np.ndarray],
                     old_rates: list[dict[str, Any]]) -> dict[str, Any]:
    summary = json.loads(get(tar, 'publish/summary.json'))
    score = arrays(tar, 'publish/scores.npz', 5)
    if summary['frames'] != 240 or summary['ticks'] != TICKS:
        raise ValueError('Wrong background cohort')
    for metric in METRICS:
        if not np.array_equal(score[metric][:2], old[metric][[2, 3]]):
            raise ValueError('Background historical controls changed')
    reference = get(tar, 'publish/reference.ivf')
    ivf(reference, 1, 960, 544, 25)
    rows = []
    for i, crf in enumerate((45, 55, 63)):
        residual = get(tar, f'publish/residual-q{crf}.ivf')
        header_bytes = get(tar, f'publish/packet-q{crf}.json')
        header = json.loads(header_bytes)
        ivf(residual, 240, 960, 544, 25)
        if (header['reference_sha256'] != sha(reference) or header['residual_sha256'] != sha(residual)
                or header['transform'] != 'identity' or header['residual_decode'] !=
                {'scale': 2, 'offset': -256, 'clip': [0, 255]}):
            raise ValueError('Background decoder contract changed')
        receipt = summary['packets'][i]
        paid_bytes = len(reference) + len(residual) + len(header_bytes)
        if paid_bytes != receipt['total_bytes']:
            raise ValueError('Background incomplete byte accounting')
        rate = paid_bytes * 8 / 9.6 / 1000 + summary['side_kbps']
        near(rate, receipt['rate_kbps'])
        means = {m: float(score[m][i+2].mean()) for m in METRICS}
        for m in METRICS:
            near(means[m], summary['assessment'][i]['means'][m])
        # A measured cheaper encode is achievable at this budget by leaving
        # bits unused. This is finite dominance, not curve extrapolation.
        dominators = []
        for j, anchor in enumerate(old_rates):
            if anchor['base_kbps'] <= rate:
                winning_metrics = [m for m in METRICS if means[m] > float(old[m][2*j].mean()) + .001]
                if winning_metrics:
                    dominators.append({'rate_kbps': anchor['base_kbps'], 'better_metrics': winning_metrics,
                                       'means': {m: float(old[m][2*j].mean()) for m in METRICS}})
        rows.append({'crf': crf, 'bytes': paid_bytes, 'rate_kbps': rate, 'mean': means,
                     'measured_lower_rate_dominance': dominators})
    return {'integrity_passed': True, 'rows': rows, 'historical_control_max_error': 0.,
            'scope': 'Three static-reference constructions on Al only; no arbitrary-background bound.'}


def audit_pixels(tar: tarfile.TarFile, old: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = json.loads(get(tar, 'publish/pixels.json'))
    if summary['frames'] != 240 or len(summary['rows']) != 2:
        raise ValueError('Incomplete pixel cohort')
    results = []
    for wi, name in enumerate(('al', 'med')):
        row = summary['rows'][wi]
        scores = arrays(tar, f'publish/{name}/scores.npz', 4)
        packet = get(tar, f'publish/{name}/geometry.pxt')
        raw = zlib.decompress(packet[8:])
        if packet[:4] != b'PXT1' or len(raw) != struct.unpack('<I', packet[4:8])[0]:
            raise ValueError('Corrupt delivered geometry')
        header = json.loads(raw)
        positions = np.asarray(header['positions'])
        if (positions.shape != (240, 2, 2) or np.any(positions < 0) or np.any(positions % 2)
                or header['sizes'] != row['sizes'] or header['order'] != [0, 1]):
            raise ValueError('Incorrect delivered rectangle geometry')
        for p, (w, h) in enumerate(header['sizes']):
            if w % 2 or h % 2 or np.any(positions[:, p, 0] + w > 1920) or np.any(positions[:, p, 1] + h > 1080):
                raise ValueError('Delivered rectangles exceed canvas')
        for m in METRICS:
            for i, value in enumerate(row['means'][m]):
                near(float(scores[m][i].mean()), value)
            if name == 'al' and np.max(np.abs(scores[m][0] - old[m][2])) > 1e-4:
                raise ValueError('Pixel historical base control changed')
        points = []
        for i, crf in enumerate((32, 45, 58)):
            prefix = f'publish/{name}/q{crf}/'
            if get(tar, prefix + 'geometry.pxt') != packet:
                raise ValueError('Operating point lacks complete geometry')
            base = get(tar, prefix + 'base.ivf')
            ivf(base, 240, 480, 272, row['fps'])
            paid_bytes = len(packet) + len(base)
            for p, (w, h) in enumerate(header['sizes']):
                data = get(tar, prefix + f'player-{p}.ivf')
                ivf(data, 240, w, h, row['fps'])
                paid_bytes += len(data)
            rate = row['rates'][i]
            if paid_bytes != rate['bytes']:
                raise ValueError('Missing pixel payload cost')
            near(paid_bytes * 8 * row['fps'] / 240 / 1000, rate['kbps'])
            points.append({'crf': crf, 'bytes': paid_bytes, 'rate_kbps': rate['kbps'],
                           'mean': {m: float(scores[m][i+1].mean()) for m in METRICS}})
        for tick in (0, 3, 6):
            with np.load(io.BytesIO(get(tar, f'publish/{name}/replay-{tick:03d}.npz'))) as z:
                images = z['images']
            if images.shape != (5, 1080, 1920, 3) or [sha(im.tobytes()) for im in images] != row['hashes'][str(tick)]:
                raise ValueError('Saved replay image changed')
            outside = np.ones((1080, 1920), bool)
            for p, (w, h) in enumerate(header['sizes']):
                x, y = positions[tick, p]
                outside[y:y+h, x:x+w] = False
            if any(not np.array_equal(image[outside], images[1][outside]) for image in images[2:]):
                raise ValueError('Pixel decoder altered undisclosed support')
        results.append({'window': row['window'], 'points': points, 'sizes': header['sizes']})
    return {'integrity_passed': True, 'rows': results,
            'scope': 'Scores/bytes/geometry/native framing and three saved support replays; no neural metric recomputation.'}


def audit_baselines(tar: tarfile.TarFile) -> dict[str, Any]:
    summary = json.loads(get(tar, 'publish/summary.json'))
    expected_points = ([('svt_roi', 43, 4), ('svt_roi', 53, 4), ('svt_roi', 63, 4), ('svt_plain', 63, 1)]
                       if summary.get('roi_only') else
                       [('dcvc', 0, 4), ('dcvc', 21, 2), ('dcvc', 42, 1),
                        ('svt_roi', 43, 4), ('svt_roi', 53, 4), ('svt_roi', 63, 4)])
    if summary['frames'] != 240 or len(summary['rows']) != 2 * len(expected_points):
        raise ValueError('Incomplete full baseline curve')
    for window in ('al', 'med'):
        if [(r['codec'], r['qp'], r['scale']) for r in summary['rows'] if r['window'] == window] != expected_points:
            raise ValueError('Baseline point grid changed')
    points = []
    for row in summary['rows']:
        data = get(tar, 'publish/' + row['stream'])
        if sha(data) != row['stream_sha256'] or len(data) != row['bytes'] or row['ticks'] != TICKS:
            raise ValueError('Baseline stream/cohort mismatch')
        fps = 25 if row['window'] == 'al' else 50
        near(len(data) * 8 * fps / 240 / 1000, row['rate_kbps'])
        if row['codec'].startswith('svt_'):
            ivf(data, 240, row['size'][0], row['size'][1], fps)
        elif struct.unpack('>4sBBBH', data[:9]) != (b'PSDC', 1, 0, 2, 240):
            raise ValueError('Incorrect native DCVC container')
        for m in METRICS:
            values = np.asarray(row['scores'][m])
            if values.shape != (80,) or not np.isfinite(values).all():
                raise ValueError('Invalid baseline scores')
            near(float(values.mean()), row['mean'][m])
        points.append({k: row[k] for k in ('window', 'codec', 'qp', 'scale', 'rate_kbps', 'mean')})
    return {'integrity_passed': True, 'rows': points,
            'scope': 'Native stream/count/rate and raw-score means; neural scores not independently recomputed.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', type=Path, required=True)
    for name in ('background', 'pixels', 'baselines'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--' + name + '-sha')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with verified_tar(a.prior, PRIOR_SHA) as tar:
        prior_row = json.loads(get(tar, 'publish/quality.json'))['rows'][0]
        with np.load(io.BytesIO(get(tar, 'publish/alcaraz_hurkacz__031@129/scores.npz'))) as z:
            old = {m: z['full_' + m].copy() for m in METRICS}
    output: dict[str, Any] = {'audit_code_sha256': sha(Path(__file__).read_bytes()),
                              'prior_sha256': PRIOR_SHA, 'neural_metrics_recomputed': False}
    for name in ('background', 'pixels', 'baselines'):
        path = getattr(a, name)
        if path is None:
            continue
        with verified_tar(path, getattr(a, name + '_sha')) as tar:
            result = (audit_background(tar, old, prior_row['rates']) if name == 'background' else
                      audit_pixels(tar, old) if name == 'pixels' else audit_baselines(tar))
            result['publication_sha256'] = getattr(a, name + '_sha')
            result['members_sha256'] = {m.name: sha(get(tar, m.name)) for m in tar if m.isfile()}
            output[name] = result
    with a.out.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(a.out), 'audited_tasks': list(output)[3:]}))


if __name__ == '__main__':
    main()
