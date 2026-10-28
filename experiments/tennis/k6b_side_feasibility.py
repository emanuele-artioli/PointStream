"""CPU-only accounting constraints from frozen real metric arrays; no new scores."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import lzma
import math
import struct
import tarfile
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

IDENTITIES = {
    'result': 'd571699973a7bea8e4c65c2fb82a56c318520129e2cc3385ca6a9988c0c8105d',
    'prior': 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1',
    'transport': '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7',
}
POLICIES = ('base', 'actual', 'ideal_R', 'ideal_Fhat')
METRICS = ('lpips', 'dists')


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Achievable upper concave hull of rate/negative-distance points."""
    if len(points) < 2 or any(not math.isfinite(v) for p in points for v in p):
        raise ValueError('Need finite anchor points')
    front: list[tuple[float, float]] = []
    for rate, quality in sorted(points, key=lambda p: (p[0], -p[1])):
        if rate <= 0:
            raise ValueError('Anchor rate must be positive')
        if front and (rate == front[-1][0] or quality <= front[-1][1]):
            continue
        front.append((rate, quality))
        while len(front) >= 3:
            a, b, c = front[-3:]
            if (b[1] - a[1]) * (c[0] - b[0]) >= (c[1] - b[1]) * (b[0] - a[0]):
                break
            front.pop(-2)
    if len(front) < 2:
        raise ValueError('Need a nondegenerate anchor quality range')
    return front


def side_ceiling(curve: list[tuple[float, float]], distance: float,
                 base_rate: float, sensitivity: float) -> float | None:
    """Equality boundary; strict wins require strictly LESS side rate."""
    if not all(math.isfinite(v) for v in (distance, base_rate, sensitivity)) or sensitivity < 0:
        raise ValueError('Invalid candidate or sensitivity')
    quality = -distance - sensitivity
    if quality < curve[0][1] or quality > curve[-1][1]:
        return None  # No endpoint clamping or extrapolation.
    return float(np.interp(quality, [q for _, q in curve], [r for r, _ in curve])) - base_rate


def verified(path: Path, role: str) -> str:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != IDENTITIES[role]:
        raise ValueError(f'{role} SHA256 mismatch')
    return digest


def member(tar: tarfile.TarFile, name: str) -> io.BytesIO:
    stream = tar.extractfile(name)
    if stream is None:
        raise ValueError(f'Missing member {name}')
    return io.BytesIO(stream.read())


def packing(packet: bytes, seconds: float) -> list[dict[str, Any]]:
    if not packet or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid packet duration')
    results = []
    for magic, encode, decode in (
        (b'ZL01', lambda b: zlib.compress(b, 9), zlib.decompress),
        (b'BZ01', lambda b: bz2.compress(b, 9), bz2.decompress),
        (b'XZ01', lambda b: lzma.compress(b, preset=6), lzma.decompress),
    ):
        wrapped = struct.pack('>4sI', magic, len(packet)) + encode(packet)
        tag, size = struct.unpack('>4sI', wrapped[:8])
        recovered = decode(wrapped[8:])
        if tag != magic or len(recovered) != size or recovered != packet:
            raise ValueError('Lossless wrapper roundtrip failed')
        results.append({'profile': magic.decode(), 'bytes': len(wrapped),
                        'kbps': len(wrapped) * 8 / seconds / 1000,
                        'sha256': hashlib.sha256(wrapped).hexdigest(), 'roundtrip': True})
    return results


def analyze(result: Path, prior: Path, transport: Path | None) -> dict[str, Any]:
    identities = {k: verified(p, k) for k, p in [('result', result), ('prior', prior)]}
    with tarfile.open(prior) as tar:
        row = json.load(member(tar, 'publish/quality.json'))['rows'][0]
        name = row['id'].replace('/', '__')
        with np.load(member(tar, f'publish/{name}/scores.npz')) as z:
            old = {m: z['full_' + m].copy() for m in METRICS}
        bootstrap = json.load(member(tar, f'publish/{name}/s4-bootstrap.json'))
    with tarfile.open(result) as tar:
        summary = json.load(member(tar, 'publish/oracle.json'))
        with np.load(member(tar, 'publish/scores.npz')) as z:
            scores = {m: z[m].copy() for m in METRICS}
            ticks = z['ticks'].tolist()
    if (row['id'] != 'alcaraz_hurkacz/031@129' or ticks != list(range(0, 240, 3))
            or ticks != row['ticks'] or ticks != summary['ticks']
            or summary['policies'] != list(POLICIES)):
        raise ValueError('Cohort or policy mismatch')
    rate = row['rates'][1]
    if rate['scale'] != 4 or rate['total_kbps'] != summary['rate_kbps']:
        raise ValueError('Operating point changed')
    base = rate['base_kbps']
    seconds = rate['base_ivf_bytes'] * 8 / base / 1000
    kps = rate['kps_bytes'] * 8 / seconds / 1000
    other = rate['side_kbps'] - kps
    curves = {}
    for metric in METRICS:
        v, o = scores[metric], old[metric]
        if (v.shape != (4, 80) or o.shape != (6, 80)
                or not np.isfinite(v).all() or not np.isfinite(o).all()
                or not np.array_equal(v[:2], o[[2, 3]])):
            raise ValueError('Invalid scores or changed controls')
        curves[metric] = hull([(r['base_kbps'], -float(o[2*i].mean()))
                               for i, r in enumerate(row['rates'])])
    policies = {}
    for i, policy in enumerate(POLICIES):
        by_sensitivity = {}
        for sensitivity in (0., .001):
            ceilings = {m: side_ceiling(curves[m], float(scores[m][i].mean()), base, sensitivity)
                        for m in METRICS}
            resolved = [v for v in ceilings.values() if v is not None]
            ceiling = min(resolved) if len(resolved) == len(METRICS) else None
            by_sensitivity[str(sensitivity)] = {
                'per_metric_side_ceiling_kbps': ceilings,
                'joint_side_ceiling_kbps': ceiling,
                'kps_only_sufficient': ceiling is not None and kps < ceiling,
                'other_only_sufficient': ceiling is not None and other < ceiling,
            }
        policies[policy] = by_sensitivity
    record = {
        'scope': 'Post-hoc fixed-pixel CPU accounting; no codec win, new quality or universal bound.',
        'window': row['id'], 'ticks': ticks, 'input_sha256': identities,
        'parent_jobs': ['20261027T211451Z-f6a754eb', '20261028T080453Z-34e50ec2'],
        'base_kbps': base, 'duration_seconds': seconds, 'kps_kbps': kps,
        'other_side_kbps': other, 'hulls_rate_negative_distance': curves,
        'means': {m: scores[m].mean(axis=1).tolist() for m in METRICS},
        'anchor_at_original_total': {
            m: -float(np.interp(rate['total_kbps'], [r for r, _ in curves[m]],
                               [q for _, q in curves[m]])) for m in METRICS
        },
        'policies': policies, 'boundary': 'Equality only; robust wins require side strictly below ceiling.',
    }
    if transport is not None:
        identities['transport'] = verified(transport, 'transport')
        with tarfile.open(transport) as tar:
            packet = member(tar, f'packets/{name}/keypoints.kps').getvalue()
        if len(packet) != rate['kps_bytes'] or hashlib.sha256(packet).hexdigest() != bootstrap['kps_sha256']:
            raise ValueError('Original paid KPS packet changed')
        record['packet_sha256'] = hashlib.sha256(packet).hexdigest()
        record['lossless_wrappers'] = packing(packet, seconds)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    began = time.monotonic()
    record = analyze(args.result, args.prior, args.transport)
    record['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['wall_seconds'] = time.monotonic() - began
    with args.out.open('x') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.out), 'wall_seconds': record['wall_seconds'],
                      'ideal_R': record['policies']['ideal_R']['0.001'],
                      'wrappers': record.get('lossless_wrappers', [])}))


if __name__ == '__main__':
    main()
