"""Independent NumPy-only audit of paid clean-background preparation and results."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
from typing import Any

import numpy as np

GOLDEN_SHA = '85f6ae06d641e8be73804d9d6428ee8f7b386cc64326082fb5f0e9b0f7f04568'
PRIOR_SHA = 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1'
TICKS = list(range(0, 240, 3))
POLICIES = ['base', 'actual', 'clean_actual', 'clean_ideal_R']
METRICS = ['lpips', 'dists']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def pixels(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(tar: tarfile.TarFile, name: str) -> io.BytesIO:
    f = tar.extractfile(name)
    if f is None:
        raise ValueError('Missing evidence: ' + name)
    return io.BytesIO(f.read())


def envelope(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    front: list[tuple[float, float]] = []
    for point in sorted(points):
        if front and point[1] <= front[-1][1]:
            continue
        front.append(point)
        while len(front) >= 3:
            a, b, c = front[-3:]
            left = (b[1] - a[1]) / (b[0] - a[0])
            right = (c[1] - b[1]) / (c[0] - b[0])
            if left < right:
                del front[-2]
            else:
                break
    return front


def audit(a: argparse.Namespace) -> dict[str, Any]:
    if sha(a.golden) != GOLDEN_SHA or sha(a.prior) != PRIOR_SHA or sha(a.prepared) != a.prepared_sha:
        raise ValueError('Frozen evidence SHA256 mismatch')
    golden = json.loads(a.golden.read_text())
    if golden['ticks'] != TICKS:
        raise ValueError('Golden cohort mismatch')
    with tarfile.open(a.prior) as tar:
        row = json.load(read(tar, 'publish/quality.json'))['rows'][0]
        with np.load(read(tar, 'publish/alcaraz_hurkacz__031@129/scores.npz')) as z:
            prior = {m: z['full_' + m].copy() for m in METRICS}
    report: dict[str, Any] = {'prepared_sha256': a.prepared_sha, 'golden_sha256': GOLDEN_SHA,
                              'prior_sha256': PRIOR_SHA, 'scope': 'CPU pixel/rate/decision audit; no neural re-judging.'}
    # Large published evidence is streamed onto local scratch, never into the checkout.
    with tempfile.TemporaryDirectory(prefix='ps-clean-audit-') as tmp:
        root = Path(tmp)
        inner = root / 'input.tar'
        with tarfile.open(a.prepared) as outer:
            stream = outer.extractfile('publish/input.tar')
            if stream is None:
                raise ValueError('Missing prepared input archive')
            with inner.open('xb') as out:
                shutil.copyfileobj(stream, out, length=2**20)
        with tarfile.open(inner) as tar:
            tar.extractall(root / 'input', filter='data')
        data = root / 'input'
        m = json.loads((data / 'manifest.json').read_text())
        if (m['ticks'] != TICKS or m['frames'] != 240 or m['policies'] != POLICIES
                or m['window'] != row['id'] or m['source_yuv_sha256'] != 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'):
            raise ValueError('Prepared cohort/source changed')
        for name, expected in m['files'].items():
            if sha(data / name) != expected:
                raise ValueError('Prepared member mismatch: ' + name)
        if len(m['files']) != 84 or set(m['hashes']) != {str(t) for t in TICKS}:
            raise ValueError('Prepared frame coverage mismatch')
        ivf = (data / 'clean.ivf').read_bytes()
        fields = struct.unpack('<HH4sHHIIII', ivf[4:32])
        if ivf[:4] != b'DKIF' or fields[:8] != (0, 32, b'AV01', 480, 272, 25, 1, 240):
            raise ValueError('Native IVF geometry/timing mismatch')
        offset, ticks = 32, []
        while offset < len(ivf):
            if offset + 12 > len(ivf):
                raise ValueError('Truncated framing')
            size, tick = struct.unpack('<IQ', ivf[offset:offset+12])
            if size <= 0 or offset + 12 + size > len(ivf):
                raise ValueError('Truncated payload')
            ticks.append(tick)
            offset += 12 + size
        if ticks != list(range(240)) or len(ivf) != m['ivf']['bytes']:
            raise ValueError('Native frame/rate count mismatch')
        if (data / 'clean.yuv').stat().st_size != 240 * 480 * 272 * 3 // 2:
            raise ValueError('Decoded YUV coverage mismatch')
        if (data / 'source-ticks.yuv').stat().st_size != 80 * 1920 * 1080 * 3 // 2:
            raise ValueError('Source YUV coverage mismatch')
        header = json.loads((data / 'bootstrap.json').read_text())
        if header['base_ivf_sha256'] != hashlib.sha256(ivf).hexdigest() or (data / 'bootstrap.json').stat().st_size != 444:
            raise ValueError('Paid receiver bootstrap mismatch')
        side = row['rates'][1]['side_kbps']
        base_rate = len(ivf) * 8 / 9.6 / 1000
        rate = base_rate + side
        if abs(m['side_kbps'] - side) > 1e-12 or abs(m['rate_kbps'] - rate) > 1e-12:
            raise ValueError('Fixed side/base accounting mismatch')
        curves = {metric: envelope([(r['base_kbps'], -float(prior[metric][2*i].mean()))
                                    for i, r in enumerate(row['rates'])]) for metric in METRICS}
        covered = all(c[0][0] <= rate <= c[-1][0] for c in curves.values())
        anchor = {metric: -float(np.interp(rate, [x for x, _ in c], [y for _, y in c]))
                  for metric, c in curves.items()} if covered else None
        gpu = None
        saved = None
        scores = None
        if a.result is not None:
            if a.result_sha is None or sha(a.result) != a.result_sha:
                raise ValueError('GPU publication SHA mismatch')
            with tarfile.open(a.result) as tar:
                saved = json.load(read(tar, 'publish/quality.json'))
                with np.load(read(tar, 'publish/scores.npz')) as z:
                    if z['ticks'].tolist() != TICKS:
                        raise ValueError('Partial GPU result cannot be accepted')
                    scores = {metric: z[metric].copy() for metric in METRICS}
                stream = tar.extractfile('publish/clean-base-rgb.npy')
                if stream is None:
                    raise ValueError('GPU base pixels absent')
                output = root / 'base.npy'
                with output.open('xb') as f:
                    shutil.copyfileobj(stream, f, length=2**20)
            if sha(output) != saved['clean_base_rgb_sha256']:
                raise ValueError('GPU base array identity mismatch')
            gpu = np.load(output, mmap_mode='r')
            if gpu.shape != (80, 1080, 1920, 3) or gpu.dtype != np.uint8:
                raise ValueError('GPU base array incomplete')
        source = np.memmap(data / 'source-ticks.yuv', np.uint8, 'r', shape=(80, 1620, 1920))
        for j, t in enumerate(TICKS):
            with np.load(data / 'frames' / f'{t:03d}.npz') as z:
                images, overlay, written, region = [z[k].copy() for k in ('images', 'overlay', 'written', 'region')]
            if (images.shape != (3, 1080, 1920, 3) or images.dtype != np.uint8
                    or overlay.shape != images[0].shape or overlay.dtype != np.uint8
                    or region.shape != (1080, 1920) or written.shape != region.shape
                    or region.dtype != bool or written.dtype != bool):
                raise ValueError('Invalid full image/support shapes')
            if [pixels(x) for x in images] != golden['hashes'][str(t)][:3]:
                raise ValueError('Original source/base/actual golden identity changed')
            if not np.array_equal(np.where(written[..., None], overlay, images[1]), images[2]):
                raise ValueError('Frozen overlay/fallback reconstruction failed')
            if pixels(np.where(region[..., None], images[0], images[1])) != golden['hashes'][str(t)][3]:
                raise ValueError('Original R oracle reconstruction changed')
            if pixels(source[j]) != m['hashes'][str(t)]['source_yuv']:
                raise ValueError('Source YUV tick identity changed')
            if gpu is not None and saved is not None:
                reconstructed = [*images, np.where(written[..., None], overlay, gpu[j]),
                                  np.where(region[..., None], images[0], gpu[j])]
                if [pixels(x) for x in reconstructed] != saved['hashes'][str(t)]:
                    raise ValueError('New GPU actual/oracle pixel reconstruction failed')
        report.update({'preparation_passed': True, 'frame_count': 80, 'golden_rgb_checks': 240,
                       'old_overlay_replays': 80, 'original_R_replays': 80,
                       'new_base_bytes': len(ivf), 'base_kbps': base_rate, 'side_kbps': side,
                       'total_kbps': rate, 'anchor': anchor, 'anchor_covered': covered})
        if saved is not None and scores is not None:
            errors = {}
            means = {}
            for metric, values in scores.items():
                if values.shape != (4, 80) or not np.isfinite(values).all():
                    raise ValueError('Incomplete/nonfinite GPU scores')
                errors[metric] = float(np.abs(values[:2] - prior[metric][[2, 3]]).max())
                if errors[metric] > 1e-4:
                    raise ValueError('Historical neural score control failed')
                means[metric] = values.mean(1).tolist()
            wins = {p: all(means[metric][i] < anchor[metric] - .001 for metric in METRICS)
                    for i, p in enumerate(POLICIES) if i >= 2} if anchor is not None else {}
            losses = {p: all(means[metric][i] > anchor[metric] + .001 for metric in METRICS)
                      for i, p in enumerate(POLICIES) if i >= 2} if anchor is not None else {}
            verdict = ('UNRESOLVED/ANCHOR-COVERAGE' if anchor is None else 'HEADROOM' if any(wins.values())
                       else 'NO-GO' if all(losses.values()) else 'MIXED/UNRESOLVED')
            if (saved['ticks'] != TICKS or saved['policies'] != POLICIES or saved['source_rgb_checks'] != 80
                    or not saved['cuda'] or 'A6000' not in saved['gpu'] or not saved['timing_secondary']
                    or saved['prepared_sha256'] != a.prepared_sha
                    or not 0 < saved['peak_allocated_mib'] <= saved['peak_reserved_mib'] < 8000
                    or abs(saved['rate_kbps'] - rate) > 1e-12
                    or saved['assessment']['verdict'] != verdict
                    or any(not np.allclose(saved['assessment']['means'][metric], means[metric],
                                           atol=1e-12, rtol=0) for metric in METRICS)):
                raise ValueError('Native decision/provenance mismatch')
            report.update({'quality_passed': True, 'result_sha256': a.result_sha, 'verdict': verdict,
                           'means': means, 'max_control_error': errors, 'new_pixel_reconstructions': 160})
    report['audit_code_sha256'] = sha(Path(__file__))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'golden', 'prior', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepared-sha', required=True)
    p.add_argument('--result', type=Path)
    p.add_argument('--result-sha')
    a = p.parse_args()
    report = audit(a)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
