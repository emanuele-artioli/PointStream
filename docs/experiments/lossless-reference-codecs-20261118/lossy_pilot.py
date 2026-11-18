"""Measure JPEG, WebP and AVIF reconstruction at the existing JPEG payload sizes."""
import hashlib
import json
import math
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire
from experiments.tennis.player_requalification import array_sha

ROOT = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib')
REGISTRY = Path('docs/experiments/joint-readiness-20261118/linux-carriers-zlib.json')
MANIFEST = Path('docs/experiments/player-reference-pilot-20261117/prepared-manifest.json')
OUT = Path('docs/experiments/lossless-reference-codecs-20261118/lossy-result.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def call(args):
    subprocess.run(args, check=True, capture_output=True, timeout=120)


def image(path):
    return np.array(Image.open(path).convert('RGB'))


def psnr(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(delta * delta)
    return float(10 * math.log10(255 * 255 / mse)) if mse else None


def sections(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError('carrier identity differs')
    return wire.unpack(data, packet._embedded_profile(data))[0]


def main():
    start = time.monotonic()
    registry = json.loads(REGISTRY.read_text())['carriers']
    manifest = json.loads(MANIFEST.read_text())['40']
    selected = sections(ROOT / 'qpng.pscp', registry['png']['sha256'])
    jpeg = sections(ROOT / 'q40.pscp', registry['40']['sha256'])
    prepared = {u['unit']: u for u in manifest['units']}
    result = {'registered_budget': '2 CPU minutes, 4 wall minutes, zero GPU',
              'jpeg_quality': 40, 'jpeg_carrier_sha256': registry['40']['sha256'],
              'png_carrier_sha256': registry['png']['sha256'], 'references': []}
    with tempfile.TemporaryDirectory(prefix='ps-lossy-refs-', dir='/private/tmp') as folder:
        root = Path(folder)
        for ref in ('ref0', 'ref1'):
            meta, raw = packet._bounded_transport(selected[ref], b'REF1', packet.MAX_REF)
            jpeg_meta, jpeg_raw = packet._bounded_transport(jpeg[ref], b'REF1', packet.MAX_REF)
            if jpeg_meta['unit'] != meta['unit'] or sha(jpeg[ref]) != registry['40']['sections'][ref]:
                raise ValueError('JPEG payload identity differs from reviewed pilot')
            source = root / f'{ref}.png'
            source.write_bytes(raw)
            rgb = image(source)
            jpeg_rgb = cv2.imdecode(np.frombuffer(jpeg_raw, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            same_reviewed = array_sha(jpeg_rgb) == prepared[meta['unit']]['decoded_rgb_sha256']
            row = {'unit': meta['unit'], 'jpeg': {'bytes': len(jpeg_raw), 'sha256': sha(jpeg_raw),
                   'psnr_db': psnr(rgb, jpeg_rgb), 'decoded_matches_reviewed_rgb': same_reviewed}, 'candidates': {}}
            for codec in ('webp', 'avif'):
                trials = []
                lo, hi = 0, 100
                for index in range(8):
                    quality = 0 if index == 0 else (lo + hi + 1) // 2
                    output = root / f'{ref}-{codec}.{codec}'
                    if codec == 'webp':
                        cmd = ['/opt/homebrew/bin/cwebp', '-quiet', '-q', str(quality), '-m', '6', '-metadata', 'none', str(source), '-o', str(output)]
                    else:
                        cmd = ['/opt/homebrew/bin/avifenc', '-q', str(quality), '--speed', '8', '--jobs', '1', str(source), str(output)]
                    call(cmd)
                    data = output.read_bytes()
                    feasible = len(data) <= len(jpeg_raw)
                    trials.append({'quality': quality, 'bytes': len(data), 'sha256': sha(data), 'feasible': feasible})
                    if feasible:
                        best = (quality, data)
                        lo = quality
                    else:
                        hi = quality - 1
                    if index == 0 and not feasible:
                        best = None
                        break
                    if lo >= hi:
                        break
                if best is None:
                    row['candidates'][codec] = {'feasible': False, 'trials': trials}
                    continue
                quality, data = best
                output.write_bytes(data)
                decoded_path = root / f'{ref}-{codec}-decoded.png'
                decoder = ['/opt/homebrew/bin/dwebp', '-quiet', str(output), '-o', str(decoded_path)] if codec == 'webp' else ['/opt/homebrew/bin/avifdec', str(output), str(decoded_path)]
                call(decoder)
                candidate = image(decoded_path)
                if candidate.shape != rgb.shape:
                    raise ValueError('decoded image dimensions differ')
                row['candidates'][codec] = {'feasible': True, 'quality': quality, 'bytes': len(data),
                                            'sha256': sha(data), 'psnr_db': psnr(rgb, candidate), 'trials': trials}
            result['references'].append(row)
    self_time = resource.getrusage(resource.RUSAGE_SELF)
    child_time = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['cpu_seconds'] = self_time.ru_utime + self_time.ru_stime + child_time.ru_utime + child_time.ru_stime
    result['wall_seconds'] = time.monotonic() - start
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['references']:
        print(row['unit'], row['jpeg'], {k: (v.get('bytes'), v.get('quality'), v.get('psnr_db')) for k, v in row['candidates'].items()})
    print('CPU', result['cpu_seconds'], 'wall', result['wall_seconds'])


if __name__ == '__main__':
    main()
