"""Measure frozen image-codec configurations on the same two paid references."""
import hashlib
import json
import math
import resource
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b_rate_streams as rates
from experiments.tennis import w8_payload as wire
from experiments.tennis.player_requalification import array_sha

CARRIER = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
JPEG_CARRIER = CARRIER.with_name('q40.pscp')
EXPECTED = '3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd'
EXPECTED_JPEG = 'b828b4599ec18778444b3b1fdab868c04a53b70bc3cef1c01e0bcd33aae38931'
OUT = Path('docs/experiments/common-reference-20261122/receipt.json')
TOOLS = {name: Path('/opt/homebrew/bin') / name for name in ('cwebp', 'dwebp', 'avifenc', 'avifdec')}
CONFIGURATIONS = {'jpeg': (40, 70, 90), 'webp': (65, 80, 90), 'avif': (45, 65, 75)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    return subprocess.run(args, check=True, capture_output=True, timeout=60)


def ref1_size(meta, payload):
    metadata = json.dumps(meta, separators=(',', ':')).encode()
    raw = struct.pack('>I', len(metadata)) + metadata + payload
    return rates.HEADER.size + len(zlib.compress(raw, 6))


def main():
    wall = time.monotonic()
    png_carrier = CARRIER.read_bytes()
    jpeg_carrier = JPEG_CARRIER.read_bytes()
    assert sha(png_carrier) == EXPECTED and sha(jpeg_carrier) == EXPECTED_JPEG
    png_sections, _, _ = packet.unpack(png_carrier)
    jpeg_sections, _ = wire.unpack(jpeg_carrier, packet._embedded_profile(jpeg_carrier))
    review = json.loads(Path('docs/experiments/player-reference-pilot-20261117/prepared-manifest.json').read_text())['40']
    reviewed_units = {unit['unit']: unit for unit in review['units']}
    result = {
        'status': 'cpu_image_diagnostic',
        'source_packet_sha256': EXPECTED,
        'jpeg_q40_packet_sha256': EXPECTED_JPEG,
        'configurations': CONFIGURATIONS,
        'opencv': {'version': cv2.__version__, 'build_sha256': sha(cv2.getBuildInformation().encode())},
        'tools': {},
        'rows': [],
        'scope': 'Two development reference images; reconstructed RGB PSNR; no generated-player result.',
    }
    for name, executable in TOOLS.items():
        flag = '-version' if name.endswith('webp') else '--version'
        result['tools'][name] = {
            'path': str(executable.resolve()),
            'sha256': sha(executable.read_bytes()),
            'version': run([str(executable), flag]).stdout.decode().strip(),
        }
    with tempfile.TemporaryDirectory(prefix='ps-common-ref-', dir='/private/tmp') as tmp:
        folder = Path(tmp)
        for section in ('ref0', 'ref1'):
            metadata, png = packet._bounded_transport(png_sections[section], b'REF1', packet.MAX_REF)
            jpeg_metadata, paid_jpeg = packet._bounded_transport(jpeg_sections[section], b'REF1', packet.MAX_REF)
            assert jpeg_metadata['unit'] == metadata['unit']
            assert ref1_size(metadata, png) == len(png_sections[section])
            src = folder / f'{section}.png'
            src.write_bytes(png)
            original = np.asarray(Image.open(src).convert('RGB'))
            assert original.shape == (784, 512, 3)
            for codec, qualities in CONFIGURATIONS.items():
                for quality in qualities:
                    encoded = folder / f'{section}-{codec}-{quality}.{codec}'
                    decoded = folder / f'{section}-{codec}-{quality}.png'
                    if codec == 'jpeg':
                        ok, raw = cv2.imencode('.jpg', original[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
                        assert ok
                        payload = raw.tobytes()
                        rgb = cv2.imdecode(raw, cv2.IMREAD_COLOR)[:, :, ::-1]
                        if quality == 40:
                            payload = paid_jpeg
                            rgb = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                            assert array_sha(rgb) == reviewed_units[metadata['unit']]['decoded_rgb_sha256']
                    else:
                        if codec == 'webp':
                            encode = [str(TOOLS['cwebp']), '-quiet', '-q', str(quality), '-m', '6',
                                      '-metadata', 'none', str(src), '-o', str(encoded)]
                            decode = [str(TOOLS['dwebp']), '-quiet', str(encoded), '-o', str(decoded)]
                        else:
                            encode = [str(TOOLS['avifenc']), '-q', str(quality), '--speed', '8',
                                      '--jobs', '1', str(src), str(encoded)]
                            decode = [str(TOOLS['avifdec']), str(encoded), str(decoded)]
                        run(encode)
                        run(decode)
                        payload = encoded.read_bytes()
                        rgb = np.asarray(Image.open(decoded).convert('RGB'))
                    assert rgb.shape == original.shape and rgb.dtype == np.uint8
                    mse = float(np.mean((original.astype(np.float64) - rgb.astype(np.float64)) ** 2))
                    assert mse > 0 and math.isfinite(mse)
                    entry = {
                        'unit': metadata['unit'], 'section': section, 'codec': codec, 'quality': quality,
                        'source_rgb_sha256': sha(original.tobytes()), 'payload_bytes': len(payload),
                        'payload_sha256': sha(payload), 'decoded_rgb_sha256': sha(rgb.tobytes()),
                        'psnr_rgb_db': 10 * math.log10(65025 / mse),
                        'modeled_ref1_bytes': ref1_size(dict(metadata, codec=f'.{codec}' if codec != 'jpeg' else '.jpg'), payload),
                    }
                    if codec == 'jpeg' and quality == 40:
                        entry['modeled_ref1_bytes'] = ref1_size(jpeg_metadata, payload)
                        entry['paid_ref1_bytes'] = len(jpeg_sections[section])
                        assert entry['modeled_ref1_bytes'] == entry['paid_ref1_bytes']
                    result['rows'].append(entry)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['cpu_seconds'] = usage.ru_utime + usage.ru_stime + child.ru_utime + child.ru_stime
    result['wall_seconds'] = time.monotonic() - wall
    assert result['cpu_seconds'] < 120 and result['wall_seconds'] < 240
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(row['section'], row['codec'], row['quality'], row['payload_bytes'], round(row['psnr_rgb_db'], 2))
    print('cpu', result['cpu_seconds'], 'wall', result['wall_seconds'])


if __name__ == '__main__':
    main()
