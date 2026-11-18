"""Check lossless image payloads for the selected paid receiver references."""
import hashlib
import json
import resource
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b_rate_streams as rates

CARRIER = Path('/Users/manu/Documents/PointStream-prepared/2026-11-18-receiver-replay-zlib/qpng.pscp')
EXPECTED = '3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd'
TOOLS = {name: f'/opt/homebrew/bin/{name}' for name in ('cwebp', 'dwebp', 'avifenc', 'avifdec')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def call(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=True)


def ref1_size(meta, data):
    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    raw = struct.pack('>I', len(meta_bytes)) + meta_bytes + data
    return rates.HEADER.size + len(zlib.compress(raw, 6))


def main():
    start = time.monotonic()
    carrier = CARRIER.read_bytes()
    if sha(carrier) != EXPECTED:
        raise ValueError('selected carrier identity changed')
    sections, _, _ = packet.unpack(carrier)
    result = {'carrier_sha256': EXPECTED, 'input': str(CARRIER), 'tools': {}, 'references': []}
    for name, path in TOOLS.items():
        result['tools'][name] = {'path': path, 'sha256': sha(Path(path).read_bytes()),
                                 'version': call([path, '-version' if name.endswith('webp') else '--version']).stdout.strip()}
    with tempfile.TemporaryDirectory(prefix='ps-reference-codecs-', dir='/private/tmp') as folder:
        root = Path(folder)
        for name in ('ref0', 'ref1'):
            meta, raw = packet._bounded_transport(sections[name], b'REF1', packet.MAX_REF)
            source = root / f'{name}.png'
            source.write_bytes(raw)
            original = np.array(Image.open(source).convert('RGB'))
            row = {'unit': meta['unit'], 'section': name, 'rgb_shape': list(original.shape),
                   'original_rgb_sha256': sha(original.tobytes()),
                   'png': {'payload_bytes': len(raw), 'payload_sha256': sha(raw),
                           'actual_ref1_bytes': len(sections[name])}, 'candidates': {}}
            if ref1_size(meta, raw) != len(sections[name]):
                raise ValueError('REF1 reproduction differs from paid section length')
            codecs = {
                'webp': ([TOOLS['cwebp'], '-quiet', '-lossless', '-m', '6', '-metadata', 'none', str(source), '-o', str(root / f'{name}.webp')],
                         [TOOLS['dwebp'], '-quiet', str(root / f'{name}.webp'), '-o', str(root / f'{name}-webp.png')]),
                'avif': ([TOOLS['avifenc'], '--lossless', '--speed', '8', '--jobs', '1', str(source), str(root / f'{name}.avif')],
                         [TOOLS['avifdec'], str(root / f'{name}.avif'), str(root / f'{name}-avif.png')]),
            }
            for codec, (encode, decode) in codecs.items():
                timer = time.monotonic()
                call(encode)
                call(decode)
                decoded = np.array(Image.open(root / f'{name}-{codec}.png').convert('RGB'))
                equal = decoded.shape == original.shape and np.array_equal(decoded, original)
                data = (root / f'{name}.{codec}').read_bytes()
                alternative_meta = dict(meta, codec=f'.{codec}')
                row['candidates'][codec] = {'payload_bytes': len(data), 'payload_sha256': sha(data),
                                            'hypothetical_ref1_bytes': ref1_size(alternative_meta, data),
                                            'exact_rgb': bool(equal), 'decoded_rgb_sha256': sha(decoded.tobytes()),
                                            'encode_decode_wall_seconds': time.monotonic()-timer,
                                            'encode_args': encode[1:-3] if codec == 'webp' else encode[1:-2]}
            result['references'].append(row)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['cpu_seconds'] = usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime
    result['wall_seconds'] = time.monotonic() - start
    result['scope'] = 'RGB identity and hypothetical REF1 bytes; alternate full packet and receiver are unimplemented'
    Path('docs/experiments/lossless-reference-codecs-20261118/result.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['references']:
        print(row['unit'], row['png']['actual_ref1_bytes'], row['candidates'])
    print('CPU', result['cpu_seconds'], 'wall', result['wall_seconds'])


if __name__ == '__main__':
    main()
