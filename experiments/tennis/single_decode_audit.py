"""Audit saved native reports/pixels without decoding a stored DCVC stream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile
from typing import Any


def audit(path: Path) -> dict[str, Any]:
    with tarfile.open(path) as tar:
        def data(name: str) -> bytes:
            stream = tar.extractfile('publish/'+name)
            if stream is None:
                raise ValueError('Missing native artifact '+name)
            return stream.read()

        summary = json.loads(data('diagnosis.json'))
        frames = summary['frames']
        checks: dict[str, bool] = {'cohort': frames in (4, 24)}
        rows = []
        for row in summary['rows']:
            policy = row['policy']
            for name, expected in row['reports_sha256'].items():
                checks[policy+':'+name+':report_identity'] = hashlib.sha256(data(policy+'/'+name+'.json')).hexdigest() == expected
            if not row['checks']['all_native_commands']:
                log = data(policy+'/encode.log').decode(errors='replace')
                rows.append({'policy': policy, 'native_commands_completed': False,
                    'recorded_errors': row['errors'], 'cuda_capture_error': 'operation not permitted when stream is capturing' in log})
                continue
            encoded = json.loads(data(policy+'/encode.json'))
            dec = json.loads(data(policy+'/decode.json'))
            fresh = json.loads(data(policy+'/fresh-decode.json'))
            container = data(policy+'/stream.psdc')
            magic, version, flags, structure, count = struct.unpack('>4sBBBH', container[:9])
            sha = hashlib.sha256(container).hexdigest()
            checks[policy+':container'] = (magic, version, flags, structure, count) == (b'PSDC', 1, 0, 2, frames)
            checks[policy+':container_hashes'] = all(r['container_sha256'] == sha for r in (encoded, dec, fresh))
            raw, raw_fresh = data(policy+'/decoded.yuv'), data(policy+'/fresh.yuv')
            frame_bytes = 960*544*3//2
            checks[policy+':pixel_bytes'] = len(raw) == len(raw_fresh) == frame_bytes*frames
            checks[policy+':fresh_bytes_equal'] = raw == raw_fresh
            pixel_hashes = [hashlib.sha256(raw[i*frame_bytes:(i+1)*frame_bytes]).hexdigest() for i in range(frames)]
            checks[policy+':four_passes'] = all(p['frame_sha256'] == pixel_hashes for r in (dec, fresh) for p in r['passes'])
            checks[policy+':encoder_intra'] = encoded['i_recon_sha256'] == pixel_hashes[0]
            rows.append({'policy': policy, 'native_commands_completed': True, 'frames': frames,
                'container_sha256': sha, 'container_bytes': len(container),
                'decoded_yuv_sha256': hashlib.sha256(raw).hexdigest(), 'four_passes_equal': checks[policy+':four_passes'],
                'encoder_intra_equal': checks[policy+':encoder_intra'], 'environment': encoded['environment'],
                'checkpoints': encoded['checkpoints']})
        return {'integrity_passed': all(checks.values()), 'checks': checks, 'rows': rows,
            'frames': frames, 'publication_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'native_summary': summary, 'stored_bitstream_decoded_again': False,
            'scope': 'Saved report/pixel integrity only. A successful fresh reproduction does not resolve historical nondeterminism or establish cross-device decoding.'}


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
                      'sha256': result['publication_sha256']}))


if __name__ == '__main__':
    main()
