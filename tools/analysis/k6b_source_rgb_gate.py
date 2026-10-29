"""Compare saved CPU source-RGB frame hashes with the prior GPU control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    'al': ('alcaraz_hurkacz/031@129', 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'),
    'med': ('medvedev_struff/014@194', '53633ce617b12d779bc8c5746d8d44cbe090ced6c49ba79dbe246cc6abacfb08'),
}
MANIFEST_SHA = '59f97141090865bafdd7c4b0d3593492de42de16eb057e63846feea5a5526058'
GPU_RESULT_SHA = 'b0a55a5c1440463fff97efa975a661c2c65db9ce35e0696fade9d5648c39ccdc'


def checked_json(path: Path, expected: str) -> Any:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Input identity mismatch: {path}')
    return json.loads(data)


def compare(manifest: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    rows = []
    ticks = list(range(0, 240, 3))
    for tag, (window, expected_yuv) in EXPECTED.items():
        saved = [row for row in manifest['rows'] if row['window'] == tag]
        original = [row for row in gpu['rows'] if row['id'] == window]
        if len(saved) != 3 or {r['scale'] for r in saved} != {2, 4, 8} or len(original) != 1:
            raise ValueError('Wrong window or scale coverage')
        reference = saved[0]['source_rgb_frame_sha256']
        if len(reference) != len(ticks) or not all(r['frames'] == 240 and r['source_yuv_sha256'] == expected_yuv and r['source_rgb_frame_sha256'] == reference for r in saved):
            raise ValueError('Source YUV identity or cross-scale frame hashes differ')
        if original[0]['converted']['sha256'] != expected_yuv or original[0]['expected_judged'] != ticks:
            raise ValueError('Prior GPU source conversion or judged ticks differ')
        prior = original[0]['source_rgb_frame_sha256']
        if set(prior) != {str(t) for t in ticks}:
            raise ValueError('Prior GPU ticks differ')
        differing = [t for t, h in zip(ticks, reference) if h != prior[str(t)]]
        rows.append({'window': window, 'ticks': len(ticks), 'source_yuv_sha256': expected_yuv,
                     'matching': len(ticks) - len(differing), 'differing_ticks': differing})
    return {'passed': all(not r['differing_ticks'] for r in rows), 'rows': rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-manifest', required=True, type=Path)
    parser.add_argument('--gpu-result', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(checked_json(args.base_manifest, MANIFEST_SHA), checked_json(args.gpu_result, GPU_RESULT_SHA)), indent=2))


if __name__ == '__main__':
    main()
