"""CPU scenario audit: compare saved-pixel side ceilings with paid dynamic data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def analyze(gap: dict[str, Any]) -> dict[str, Any]:
    dynamic = 4236*8/9.6/1000  # Real frozen KPS packet, including its native framing.
    ceiling = gap['policies']['clean_actual']['joint_side_ceiling_kbps']
    scenarios = [
        {'name': 'existing delivery', 'remaining_side_kbps': gap['side_kbps']},
        {'name': 'all static overhead free; dynamic KPS paid', 'remaining_side_kbps': dynamic},
        {'name': 'all side information free (nondeployable diagnostic)', 'remaining_side_kbps': 0.},
    ]
    for row in scenarios:
        row['within_joint_equality_ceiling'] = row['remaining_side_kbps'] <= ceiling
        row['excess_kbps'] = max(0., row['remaining_side_kbps']-ceiling)
    return {'scope': 'Fixed clean-actual pixels on familiar Al only; no new quality and no universal bound.',
        'joint_side_equality_ceiling_kbps': ceiling, 'dynamic_kps_kbps': dynamic,
        'scenarios': scenarios,
        'decision': 'Static-asset discounts alone cannot rescue these fixed pixels; dynamic transport or prediction must change.',
        'cache_formula': 'effective_bps = ongoing_bps + 8*asset_bytes/lifetime_seconds + refresh_bps',
        'rules': ['Cold and warm delivery both reported; no arbitrary infinite lifetime.',
                  'Decoded-history reuse does not resend past pixels; startup and identifiers remain paid.',
                  'General preinstalled decoder must transfer to unseen content; content deltas paid.',
                  'Anchors get equal cache/adaptation privileges.',
                  'Viewer-count amortization does not reduce a single receiver download.'],
        'restoration_audit': {
            'method': 'PnP-VCVE CVPR2025', 'code': 'https://github.com/ZeldaM1/PnP-VCVE',
            'source': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_Plug-and-Play_Versatile_Compressed_Video_Enhancement_CVPR_2025_paper.pdf',
            'available_setup': 'H.264 decoded video plus codec-specific motion/partition/quality metadata.',
            'local_path': 'SVT-AV1/dav1d and DCVC-UF; no audited compatible PnP checkpoint or metadata adapter.',
            'decision': 'INPUT-COMPATIBILITY STOP; do not count H.264 checkpoint scores as an AV1 test.',
            'training_or_inference_run': False},
        'next': 'Measure paid decoded-memory predictors; conditional feature coding and general restoration remain separate follow-ons.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gap', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    raw = a.gap.read_bytes()
    result = analyze(json.loads(raw))
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=time.monotonic()-start)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'decision': result['decision'], 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
