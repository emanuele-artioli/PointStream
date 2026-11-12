"""Exact fresh-Ada repeat comparison; this module does not authorize a run.

The historical accepted16 archive remains the source identity reference only;
its cross-class prediction and codec output are deliberately not substituted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_checkpoint as cp, k5d_checkpoint_fleet as cf
from experiments.tennis import k5d_cost_pilot as cost, k5d_source_oracle as oracle


def compare(reference: Path, candidate: Path, accepted16: Path) -> dict[str, Any]:
    """Verify both fresh runs and their fixed source, then compare exact output."""
    original, labels = cf.reference16(accepted16)
    reports = [json.loads((root / 'pilot.json').read_text()) for root in (reference, candidate)]
    for root, report in zip((reference, candidate), reports):
        source = json.loads((root / 'source-qualification.json').read_text())
        cost.compare_contract(original, labels, report, source, exact=False, capability=(8, 9))
        if report['frames'] != 16 or report['conversion']['yuv_sha256'] != original['conversion']['yuv_sha256']:
            raise ValueError('fixed source YUV or frame count differs')
    result = cp.compare_equivalence(reference, candidate, oracle.sha(reference / 'pilot.json'))
    for a, b in zip(reports[0]['points'], reports[1]['points']):
        for key in ('native_bytes', 'wire_bytes'):
            if a[key] != b[key]:
                raise ValueError('fresh repeat byte count differs: ' + key)
        for key in ('native_bytes', 'zero_ball_wire_bytes'):
            if a['baseline'][key] != b['baseline'][key]:
                raise ValueError('fresh repeat baseline byte count differs: ' + key)
        for key in ('wire_sha256', 'wire_bytes', 'native_sha256', 'native_bytes',
                    'decoded_yuv_sha256', 'rendered_rgb_sha256'):
            if a['receiver'][key] != b['receiver'][key]:
                raise ValueError('fresh repeat receiver differs: ' + key)
    return {**result, 'scope': 'fresh Ada repeat on calibration frames; not independent evidence'}
