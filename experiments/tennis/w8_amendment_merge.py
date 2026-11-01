"""Merge the one independently audited W8 trial into the accepted W9 anchors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.tennis import w7_analysis, w8_oracle, w9_repair_intervals
from experiments.tennis.w7_billing_corrected import MANIFEST

TARGET_KBPS = 46.315062222222224
SELECTION_SHA = '4f5af8c90f5fd909abac2aed123917795f388b3961e31e18cb2e0f306aa7acb0'
STREAM_SHA = '182c7eee4a0ea2e98ef0a967ecfcdf8958898ad7ce035e54f0ea91b5109b6fa3'


def screen(inbox: Path, prior_repair: dict[str, str], trial: dict[str, str], spec_sha: str) -> dict[str, Any]:
    """Fail closed unless full audit pins the frozen selection and normalized spec."""
    audit_path = Path(trial['audit'])
    w7_analysis.verify(audit_path, trial['audit_sha256'])
    audit = json.loads(audit_path.read_text())
    if audit.get('manifest_sha256') != SELECTION_SHA or audit.get('spec_sha256') != spec_sha:
        raise ValueError('trial audit does not pin the selected manifest and normalized spec')
    hybrid, anchors = w8_oracle.accepted_nad000(inbox, prior_repair)
    hashes = json.loads(MANIFEST.read_text())['results']
    merged = w9_repair_intervals.verified_merge('nad000', trial, inbox, hashes, TARGET_KBPS)
    matches = [p for p in merged['by_policy']['s2_o20'] if p.get('stream_sha256') == STREAM_SHA]
    if len(matches) != 1 or matches[0]['crf'] != 67 or matches[0]['rate_kbps'] != 29804 * 8 / 4.8 / 1000:
        raise ValueError('missing or changed selected trial')
    point = matches[0]
    new = {'kbps': float(point['rate_kbps']), **{m: float(point[f'foreground_{m}']) for m in w8_oracle.METRICS}}
    if any(p['kbps'] == new['kbps'] for p in anchors['s2_o20']):
        raise ValueError('trial rate duplicates an accepted point')
    anchors['s2_o20'].append(new)
    return {'target_kbps': TARGET_KBPS, 'hybrid': hybrid, 'anchors': anchors,
            'gate': w8_oracle.gate(TARGET_KBPS, hybrid, anchors),
            'prior_repair': prior_repair, 'trial': trial, 'spec_sha256': spec_sha,
            'selection_sha256': SELECTION_SHA, 'stream_sha256': STREAM_SHA}
