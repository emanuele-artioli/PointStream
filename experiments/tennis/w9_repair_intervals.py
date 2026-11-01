"""Apply preregistered W9 interval bounds to independently audited repair scores.

Config maps window names to `{archive, sha256, audit, audit_sha256}`. Every
archive and separate full-audit identity is verified before its scored rows are
merged with the matching original archive. Incomplete configs explicitly report
missing repairs and cannot make a completed-repair claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from experiments.tennis import w1_score, w7_analysis, w9_saved_intervals, w9_tier_intervals
from experiments.tennis.w7_billing_corrected import MANIFEST, REPOSITORY, TIER2

REPAIR_WINDOWS = ('al028', 'al034', 'al043', 'nad000', 'nad001', 'nad002')


def verified_merge(name: str, row: dict[str, str], inbox: Path, hashes: dict[str, str], target: float) -> dict[str, Any]:
    """Require a separately hashed passing full audit before reading repair scores."""
    repair, audit = Path(row['archive']), Path(row['audit'])
    w7_analysis.verify(repair, row['sha256'])
    w7_analysis.verify(audit, row['audit_sha256'])
    record = json.loads(audit.read_text())
    controls = record.get('raw_control_errors', {})
    if (record.get('passed') is not True or record.get('archive_sha256') != row['sha256']
            or not record.get('checks') or not all(record['checks'].values()) or len(controls) != 3
            or any(len(v) != 4 or any(not math.isfinite(x) or not 0 <= x <= 1e-3 for x in v.values()) for v in controls.values())):
        raise ValueError(f'{name}: invalid independent full audit')
    job = w7_analysis.W7_INPUTS[name]['score']['job'] if name in w7_analysis.WINDOWS else TIER2[name][1]
    return w1_score.merge_repair_scores(inbox / job / 'full/published.tar', hashes[job], repair, row['sha256'], target)


def compute(inbox: Path, repairs: dict[str, dict[str, str]]) -> dict[str, Any]:
    tiers, omissions = w9_saved_intervals.load_tiers(inbox)
    names = (w7_analysis.WINDOWS, w7_analysis.TIER2_WINDOWS)
    all_names = set(REPAIR_WINDOWS)
    if not set(repairs) <= all_names:
        raise ValueError('unknown repair window')
    hashes = json.loads(MANIFEST.read_text())['results']
    for tier_index, tier_names in enumerate(names):
        corrected = json.loads((REPOSITORY / f'docs/experiments/w9-inputs/w7-tier{tier_index + 1}-billing-corrected.json').read_text())
        by_name = {x['window']: x for x in corrected['windows']}
        for index, name in enumerate(tier_names):
            if name not in repairs:
                continue
            merged = verified_merge(name, repairs[name], inbox, hashes, by_name[name]['rate_kbps'])
            for family, points in merged['by_policy'].items():
                if family in w7_analysis.SVT:
                    tiers[tier_index][index]['anchors'][family] = points
    strict = copy.deepcopy(tiers)
    for tier in strict:
        for window in tier:
            window['required'] = list(w7_analysis.ANCHORS)
    return {'scope': 'conditional original-family-exclusion sensitivity and strict full-family coverage',
            'audited_repairs': sorted(repairs), 'missing_repairs': sorted(all_names - set(repairs)),
            'omitted_families': omissions,
            'conditional': {'tier1': w9_tier_intervals.intervals(tiers[0]),
                            'tier2': w9_tier_intervals.intervals(tiers[1]),
                            'joint': w9_tier_intervals.joint_intervals(tiers)},
            'strict': {'tier1': w9_tier_intervals.intervals(strict[0]),
                       'tier2': w9_tier_intervals.intervals(strict[1]),
                       'joint': w9_tier_intervals.joint_intervals(strict)}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox', type=Path, required=True)
    p.add_argument('--repairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compute(args.inbox, json.loads(args.repairs.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
