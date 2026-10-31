"""Apply W9 interval enclosures to the original, saved W7 anchor curves.

This is the original-family-exclusion sensitivity, not a complete coverage
claim. No new repair quality results are read. The inbox is read only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.tennis import w5_dcvc, w7_analysis, w9_tier_intervals
from experiments.tennis.w7_billing_corrected import MANIFEST, REPOSITORY, TIER2

EXPECTED_RESULTS = (
    '62376a4f3dcbe406d2065e54c12ebcbc0d2579aeb97c243cbf843bdf15a66e1f',
    'a60ea553077203bd3bbf894843bbe826d613c7c1effdab371ddad0b7b539d979',
)
ADAPTERS = {'al': 1_488_252, 'med': 1_495_592, 'nad': 1_489_876}


def load_tiers(inbox: Path) -> tuple[list[list[dict[str, Any]]], dict[str, list[str]]]:
    hashes = json.loads(MANIFEST.read_text())['results']
    tiers = []
    omissions: dict[str, list[str]] = {}
    for tier, names in ((1, w7_analysis.WINDOWS), (2, w7_analysis.TIER2_WINDOWS)):
        result_path = REPOSITORY / f'docs/experiments/w9-inputs/w7-tier{tier}-billing-corrected.json'
        w7_analysis.verify(result_path, EXPECTED_RESULTS[tier - 1])
        by_window = {r['window']: r for r in json.loads(result_path.read_text())['windows']}
        windows = []
        for name in names:
            score_job = w7_analysis.W7_INPUTS[name]['score']['job'] if tier == 1 else TIER2[name][1]
            archive = inbox / score_job / 'full/published.tar'
            w7_analysis.verify(archive, hashes[score_job])
            score = w7_analysis.read_json(archive, 'score')
            result = by_window[name]
            bill = result['billing_correction']
            adapter = ADAPTERS[w7_analysis.match_of(name)]
            anchor = {k: v for k, v in score['analysis']['by_policy'].items() if k in w7_analysis.SVT}
            anchor['dcvc'] = w5_dcvc.points([tuple(r) for r in result['dcvc_points']])
            required = [f for f in w7_analysis.ANCHORS if anchor.get(f)]
            omissions[name] = [f for f in w7_analysis.ANCHORS if f not in required]
            windows.append({'fixed_kbps': bill['old_rate_kbps'] - (adapter + bill['old_bank_bytes']) * 8 / 7200 / 1000,
                            'one_time_bytes': adapter + 6 * bill['png_bytes_two_crops'],
                            'hybrid': {m: result['hybrid'][m]['foreground'] for m in w7_analysis.METRICS},
                            'anchors': anchor, 'required': required})
        tiers.append(windows)
    return tiers, omissions


def compute(inbox: Path) -> dict[str, Any]:
    tiers, omissions = load_tiers(inbox)
    return {'scope': 'conditional original-family-exclusion sensitivity; no repair scores', 'omitted_families': omissions,
            'tier1': w9_tier_intervals.intervals(tiers[0]), 'tier2': w9_tier_intervals.intervals(tiers[1]),
            'joint': w9_tier_intervals.joint_intervals(tiers)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compute(args.inbox)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
