"""Recompute corrected W7 from the six independently audited repair archives."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from experiments.tennis import w7_analysis as w
from experiments.tennis import w9_repair_intervals as repair
from experiments.tennis.w9_saved_intervals import EXPECTED_RESULTS
from experiments.tennis.w7_billing_corrected import MANIFEST, REPOSITORY, TIER2


def compute(inbox: Path, repairs: dict[str, dict[str, str]]) -> dict[str, Any]:
    if set(repairs) != set(repair.REPAIR_WINDOWS):
        raise ValueError('all six audited repair windows are required')
    hashes = json.loads(MANIFEST.read_text())['results']
    tiers: dict[str, Any] = {}
    for tier, names in ((1, w.WINDOWS), (2, w.TIER2_WINDOWS)):
        result_path = REPOSITORY / f'docs/experiments/w9-inputs/w7-tier{tier}-billing-corrected.json'
        w.verify(result_path, EXPECTED_RESULTS[tier - 1])
        corrected = {r['window']: r for r in json.loads(result_path.read_text())['windows']}
        results = []
        for name in names:
            if tier == 1:
                source = w.W7_INPUTS[name]
                judge_job, score_job = source['judge']['job'], source['score']['job']
                dcvc_job = source['dcvc'][0].get('job') if source['dcvc'] else None
            else:
                judge_job, score_job, dcvc_job = TIER2[name]
            def read(job: str, kind: str) -> Any:
                path = inbox / job / 'full/published.tar'
                w.verify(path, hashes[job])
                return w.read_json(path, kind)
            quality = copy.deepcopy(read(judge_job, 'judge'))
            judge = w.hybrid_from_judge(quality, name)
            judge_row = next(r for r in quality['rows'] if r['id'] == judge['id'])
            next(r for r in judge_row['rates'] if r['scale'] == 4)['total_kbps'] = corrected[name]['rate_kbps']
            score = copy.deepcopy(read(score_job, 'score'))
            if name in repairs:
                merged = repair.verified_merge(name, repairs[name], inbox, hashes, corrected[name]['rate_kbps'])
                # Historical embedded decisions/targets describe the old rate.
                score['analysis'] = {'by_policy': merged['by_policy']}
            if name == 'al028':
                point_path = REPOSITORY / 'docs/experiments/w7-specs/dcvc-points-al028-scale2.json'
                w.verify(point_path, 'e5d05fe54feedf186bc83b44b153bd2e6ae3247d60648859ef69e73f154408f1')
                sources = [json.loads(point_path.read_text())]
            else:
                sources = [read(dcvc_job, 'dcvc')] if dcvc_job else []
            results.append(w.analyse_window(name, quality, score, sources))
        matches = ('al', 'med') if tier == 1 else ('nad',)
        tiers[f'tier{tier}'] = {'windows': results, 'pooled': w.summarize_group(results),
                                'clusters': {match: w.summarize_group([r for r in results if r['window'].startswith(match)])
                                             for match in matches},
                                'missing_families': {r['window']: r['anchors_excluded'] for r in results if r['anchors_excluded']}}
        if tier == 1:
            tiers['tier1']['sensitivity_without_missing_dcvc'] = w.summarize_group([r for r in results if r['window'] != 'med000'])
    return {'scope': 'corrected 2h W7; conditional on resolved anchor families', **tiers}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inbox', type=Path, required=True)
    p.add_argument('--repairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compute(a.inbox, json.loads(a.repairs.read_text()))
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
