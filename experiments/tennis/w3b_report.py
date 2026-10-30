"""W3b report: W1's dominance rule at the hybrid's two Al rates, and DCVC against the stronger SVT envelope.

Inputs are saved summaries (W3 job 20261029T181050Z-64feb092, the W3b job, W1's scoring job
20261029T095649Z-43181442); nothing is encoded or scored here. Rule: docs/experiments/w3-neural-baseline.md, W3b."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.tennis import w3_analysis as an

HYBRID = {'lpips': {'whole': 0.2757866112979842, 'foreground': 0.34587557402445757},
          'dists': {'whole': 0.1696124367415905, 'foreground': 0.2525512419641018}}
RATES = {'primary': 22.737962222222222, 'rebilled_kps2': 22.737962222222222 - (4236 - 2701) * 8 / 9.6 / 1000}
FIELDS = {'full_lpips': 'whole_lpips', 'full_dists': 'whole_dists', 'foreground_lpips': 'foreground_lpips',
          'foreground_dists': 'foreground_dists'}


def al_points(summary: dict[str, Any], codec: str) -> list[dict[str, Any]]:
    """Al's points of one W3-format summary as flat rows (decode flags kept for DCVC)."""
    window = next(w for w in summary['windows'] if w['window'] == 'al')
    rows = []
    for p in window['dcvc' if codec == 'dcvc' else 'svt']:
        row = {'codec': codec, 'scale': p['scale'], 'qp': p['qp'], 'rate_kbps': p['kbps'],
               **{new: p['mean'][old] for old, new in FIELDS.items()}}
        if codec == 'dcvc':
            row.update(max_decode_diff=p['max_decode_diff'], accepted=p['accepted'])
        rows.append(row)
    return sorted(rows, key=lambda r: r['rate_kbps'])


def w1_plain_points(score_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """W1's plain (offset 0) half- and quarter-resolution CRF points of the scoring job."""
    rows = []
    for policy, scale in (('s2_o0', 2), ('s4_o0', 4)):
        for p in score_summary['analysis']['by_policy'][policy]:
            rows.append({'codec': 'svt-av1', 'scale': scale, 'qp': p['crf'], 'rate_kbps': p['rate_kbps'],
                         **{k: p[k] for k in FIELDS.values()}})
    return rows


def bd_rates(dcvc: list[dict[str, Any]], anchor: list[dict[str, Any]]) -> dict[str, Any]:
    test = {m: [(p['rate_kbps'], p[f]) for p in dcvc] for m, f in FIELDS.items()}
    ref = {m: [(p['rate_kbps'], p[f]) for p in anchor] for m, f in FIELDS.items()}
    return an.envelope_comparison(test, ref)


def brief(comparison: dict[str, Any]) -> dict[str, Any]:
    return {m: {'bd_rate_percent': r['bd_rate_percent'], 'status': r.get('status'),
                'quality_overlap_negative_distance': r['quality_overlap_negative_distance'],
                'rate_dominance': r['rate_dominance']} for m, r in comparison.items()}


def report(w3: dict[str, Any], w3b: dict[str, Any], w1: dict[str, Any]) -> dict[str, Any]:
    dcvc = sorted([*al_points(w3, 'dcvc'), *al_points(w3b, 'dcvc')], key=lambda r: r['rate_kbps'])
    bases = al_points(w3, 'svt')
    strong = sorted([*bases, *w1_plain_points(w1)], key=lambda r: r['rate_kbps'])
    verdicts: dict[str, Any] = {}
    for name, rate in RATES.items():
        verdicts[name] = {'rate_kbps': rate}
        for scale in sorted({p['scale'] for p in dcvc}):
            verdicts[name][f'scale{scale}'] = an.dominance([p for p in dcvc if p['scale'] == scale], rate, HYBRID)
    return {'dcvc_points': dcvc, 'strong_svt_points': strong, 'dominance': verdicts,
            'bd_rate_vs_strong_svt': brief(bd_rates(dcvc, strong)),
            'bd_rate_vs_crf63_bases': brief(bd_rates(dcvc, bases)),
            'bd_rate_w3_points_vs_strong_svt': brief(bd_rates(al_points(w3, 'dcvc'), strong))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('w3', 'w3b', 'w1'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    result = report(*(json.loads(getattr(a, k).read_text()) for k in ('w3', 'w3b', 'w1')))
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    for name, per_scale in result['dominance'].items():
        for scale, verdict in per_scale.items():
            if scale.startswith('scale'):
                print(name, scale, {m: (v['verdict'], v['whole'], v['foreground']) for m, v in verdict.items()})
    for key in ('bd_rate_vs_strong_svt', 'bd_rate_vs_crf63_bases', 'bd_rate_w3_points_vs_strong_svt'):
        print(key, {m: (v['bd_rate_percent'], v['rate_dominance']['verdict']) for m, v in result[key].items()})


if __name__ == '__main__':
    main()
