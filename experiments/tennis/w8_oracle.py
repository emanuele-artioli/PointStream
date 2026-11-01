"""Pure W8 accounting and no-extrapolation quality screen, with no file discovery."""
from __future__ import annotations

import math
from typing import Any

FAMILIES = ('s2_o0', 's4_o0', 's2_o20', 'dcvc')
METRICS = ('lpips', 'dists')


def rates(accounting: dict[str, int], duration: float = 4.8, horizon: float = 7200.0,
          *, cache_contract: dict[str, Any] | None = None, expected_cache: dict[str, Any] | None = None) -> dict[str, float]:
    if set(accounting) != {'frame', 'window', 'match', 'reference'} or any(type(v) is not int or v < 0 for v in accounting.values()):
        raise ValueError('invalid accounting')
    if not math.isfinite(duration) or not math.isfinite(horizon) or duration <= 0 or horizon <= 0:
        raise ValueError('invalid duration')
    result = {'cold_kbps': sum(accounting.values()) * 8 / duration / 1000}
    if cache_contract is not None or expected_cache is not None:
        if cache_contract is None or expected_cache is None or cache_contract != expected_cache:
            raise ValueError('conditional reuse requires exact object identities')
        if set(cache_contract) != {'match_sha256', 'reference_sha256_by_unit'} or len(cache_contract['reference_sha256_by_unit']) != 2:
            raise ValueError('incomplete cache contract')
        result['conditional_kbps'] = ((accounting['frame'] + accounting['window']) / duration + (accounting['reference'] + accounting['match']) / horizon) * 8 / 1000
        result['warm_kbps'] = (accounting['frame'] + accounting['window']) * 8 / duration / 1000
    return result


def gate(rate: float, hybrid: dict[str, float], anchors: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Strict player win versus all four raw families, interpolating only within brackets."""
    if not math.isfinite(rate) or rate <= 0 or set(anchors) != set(FAMILIES) or set(hybrid) != set(METRICS):
        raise ValueError('invalid rate or family/metric coverage')
    cells: dict[str, Any] = {}
    for family in FAMILIES:
        points = sorted(anchors[family], key=lambda p: p['kbps'])
        if len(points) < 2 or any(set(p) != {'kbps', *METRICS} or any(not math.isfinite(p[k]) for k in p) or p['kbps'] <= 0 for p in points):
            raise ValueError('invalid raw anchor points')
        if any(a['kbps'] >= b['kbps'] for a, b in zip(points, points[1:])):
            raise ValueError('duplicate anchor rate')
        bracket = [points[0]['kbps'], points[-1]['kbps']]
        for metric in METRICS:
            value = hybrid[metric]
            if not math.isfinite(value):
                raise ValueError('nonfinite hybrid')
            result: dict[str, Any] = {'bracket_kbps': bracket, 'bracket_headroom_kbps': bracket[1] - rate}
            if rate < bracket[0] or rate > bracket[1]:
                result.update({'won': None, 'gap': None})
            else:
                left, right = next((a, b) for a, b in zip(points, points[1:]) if a['kbps'] <= rate <= b['kbps'])
                f = math.log(rate / left['kbps']) / math.log(right['kbps'] / left['kbps'])
                score = left[metric] + f * (right[metric] - left[metric])
                result.update({'won': value < score, 'gap': score - value, 'interpolated_error': score})
            cells[f'{family}:{metric}'] = result
    outcomes = [c['won'] for c in cells.values()]
    return {'decision': 'unresolved' if None in outcomes else 'pass' if all(outcomes) else 'fail', 'cells': cells}


def accepted_nad000(inbox: Any, repair: dict[str, str]) -> tuple[dict[str, float], dict[str, list[dict[str, float]]]]:
    """Load hash-verified raw judge, accepted DCVC and independently audited W9 repair."""
    import json

    from experiments.tennis import w7_analysis, w9_repair_intervals, w9_saved_intervals
    from experiments.tennis.w7_billing_corrected import MANIFEST, REPOSITORY, TIER2

    hashes = json.loads(MANIFEST.read_text())['results']
    judge_job, _, dcvc_job = TIER2['nad000']
    judge = inbox / judge_job / 'full/published.tar'
    dcvc = inbox / dcvc_job / 'full/published.tar'
    w7_analysis.verify(judge, hashes[judge_job])
    w7_analysis.verify(dcvc, hashes[dcvc_job])
    result_path = REPOSITORY / 'docs/experiments/w9-inputs/w7-tier2-billing-corrected.json'
    w7_analysis.verify(result_path, w9_saved_intervals.EXPECTED_RESULTS[1])
    rows = json.loads(result_path.read_text())['windows']
    corrected = next(row for row in rows if row['window'] == 'nad000')
    merged = w9_repair_intervals.verified_merge('nad000', repair, inbox, hashes, corrected['rate_kbps'])
    quality = w7_analysis.hybrid_from_judge(w7_analysis.read_json(judge, 'judge'), 'nad000')['hybrid']
    dcvc_rows, warnings = w7_analysis.dcvc_rows([w7_analysis.read_json(dcvc, 'dcvc')], 'nad000')
    if warnings:
        raise ValueError(f'ambiguous DCVC points: {warnings}')
    anchors = {family: [{'kbps': float(p['rate_kbps']), **{m: float(p[f'foreground_{m}']) for m in METRICS}}
                        for p in merged['by_policy'][family]] for family in FAMILIES if family != 'dcvc'}
    anchors['dcvc'] = [{'kbps': row[1], 'lpips': row[4], 'dists': row[5]} for row in dcvc_rows]
    return {m: float(quality[m]['foreground']) for m in METRICS}, anchors


def strict_win_intervals(hybrid: dict[str, float], anchors: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Exact piecewise-log candidate-rate intervals; boundaries are not declared wins."""
    if set(anchors) != set(FAMILIES):
        raise ValueError('all families required')
    # gate validates all values and duplicate-rate ambiguity.
    bounds = [(min(p['kbps'] for p in anchors[f]), max(p['kbps'] for p in anchors[f])) for f in FAMILIES]
    low, high = max(x[0] for x in bounds), min(x[1] for x in bounds)
    if low >= high:
        return {'intervals_kbps': [], 'maximum_win_rate_supremum_kbps': None, 'common_bracket_kbps': [low, high]}
    cuts = {low, high}
    for family in FAMILIES:
        points = sorted(anchors[family], key=lambda p: p['kbps'])
        cuts.update(p['kbps'] for p in points if low < p['kbps'] < high)
        for a, b in zip(points, points[1:]):
            for metric in METRICS:
                y0, y1, target = a[metric], b[metric], hybrid[metric]
                if y0 != y1:
                    fraction = (target - y0) / (y1 - y0)
                    if 0 < fraction < 1:
                        x = math.exp(math.log(a['kbps']) + fraction * math.log(b['kbps'] / a['kbps']))
                        if low < x < high:
                            cuts.add(x)
    ordered = sorted(cuts)
    wins: list[list[float]] = []
    for lo, hi in zip(ordered, ordered[1:]):
        if gate(math.sqrt(lo * hi), hybrid, anchors)['decision'] == 'pass':
            if wins and wins[-1][1] == lo and gate(lo, hybrid, anchors)['decision'] == 'pass':
                wins[-1][1] = hi
            else:
                wins.append([lo, hi])
    return {'intervals_kbps': wins, 'maximum_win_rate_supremum_kbps': max((x[1] for x in wins), default=None),
            'common_bracket_kbps': [low, high], 'note': 'interval endpoints require separate strict gate evaluation'}
