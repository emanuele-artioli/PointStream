"""W7 Tier 1 analysis: the pre-registered player-gap rule over the six windows' result archives (CPU only).

Rule (docs/experiments/w7-untouched.md, "Decision rule (primary, per tier)" and "Narrowing"). Per window,
delta = (best anchor's player-region score at the hybrid's rate) - (the scale-4 hybrid's), on LPIPS and on DISTS
(lower is better, so delta > 0 means the hybrid wins). Anchors: SVT plain half (s2_o0), plain quarter (s4_o0), half with
+20 QP outside R (s2_o20) and DCVC-UF; "best" is the lowest score among the anchors resolved at the hybrid's rate
(log-rate interpolation between bracketing points only, never extrapolated; an unresolved or missing family is excluded and
listed). Rate: the hybrid's KPS2 total. The tier holds if the mean delta over the windows has a 95% two-sided t-interval
entirely above zero on LPIPS AND on DISTS. Also reported: each match as a cluster, W1's dominance outcome (A or B) per anchor
family, and the whole-frame difference (best anchor whole - hybrid whole) at equal rate. DCVC is missing for med000 by user
decision: that window's best anchor is over the SVT families only (flagged), and the pooled result without it is a sensitivity row.

    python -m experiments.tennis.w7_analysis --window al028 judge=PATH score=PATH dcvc=PATH [sha256=HEX] \\
        --window med000 judge=PATH score=PATH --out result.json
    python -m experiments.tennis.w7_analysis inputs          # job table and a command template

Inputs per window (a fleet `published.tar` or a plain JSON; a member is found by file name anywhere in the tar):
- judge  (`k6b_real_base_judge`; `quality-summary.json`, else `quality.json`): the row whose id starts with `<match>/<clip>@`;
  scale-4 hybrid player-region (`foreground_*`) and whole (`full_*`) LPIPS/DISTS and `total_kbps` (KPS2 total for W7 windows).
- score  (`w1_score`; `score-summary.json`, else `score.json`): `analysis.by_policy` points, interpolated here at the judge's
  rate with `w1_roi.policy_at_target` (cross-checked against `analysis.at_target` when its target is the same rate).
- dcvc   (`w3_neural_baseline`; `w3-summary.json`, else `summary.json`, or the W5-order rows file
  `docs/experiments/w7-specs/dcvc-points-al028-scale2.json`): accepted scale-2 points (QP, kbps, whole/fg LPIPS/DISTS),
  interpolated with `w5_dcvc.points` + `w1_roi.policy_at_target`. May be given several times (points are merged).
- `sha256=HEX` after an input verifies that file (full digest, or a prefix of at least 12 hex digits).

Accepted jobs (docs/experiments/w7-untouched.md; `full/published.tar`; full sha256 where the doc gives one, else the prefix):

  W7_INPUTS (see the dict below; `inputs` prints it)
  al028  judge 20261030T185245Z-e51a4dec (cbc26962935b3542...e75837)  score 20261030T201626Z-e32a981b (48846a4a9404a7a0...b21)
         dcvc  docs/experiments/w7-specs/dcvc-points-al028-scale2.json (from 20261030T214920Z-ea50a2cf + 20261030T221148Z-4973d2f4)
  al034  judge 20261031T110824Z-9d59bd9b (e9c32cde)  score 20261031T113142Z-4be48588 (477e8bd0)  dcvc 20261031T113203Z-cec062e0 (68ebe321)
  al043  judge 20261031T110838Z-ff9d8b4f (2f01e4df)  score 20261031T112231Z-b3a5d0d6 (cbf3c726)  dcvc 20261031T112311Z-074b29fc (dc2c33e0)
  med000 judge 20261031T110838Z-ff9d8b4f (2f01e4df)  score 20261031T112710Z-7930b798 (936e866c)  dcvc none (user decision)
  med024 judge 20261031T110824Z-9d59bd9b (e9c32cde)  score 20261031T113152Z-f858b1c5 (13ad9d70)  dcvc 20261031T115155Z-4242c23c (rerun)
  med019 judge 20261031T104813Z-6d29c907 (812c18a2aa7943c1...9d92335)  score 20261031T105818Z-014c1510 (65aec2bca1f0238b...922a17)
         dcvc 20261031T112733Z-80920718 (edca7d76)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import w1_roi as w
from experiments.tennis import w5_dcvc

WINDOWS = ('al028', 'al034', 'al043', 'med000', 'med024', 'med019')
MATCHES = {'al': 'alcaraz_hurkacz', 'med': 'medvedev_struff', 'nad': 'nadal_kyrgios'}
TIER2_WINDOWS = ('nad000', 'nad001', 'nad002')
ANCHORS = ('s2_o0', 's4_o0', 's2_o20', 'dcvc')
SVT = ('s2_o0', 's4_o0', 's2_o20')
METRICS = w.METRICS
DCVC_SCALE = 2
CONFIDENCE = 0.95

#: Accepted jobs per window. `sha256` is the full digest of `full/published.tar` where the experiments entry gives it, else None
#: with `sha256_prefix` as written there. `window ids` of the archives that cover two windows are listed under `shared_with`.
W7_INPUTS: dict[str, dict[str, Any]] = {
    'al028': {
        'judge': {'job': '20261030T185245Z-e51a4dec', 'sha256': 'cbc26962935b3542b0b7cf77f724010de19323fedd6fe02cb822ec6e43e75837'},
        'score': {'job': '20261030T201626Z-e32a981b', 'sha256': '48846a4a9404a7a0834cbe156a5d57988db114240773ba2a9657b7a08246db21'},
        'dcvc': [{'file': 'docs/experiments/w7-specs/dcvc-points-al028-scale2.json',
                  'note': 'committed scale-2 curve: QP 6 and 18 from 20261030T214920Z-ea50a2cf (sha256 66a715b22f6bcde05ce28f94d2eb586b7a215558964599302bff5105e8bab6ec), '
                          'QP 24 and 30 from 20261030T221148Z-4973d2f4 (sha256 63104b7832ab9d8f8e6125bc2acf852feffd4954ff3950ce33d08f6523270e49)'}]},
    'al034': {
        'judge': {'job': '20261031T110824Z-9d59bd9b', 'sha256': None, 'sha256_prefix': 'e9c32cde', 'shared_with': ['med024']},
        'score': {'job': '20261031T113142Z-4be48588', 'sha256': None, 'sha256_prefix': '477e8bd0'},
        'dcvc': [{'job': '20261031T113203Z-cec062e0', 'sha256': None, 'sha256_prefix': '68ebe321'}]},
    'al043': {
        'judge': {'job': '20261031T110838Z-ff9d8b4f', 'sha256': None, 'sha256_prefix': '2f01e4df', 'shared_with': ['med000']},
        'score': {'job': '20261031T112231Z-b3a5d0d6', 'sha256': None, 'sha256_prefix': 'cbf3c726'},
        'dcvc': [{'job': '20261031T112311Z-074b29fc', 'sha256': None, 'sha256_prefix': 'dc2c33e0'}]},
    'med000': {
        'judge': {'job': '20261031T110838Z-ff9d8b4f', 'sha256': None, 'sha256_prefix': '2f01e4df', 'shared_with': ['al043']},
        'score': {'job': '20261031T112710Z-7930b798', 'sha256': None, 'sha256_prefix': '936e866c'},
        'dcvc': []},  # dropped by user decision (2026-10-31): decode nondeterminism at QP 18; SVT-only best anchor
    'med024': {
        'judge': {'job': '20261031T110824Z-9d59bd9b', 'sha256': None, 'sha256_prefix': 'e9c32cde', 'shared_with': ['al034']},
        'score': {'job': '20261031T113152Z-f858b1c5', 'sha256': None, 'sha256_prefix': '13ad9d70'},
        'dcvc': [{'job': '20261031T115155Z-4242c23c', 'sha256': None, 'note': 'rerun; NOT the failed original 20261031T113212Z-c4df5a5f'}]},
    'med019': {
        'judge': {'job': '20261031T104813Z-6d29c907', 'sha256': '812c18a2aa7943c1d9ce28a4685529a5fcfebefabf8d30b441a238f4b9d92335'},
        'score': {'job': '20261031T105818Z-014c1510', 'sha256': '65aec2bca1f0238bfdcc5f42cc9c03d032dcafbaaf1923e4139fb69ec8922a17'},
        'dcvc': [{'job': '20261031T112733Z-80920718', 'sha256': None, 'sha256_prefix': 'edca7d76'}]},
}

#: Archive member names to look for, in order of preference, per input kind.
MEMBERS = {'judge': ('quality-summary.json', 'quality.json'), 'score': ('score-summary.json', 'score.json'),
           'dcvc': ('w3-summary.json', 'summary.json')}


# ---------------------------------------------------------------- reading


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str | None) -> str:
    digest = sha256_of(path)
    if expected is not None:
        expected = expected.lower()
        if len(expected) < 12 or not digest.startswith(expected):
            raise ValueError(f'sha256 mismatch for {path}: expected {expected}, found {digest}')
    return digest


def read_json(path: Path, kind: str) -> Any:
    """The JSON of a plain file, or of the member of a tar archive named like `MEMBERS[kind]` (searched at any depth)."""
    if not tarfile.is_tarfile(path):
        return json.loads(path.read_text())
    with tarfile.open(path) as tar:
        names = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for name in MEMBERS[kind]:
            if name in names:
                member = tar.extractfile(names[name])
                assert member is not None
                return json.load(member)
    raise ValueError(f'{path}: no member named {" or ".join(MEMBERS[kind])}')


def window_prefix(name: str) -> str:
    m = re.fullmatch(r'(al|med|nad)(\d{3})', name)
    if m is None:
        raise ValueError(f'unknown window name: {name}')
    return f'{MATCHES[m.group(1)]}/{m.group(2)}@'


def match_of(name: str) -> str:
    window_prefix(name)
    return next(key for key in MATCHES if name.startswith(key))


def hybrid_from_judge(quality: dict[str, Any], name: str) -> dict[str, Any]:
    """Scale-4 hybrid means and KPS2 total of the window's row in the real-base judge's summary.
    The judge orders `means` per scale, base then hybrid, in the order of `rates`, so the hybrid of rate i is entry 2i+1."""
    rows = [r for r in quality['rows'] if str(r['id']).startswith(window_prefix(name))]
    if len(rows) != 1:
        raise ValueError(f'{name}: expected one judge row with id {window_prefix(name)}*, found {len(rows)}')
    row = rows[0]
    scales = [r['scale'] for r in row['rates']]
    if 4 not in scales:
        raise ValueError(f'{name}: no scale-4 rate in the judge row')
    i = scales.index(4)
    means = row['means']
    hybrid = {m: {'whole': float(means[f'full_{m}'][2 * i + 1]), 'foreground': float(means[f'foreground_{m}'][2 * i + 1])} for m in METRICS}
    return {'id': row['id'], 'hybrid': hybrid, 'rate_kbps': float(row['rates'][i]['total_kbps'])}


def svt_policies(score: dict[str, Any], rate: float) -> dict[str, dict[str, Any] | None]:
    """Every SVT policy of the W1 score summary at `rate` (None: no bracketing pair; never extrapolated)."""
    by_policy = score['analysis']['by_policy']
    out = {name: w.policy_at_target(rows, rate) for name, rows in by_policy.items()}
    for family in SVT:
        out.setdefault(family, None)
    return out


def dcvc_rows(sources: list[Any], name: str, scale: int = DCVC_SCALE) -> tuple[list[tuple[float, ...]], list[str]]:
    """Accepted DCVC points of one window as (QP, kbps, whole LPIPS, whole DISTS, fg LPIPS, fg DISTS) rows (the w5_dcvc order).
    A W3 summary contributes its accepted points of `scale` for the window; a bare list is already in that order."""
    rows: dict[float, tuple[float, ...]] = {}
    notes: list[str] = []

    def add(row: tuple[float, ...]) -> None:
        if row[0] in rows and abs(rows[row[0]][1] - row[1]) > 1e-6:
            notes.append(f'QP {row[0]:g} appears with two rates ({rows[row[0]][1]:.3f}, {row[1]:.3f}); the first is kept')
        else:
            rows.setdefault(row[0], row)

    for source in sources:
        if isinstance(source, list):
            for r in source:
                if len(r) != 6:
                    raise ValueError('a DCVC points row has six values: QP, kbps, whole LPIPS, whole DISTS, fg LPIPS, fg DISTS')
                add(tuple(float(x) for x in r))
            continue
        entries = [x for x in source['windows'] if x.get('window') == name]
        if len(entries) != 1:
            raise ValueError(f'{name}: the DCVC summary has {len(entries)} entries for this window')
        for p in entries[0]['dcvc']:
            if p['scale'] != scale:
                continue
            if p.get('accepted') is not True:
                notes.append(f"scale {p['scale']} QP {p['qp']} not accepted: excluded")
                continue
            m = p['mean']
            add((float(p['qp']), float(p['kbps']), m['full_lpips'], m['full_dists'], m['foreground_lpips'], m['foreground_dists']))
    return sorted(rows.values(), key=lambda r: r[1]), notes


def dcvc_value(table: list[tuple[float, ...]], rate: float) -> dict[str, Any] | None:
    """The W5 interpolation (log-rate between the bracketing points), unchanged: `w5_dcvc.points` + `w1_roi.policy_at_target`."""
    return w.policy_at_target(w5_dcvc.points(table), rate)


# ---------------------------------------------------------------- statistics


def t_interval(values: list[float], confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Mean and two-sided t-interval of `values` (n - 1 degrees of freedom); bounds are None for fewer than two values."""
    from scipy import stats
    import numpy as np

    n = len(values)
    mean = float(np.mean(values)) if n else None
    if n < 2:
        return {'n': n, 'mean': mean, 'sd': None, 't': None, 'lo': None, 'hi': None, 'excludes_zero_above': None}
    sd = float(np.std(values, ddof=1))
    t = float(stats.t.ppf(0.5 + confidence / 2, n - 1))
    half = t * sd / n ** 0.5
    return {'n': n, 'mean': mean, 'sd': sd, 't': t, 'lo': mean - half, 'hi': mean + half, 'excludes_zero_above': bool(mean - half > 0)}


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {'windows': [r['window'] for r in results]}
    for m in METRICS:
        out[m] = t_interval([r['gap'][m]['delta'] for r in results])
    out['holds'] = bool(len(results) >= 2 and all(out[m]['excludes_zero_above'] for m in METRICS))
    return out


# ---------------------------------------------------------------- one window


def analyse_window(name: str, quality: dict[str, Any], score: dict[str, Any], dcvc_sources: list[Any]) -> dict[str, Any]:
    judged = hybrid_from_judge(quality, name)
    hybrid, rate = judged['hybrid'], judged['rate_kbps']
    warnings: list[str] = []
    if score.get('window') not in (None, name):
        warnings.append(f"score summary is for window {score.get('window')!r}")
    sh = score.get('hybrid', {}).get('hybrid')
    if sh is not None:
        diff = max(abs(sh[m][k] - hybrid[m][k]) for m in METRICS for k in ('whole', 'foreground'))
        rate_diff = abs(score['hybrid'].get('kps2_kbps', rate) - rate)
        if diff > 1e-9 or rate_diff > 1e-9:
            warnings.append(f'judge and score summaries disagree on the hybrid (max score diff {diff:.2e}, rate diff {rate_diff:.2e} kbps)')
    policies = svt_policies(score, rate)
    embedded = score['analysis'].get('at_target', {})
    if abs(score['analysis'].get('target_kbps', -1) - rate) < 1e-9:
        for family in SVT:
            a, b = embedded.get(family), policies[family]
            if (a is None) != (b is None) or (a is not None and b is not None and abs(a['foreground_lpips'] - b['foreground_lpips']) > 1e-9):
                warnings.append(f'recomputed {family} differs from analysis.at_target')
    else:
        warnings.append('score summary was computed at another target rate; its at_target is not used (recomputed from by_policy)')

    table, notes = dcvc_rows(dcvc_sources, name) if dcvc_sources else ([], [])
    warnings += notes
    family_status: dict[str, str] = {f: 'resolved' if policies[f] is not None else 'unresolved' for f in SVT}
    all_values: dict[str, dict[str, Any] | None] = dict(policies)
    if dcvc_sources:
        all_values['dcvc'] = dcvc_value(table, rate)
        family_status['dcvc'] = 'resolved' if all_values['dcvc'] is not None else 'unresolved'
    else:
        family_status['dcvc'] = 'missing'
    anchors: dict[str, dict[str, Any]] = {f: v for f in ANCHORS if (v := all_values.get(f)) is not None}
    excluded = {f: s for f, s in family_status.items() if s != 'resolved'}
    if not anchors:
        raise ValueError(f'{name}: no anchor family resolved at {rate:.4f} kbps')

    gap: dict[str, Any] = {}
    whole: dict[str, Any] = {}
    for m in METRICS:
        best = min(anchors, key=lambda f: anchors[f][f'foreground_{m}'])
        best_value = anchors[best][f'foreground_{m}']
        gap[m] = {'delta': best_value - hybrid[m]['foreground'], 'best_anchor': best, 'best_anchor_score': best_value,
                  'hybrid_score': hybrid[m]['foreground'], 'per_anchor': {f: v[f'foreground_{m}'] - hybrid[m]['foreground'] for f, v in anchors.items()}}
        best_whole = min(anchors, key=lambda f: anchors[f][f'whole_{m}'])
        whole[m] = {'difference': anchors[best_whole][f'whole_{m}'] - hybrid[m]['whole'], 'best_anchor': best_whole,
                    'hybrid_score': hybrid[m]['whole'], 'per_anchor': {f: v[f'whole_{m}'] - hybrid[m]['whole'] for f, v in anchors.items()}}
    resolved_all: dict[str, dict[str, Any]] = {f: v for f, v in all_values.items() if v is not None}
    svt_only: dict[str, dict[str, Any] | None] = {f: v for f, v in resolved_all.items() if f != 'dcvc'}
    decision = w.decide(dict(resolved_all), hybrid)
    return {
        'window': name, 'judge_row': judged['id'], 'rate_kbps': rate, 'hybrid': hybrid, 'gap': gap, 'whole_frame': whole,
        'anchors_used': sorted(anchors), 'anchors_excluded': excluded, 'svt_only_best_anchor': 'dcvc' not in anchors,
        'dcvc_points': [list(r) for r in table],
        'dcvc_bracket_kbps': resolved_all['dcvc']['rate_kbps'] if 'dcvc' in resolved_all else None,
        'dominance': {'outcome': decision['outcome'], 'outcome_svt_only': w.decide(svt_only, hybrid)['outcome'],
                      'dominating': decision['dominating'], 'margin': decision['margin'],
                      'per_family': {f: {m: w.dominates(v, m, hybrid) for m in METRICS} for f, v in sorted(resolved_all.items())}},
        'w1_embedded_outcome': score['analysis'].get('decision', {}).get('outcome'),
        'at_hybrid_rate': {f: {k: v[k] for k in v if k.startswith(('whole', 'foreground'))} for f, v in sorted(resolved_all.items())},
        'warnings': warnings}


def analyse(inputs: dict[str, dict[str, list[Path]]], checks: dict[Path, str | None] | None = None) -> dict[str, Any]:
    """The whole rule over the named windows. `inputs[name]` has lists under 'judge', 'score' and 'dcvc' (the last may be empty)."""
    checks = checks or {}
    cache: dict[tuple[Path, str], Any] = {}
    digests: dict[str, str] = {}

    def load(path: Path, kind: str) -> Any:
        if (path, kind) not in cache:
            digests[str(path)] = verify(path, checks.get(path))
            cache[(path, kind)] = read_json(path, kind)
        return cache[(path, kind)]

    tier2 = any(name.startswith('nad') for name in inputs)
    if tier2 and any(not name.startswith('nad') for name in inputs):
        raise ValueError('Tier 1 and Tier 2 windows cannot be pooled')
    expected = TIER2_WINDOWS if tier2 else WINDOWS
    tier = 2 if tier2 else 1
    results = []
    for name, files in inputs.items():
        judge, score = files.get('judge', []), files.get('score', [])
        if len(judge) != 1 or len(score) != 1:
            raise ValueError(f'{name}: exactly one judge and one score input are required')
        results.append(analyse_window(name, load(judge[0], 'judge'), load(score[0], 'score'), [load(p, 'dcvc') for p in files.get('dcvc', [])]))
    pooled = summarize_group(results)
    clusters = {m: summarize_group([r for r in results if match_of(r['window']) == m]) for m in MATCHES
                if any(match_of(r['window']) == m for r in results)}
    kept = [r for r in results if r['anchors_excluded'].get('dcvc') != 'missing']
    sensitivity = {'excluded': [r['window'] for r in results if r['anchors_excluded'].get('dcvc') == 'missing'], **summarize_group(kept)}
    missing = [n for n in expected if n not in inputs]
    return {'tier': tier, 'rule': f'W7 Tier {tier} player gap: mean of (best anchor player-region score - hybrid player-region score) over windows, 95% two-sided t-interval above zero on LPIPS and on DISTS',
            'rate': 'KPS2 total of the scale-4 hybrid', 'anchors': list(ANCHORS), 'dcvc_scale': DCVC_SCALE,
            'complete': not missing, 'missing_windows': missing, f'tier{tier}_holds': bool(pooled['holds'] and not missing),
            'pooled': pooled, 'clusters': clusters, 'sensitivity_without_missing_dcvc': sensitivity,
            'windows': {r['window']: r for r in results}, 'input_sha256': digests}


# ---------------------------------------------------------------- output and command line


def fmt(x: float | None, digits: int = 4) -> str:
    return 'n/a' if x is None else f'{x:+.{digits}f}'


def interval_line(label: str, g: dict[str, Any]) -> str:
    parts = [f"{m} {fmt(g[m]['mean'])} [{fmt(g[m]['lo'])}, {fmt(g[m]['hi'])}]" for m in METRICS]
    return f"{label:<24} n={g[METRICS[0]]['n']}  " + '  '.join(parts) + f"  holds={g['holds']}"


def summary_text(result: dict[str, Any]) -> str:
    tier = result.get('tier', 1)
    lines = [f'W7 Tier {tier} player gap (best anchor - hybrid, player region; > 0 favours the hybrid), rate = hybrid KPS2 total']
    for name, r in result['windows'].items():
        flag = '  [SVT-only best anchor]' if r['svt_only_best_anchor'] else ''
        excl = ('  excluded: ' + ', '.join(f'{f} ({s})' for f, s in r['anchors_excluded'].items())) if r['anchors_excluded'] else ''
        lines.append(f"  {name:<7} {r['rate_kbps']:7.3f} kbps  lpips {fmt(r['gap']['lpips']['delta'])} ({r['gap']['lpips']['best_anchor']})  "
                     f"dists {fmt(r['gap']['dists']['delta'])} ({r['gap']['dists']['best_anchor']})  "
                     f"whole {fmt(r['whole_frame']['lpips']['difference'])}/{fmt(r['whole_frame']['dists']['difference'])}  "
                     f"W1 {r['dominance']['outcome']}{flag}{excl}")
    lines.append(interval_line('pooled', result['pooled']))
    for m, g in result['clusters'].items():
        lines.append(interval_line('cluster ' + m, g))
    s = result['sensitivity_without_missing_dcvc']
    lines.append(interval_line('without ' + ','.join(s['excluded']) if s['excluded'] else 'sensitivity (none)', s))
    warns = [f"{n}: {x}" for n, r in result['windows'].items() for x in r['warnings']]
    lines += ['  warning ' + x for x in warns]
    verdict = 'HOLDS' if result[f'tier{tier}_holds'] else 'does not hold'
    lines.append(f"Tier {tier} player win {verdict}" + ('' if result['complete'] else f" (INCOMPLETE: missing {', '.join(result['missing_windows'])})"))
    return '\n'.join(lines)


COMMAND = ("python -m experiments.tennis.w7_analysis \\\n"
           "  --window al028 judge=<al028-judge>/published.tar score=<al028-score>/published.tar dcvc=docs/experiments/w7-specs/dcvc-points-al028-scale2.json \\\n"
           "  --window al034 judge=<9d59bd9b>/published.tar score=<4be48588>/published.tar dcvc=<cec062e0>/published.tar \\\n"
           "  --window al043 judge=<ff9d8b4f>/published.tar score=<b3a5d0d6>/published.tar dcvc=<074b29fc>/published.tar \\\n"
           "  --window med000 judge=<ff9d8b4f>/published.tar score=<7930b798>/published.tar \\\n"
           "  --window med024 judge=<9d59bd9b>/published.tar score=<f858b1c5>/published.tar dcvc=<4242c23c>/published.tar \\\n"
           "  --window med019 judge=<6d29c907>/published.tar score=<014c1510>/published.tar dcvc=<80920718>/published.tar \\\n"
           "  --out result.json   # add sha256=HEX after any input to verify it")


def inputs_text() -> str:
    return json.dumps(W7_INPUTS, indent=1) + '\n\nCommand template (fetch each job\'s full/published.tar):\n' + COMMAND


def parse_args(argv: list[str]) -> tuple[dict[str, dict[str, list[Path]]], dict[Path, str | None], Path | None]:
    inputs: dict[str, dict[str, list[Path]]] = {}
    checks: dict[Path, str | None] = {}
    out: Path | None = None
    current: dict[str, list[Path]] | None = None
    last: Path | None = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ('--window', '--out'):
            if i + 1 >= len(argv):
                raise SystemExit(f'{token} needs a value')
            value = argv[i + 1]
            i += 2
            if token == '--out':
                out = Path(value)
            else:
                if value in inputs:
                    raise SystemExit(f'window {value} given twice')
                try:
                    window_prefix(value)
                except ValueError as error:
                    raise SystemExit(str(error))
                current = inputs[value] = {'judge': [], 'score': [], 'dcvc': []}
            continue
        key, sep, value = token.partition('=')
        if not sep or current is None:
            raise SystemExit(f'unexpected argument {token!r} (give --window NAME first, then judge=, score=, dcvc=, sha256=)')
        if key == 'sha256':
            if last is None:
                raise SystemExit('sha256= must follow an input')
            checks[last] = value
        elif key in current:
            last = Path(value)
            current[key].append(last)
            checks.setdefault(last, None)
        else:
            raise SystemExit(f'unknown input {key!r}; use judge=, score=, dcvc= or sha256=')
        i += 1
    return inputs, checks, out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print((__doc__ or '') + '\nCommand template:\n' + COMMAND)
        return
    if argv[0] == 'inputs':
        print(inputs_text())
        return
    inputs, checks, out = parse_args(argv)
    if not inputs:
        raise SystemExit('give at least one --window')
    result = analyse(inputs, checks)
    print(summary_text(result))
    if out is not None:
        out.write_text(json.dumps(result, indent=1))
        print('wrote', out)


if __name__ == '__main__':
    main()
