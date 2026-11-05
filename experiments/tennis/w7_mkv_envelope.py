"""Build W7 MKV anchor hulls from independently verified score rows.

No archive admission, job dispatch, hybrid bank rebilling or claim decision lives
here. Callers must first independently verify every publication and join against
the approved W7 receipts. Missing points prevent a complete study envelope.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5
from experiments.tennis import w1_roi, w7_analysis

METRICS = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists')


OLD_REVIEW_SHA = '581b015e88bbb5d5ce6a5900cf8e3ac8ecc32d649d7f9887ee29893b8d19a18b'
CONTROL_REVIEW_SHA = '7a6ff1d5ce5d540b5d31825147cc5c198c66149cdf11741fb56786cfd186d7b3'
NEW_REVIEW_SHA = 'a01fb9ff905918e8574e48ce6d74d4b0f44f9ce9099a46dd74f85a0d2c8726f1'
INVENTORY_SHA = '867d9fff8974f4ec5320c9b830138d4aac2ee22c285d7de45aebb04dff3bab11'
DELIVERY_MAP_SHA = 'fd53ad6f70b589e1849cd94ede731c1259867b48d0e2b80cf64fcd52dbc5377e'


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_historical(old_path: Path, control_path: Path, archives: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate the two root-reviewed receipts and all nine old score archives."""
    if _file_sha(old_path) != OLD_REVIEW_SHA or _file_sha(control_path) != CONTROL_REVIEW_SHA:
        raise ValueError('historical review receipt SHA mismatch')
    old, controls = json.loads(old_path.read_text()), json.loads(control_path.read_text())
    if old.get('approved_historical_reuse') is not True or controls.get('approved_historical_control_reuse') is not True:
        raise ValueError('historical reuse not approved')
    jobs = old['historical_jobs']
    if len(jobs) != 9 or set(archives) != set(jobs):
        raise ValueError('exactly nine historical producer archives required')
    loaded = {}
    for job, path in archives.items():
        if _file_sha(path) != jobs[job]['score_archive_sha256']:
            raise ValueError('historical score archive SHA mismatch: ' + job)
        with tarfile.open(path) as tar:
            members = [m for m in tar if m.name == 'publish/scores.npz' and m.isfile()]
            if len(members) != 1 or members[0].size > 64_000_000:
                raise ValueError('historical scores member missing or ambiguous')
            stream = tar.extractfile(members[0])
            if stream is None:
                raise ValueError('historical scores member unreadable')
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as scores:
                if scores['ticks'].tolist() != w1_roi.TICKS:
                    raise ValueError('historical tick set differs')
                loaded[job] = {name: scores[name].copy() for name in scores.files if name != 'ticks'}
    rows = []
    for role, receipt in (('rerate', old), ('control', controls)):
        for record in receipt['rows']:
            score = loaded[record['score_job']][record['name']]
            if score_digest(score) != record['score_array_sha256']:
                raise ValueError('historical score array SHA mismatch')
            rows.append({'role': role, 'window': record['window'], 'name': record['name'],
                         'original_ivf_sha256': record.get('original_ivf_sha256', record.get('stream_sha256')),
                         'native_yuv_sha256': record.get('native_yuv_sha256', record.get('historical_native_yuv_sha256')),
                         'mkv_sha256': record['mkv_sha256'], 'mkv_bytes': record['mkv_bytes'],
                         'fps': record['fps'], 'values': score, 'gpu_class': record['gpu_class'],
                         'score_job': record['score_job']})
    if len(rows) != 71:
        raise ValueError('historical coverage differs')
    return rows, old, controls



def load_new(review_path: Path, inventory_path: Path, delivery_path: Path,
             archives: dict[tuple[str, str], Path]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate both actual campaigns and four new per-tick score arrays."""
    if (_file_sha(review_path) != NEW_REVIEW_SHA or _file_sha(inventory_path) != INVENTORY_SHA
            or _file_sha(delivery_path) != DELIVERY_MAP_SHA):
        raise ValueError('new-score audit, inventory or delivery map SHA differs')
    audit, inventory, delivery = (json.loads(p.read_text()) for p in (review_path, inventory_path, delivery_path))
    if audit.get('passed') is not True or audit.get('ticks') != w1_roi.TICKS:
        raise ValueError('actual four-score audit and full ticks required')
    selected = [r for r in audit['rows'] if r['kind'] == 'new']
    if len(selected) != 4 or len({(r['window'], r['name']) for r in selected}) != 4:
        raise ValueError('exactly four audited new scores required')
    expected = {(r['window'], r['name']): r for r in inventory['new_points']}
    mapped = {(r['window'], r['name']): r for r in delivery['rows'] if r['role'] == 'new_score'}
    if set(expected) != set(mapped) or set(expected) != {(r['window'], r['name']) for r in selected}:
        raise ValueError('new-score role coverage differs')
    if set(archives) != {(r['job_id'], r['stage']) for r in audit['rows']}:
        raise ValueError('exact campaign publications required')
    output, reviewed = [], []
    for record in audit['rows']:
        if record['kind'] == 'control':
            archive = archives[record['job_id'], record['stage']]
            if _file_sha(archive) != record['score_archive_sha256'] or record.get('historical_control_max_abs_diff') != 0:
                raise ValueError('new control publication or parity differs')
            with tarfile.open(archive) as tar:
                members = [m for m in tar if m.name == record['score_member'] and m.isfile()]
                if len(members) != 1 or members[0].size > 4_000_000:
                    raise ValueError('control score member absent or ambiguous')
                stream = tar.extractfile(members[0])
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != record['score_sha256']:
                    raise ValueError('control score member SHA differs')
            continue
        if record['kind'] != 'new':
            raise ValueError('unexpected score role in audit')
        key = record['window'], record['name']
        source, delivered = expected[key], mapped[key]
        archive = archives[record['job_id'], record['stage']]
        if (_file_sha(archive) != record['score_archive_sha256']
                or record['original_ivf_sha256'] != source['original_ivf_sha256']
                or record['native_yuv_sha256'] != source['historical_native_yuv_sha256']
                or record['original_ivf_sha256'] != delivered['original_ivf_sha256']
                or record['native_yuv_sha256'] != delivered['historical_native_yuv_sha256']
                or record['mkv_sha256'] != delivered['delivered_mkv_sha256']
                or record['mkv_bytes'] != delivered['delivered_mkv_bytes']
                or record['fps'] != delivered['fps']):
            raise ValueError('new score publication or delivered stream identity differs')
        with tarfile.open(archive) as tar:
            matches = [m for m in tar if m.name == record['score_member'] and m.isfile()]
            if len(matches) != 1 or matches[0].size > 4_000_000:
                raise ValueError('new score member missing or ambiguous')
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise ValueError('new score member unreadable')
            blob = stream.read()
        if hashlib.sha256(blob).hexdigest() != record['score_sha256']:
            raise ValueError('new score member SHA differs')
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            if data['ticks'].tolist() != w1_roi.TICKS:
                raise ValueError('new score ticks differ')
            values = data['values'].copy()
        row = {'role': 'new', 'window': key[0], 'name': key[1],
               'original_ivf_sha256': record['original_ivf_sha256'], 'native_yuv_sha256': record['native_yuv_sha256'],
               'mkv_sha256': record['mkv_sha256'], 'mkv_bytes': record['mkv_bytes'], 'fps': record['fps'],
               'values': values, 'gpu_class': 'A6000', 'score_job': record['job_id']}
        reviewed.append({k: v for k, v in row.items() if k not in ('role', 'values') } | {'score_array_sha256': score_digest(values)})
        output.append(row)
    return output, {'approved_new_scores': True, 'rows': reviewed, 'audit_sha256': NEW_REVIEW_SHA}, inventory


def score_digest(values: Any) -> str:
    array = np.asarray(values)
    if array.shape != (4, 80) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError('each score must be a finite 4-by-80 floating array')
    return hashlib.sha256(array.tobytes()).hexdigest()


def verified_views(inventory: dict[str, Any], rows: list[dict[str, Any]],
                   old_review: dict[str, Any], control_review: dict[str, Any],
                   new_review: dict[str, Any]) -> dict[str, Any]:
    """Join admitted historical reviews and the future independent four-point audit.

    The caller must authenticate the review files and their SHA-bound publication
    provenance before calling. The new review must include actual accepted hashes;
    a list of predicted points is insufficient.
    """
    if old_review.get('approved_historical_reuse') is not True or control_review.get('approved_historical_control_reuse') is not True:
        raise ValueError('both historical reuse reviews are required')
    if new_review.get('approved_new_scores') is not True:
        raise ValueError('independent four-point score audit is not accepted')
    groups = [('rerate', inventory['points']), ('control', inventory['controls']), ('new', inventory['new_points'])]
    if [len(group) for _, group in groups] != [53, 18, 4] or len(rows) != 75:
        raise ValueError('all 75 declared roles are required')
    expected = {(role, r['window'], r['name']): r for role, group in groups for r in group}
    if len(expected) != 75:
        raise ValueError('inventory role keys must be unique')
    reviews = {'rerate': old_review['rows'], 'control': control_review['rows'], 'new': new_review['rows']}
    reviewed = {role: {(r['window'], r['name']): r for r in rs} for role, rs in reviews.items()}
    if [len(reviewed[role]) for role in ('rerate', 'control', 'new')] != [53, 18, 4]:
        raise ValueError('review coverage differs')
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = row['role'], row['window'], row['name']
        if key in seen or key not in expected:
            raise ValueError('unknown or duplicate role')
        seen.add(key)
        item = expected[key]
        review = reviewed[key[0]].get((key[1], key[2]))
        if review is None:
            raise ValueError('unreviewed role')
        if (row['original_ivf_sha256'] != item['original_ivf_sha256']
                or row['native_yuv_sha256'] != item['historical_native_yuv_sha256']
                or row['original_ivf_sha256'] != review.get('original_ivf_sha256', review.get('stream_sha256'))
                or row['native_yuv_sha256'] != review.get('native_yuv_sha256', review.get('historical_native_yuv_sha256'))
                or row['mkv_sha256'] != review['mkv_sha256']
                or row['mkv_bytes'] != review['mkv_bytes']
                or row['fps'] != review['fps']
                or score_digest(row['values']) != review['score_array_sha256']):
            raise ValueError('score, original stream or delivery differs from review')
        if type(row['mkv_bytes']) is not int or row['mkv_bytes'] <= 0 or row['fps'] not in (25, 50):
            raise ValueError('invalid delivered bytes or cadence')
        out = dict(row)
        out['policy'] = item['policy']
        out['crf'] = item['crf']
        out['wire_kbps'] = 8 * row['mkv_bytes'] * row['fps'] / (240 * 1000)
        result.append(out)
    if seen != set(expected):
        raise ValueError('incomplete role inventory')
    duplicate_pairs = {(r['window'], r['control'], r['roi']) for r in control_review['duplicates']}
    if len(duplicate_pairs) != 4:
        raise ValueError('exact four audited duplicate pairs required')
    by_role = {(r['role'], r['window'], r['name']): r for r in result}
    removed = []
    for window, control, roi in sorted(duplicate_pairs):
        c, p = by_role['control', window, control], by_role['rerate', window, roi]
        if (c['original_ivf_sha256'] != p['original_ivf_sha256'] or c['native_yuv_sha256'] != p['native_yuv_sha256']
                or c['mkv_bytes'] != p['mkv_bytes'] or c['fps'] != p['fps']
                or not np.array_equal(c['values'], p['values'])):
            raise ValueError('audited duplicate no longer equal')
        removed.append({'window': window, 'control': control, 'roi': roi,
                        'control_mkv_sha256': c['mkv_sha256'], 'roi_mkv_sha256': p['mkv_sha256']})
    dropped = {(x['window'], x['control']) for x in removed}
    primary = [r for r in result if r['role'] != 'control' or (r['window'], r['name']) not in dropped]
    sensitivity = [r for r in result if r['role'] != 'control']
    if (len(primary), len(sensitivity)) != (71, 57):
        raise ValueError('unexpected deduplicated coverage')
    return {'primary': primary, 'historical_family_sensitivity': sensitivity, 'all_roles': result, 'deduplicated': removed}


def hulls(rows: list[dict[str, Any]], stress: float = 0.0) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Return upper concave hulls in quality = negative error; stress is ±.001."""
    if stress not in (-0.001, 0.0, 0.001):
        raise ValueError('stress must be exactly -0.001, 0, or +0.001')
    out: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for window in w7_analysis.WINDOWS:
        subset = [r for r in rows if r['window'] == window]
        if not subset:
            raise ValueError('missing window')
        out[window] = {}
        for index, name in enumerate(METRICS):
            points = [(r['wire_kbps'], -float(np.mean(np.asarray(r['values'])[index]) + stress)) for r in subset]
            if any(not math.isfinite(rate) or rate <= 0 or not math.isfinite(q) for rate, q in points):
                raise ValueError('invalid rate or quality')
            out[window][name] = g5.envelope(points)
    return out


def pooled_interval(values: list[float]) -> dict[str, Any]:
    """Use the unchanged W7 window-level 95% t-interval."""
    return w7_analysis.t_interval(values)


def policy_at_target(rows: list[dict[str, Any]], rate: float) -> dict[str, Any] | None:
    """Use W1's original bracket-only log-rate interpolation."""
    return w1_roi.policy_at_target(rows, rate)


def compare_curves(test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> dict[str, Any]:
    """PCHIP log-rate over real quality overlap; do not invent a BD value."""
    from experiments.tennis import k5

    t, a = g5.pareto(test), g5.envelope(anchor)
    if len(t) < 2 or len(a) < 2 or min(t[-1][1], a[-1][1]) <= max(t[0][1], a[0][1]):
        return {'bd_rate': None, 'basis': 'no overlapping quality range'}
    return {'bd_rate': k5.bd_rate(test, anchor, overlap=1e-12), 'basis': 'PCHIP on overlapping quality range'}


BILLING_SHA = '62376a4f3dcbe406d2065e54c12ebcbc0d2579aeb97c243cbf843bdf15a66e1f'
ADAPTER_BYTES = {'al': 1_488_252, 'med': 1_495_592}


def load_billing(path: Path) -> dict[str, dict[str, Any]]:
    """Load the previously audited corrected two-hour W7 hybrid/DCVC record."""
    if _file_sha(path) != BILLING_SHA:
        raise ValueError('corrected W7 billing artifact SHA differs')
    data = json.loads(path.read_text())
    result = {r['window']: r for r in data['windows']}
    if set(result) != set(w7_analysis.WINDOWS) or len(data['windows']) != 6:
        raise ValueError('corrected W7 window coverage differs')
    return result


def bank_rate(row: dict[str, Any], window: str, horizon: int) -> float:
    """W9's all-assets amortisation, starting from the corrected 7200-s rate."""
    if horizon not in (7200, 3600, 1800):
        raise ValueError('unsupported fixed horizon')
    prefix = next((p for p in ADAPTER_BYTES if window.startswith(p)), None)
    if prefix is None:
        raise ValueError('unknown window')
    amount = ADAPTER_BYTES[prefix] + row['billing_correction']['intended_bank_bytes']
    rate = float(row['rate_kbps']) + 8 * amount / 1000 * (1 / horizon - 1 / 7200)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('invalid corrected rate')
    return rate


def _in_bracket_hull(points: list[tuple[float, float]], rate: float) -> float | None:
    front = g5.envelope(points)
    if len(front) < 2 or rate < front[0][0] or rate > front[-1][0]:
        return None
    return g5.anchor_quality(points, rate)


def decision_report(views: dict[str, Any], billing: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """W7 gaps and t intervals at corrected bank rates, across fixed stress cases.

    A missing bracket leaves the corresponding window unresolved. The selected
    families are preserved as a historical sensitivity, with DCVC as before.
    """
    output: dict[str, Any] = {'scope': 'W7 MKV sensitivity, not fresh historical neural scoring',
                              'deduplicated': views['deduplicated'], 'scenarios': {},
                              'roles': [{k: r[k] for k in ('role', 'window', 'name', 'policy', 'crf',
                                                           'original_ivf_sha256', 'native_yuv_sha256',
                                                           'mkv_sha256', 'mkv_bytes', 'fps', 'wire_kbps',
                                                           'gpu_class', 'score_job')} |
                                        {'score_array_sha256': score_digest(r['values']),
                                         'means': [float(x) for x in np.mean(r['values'], axis=1)]}
                                        for r in views['all_roles']]}
    for horizon in (7200, 3600, 1800):
        for stress in (-0.001, 0.0, 0.001):
            for view_name in ('primary', 'historical_family_sensitivity'):
                by_window: dict[str, Any] = {}
                for name in w7_analysis.WINDOWS:
                    base = billing[name]
                    rate = bank_rate(base, name, horizon)
                    subset = [r for r in views[view_name] if r['window'] == name]
                    metrics: dict[str, Any] = {}
                    for metric in ('lpips', 'dists'):
                        index = METRICS.index('foreground_' + metric)
                        candidates = []
                        if view_name == 'primary':
                            points = [(r['wire_kbps'], -float(np.mean(r['values'][index]) + stress)) for r in subset]
                            quality = _in_bracket_hull(points, rate)
                            if quality is not None:
                                candidates.append(('svt_achievable_hull', -quality))
                        else:
                            for family in w7_analysis.SVT:
                                family_rows = [{'rate_kbps': r['wire_kbps'], 'crf': r['crf'],
                                                'foreground_lpips': float(np.mean(r['values'][1])) + stress,
                                                'foreground_dists': float(np.mean(r['values'][3])) + stress,
                                                'whole_lpips': float(np.mean(r['values'][0])) + stress,
                                                'whole_dists': float(np.mean(r['values'][2])) + stress}
                                               for r in subset if r['policy'] == family]
                                value = w1_roi.policy_at_target(family_rows, rate) if family_rows else None
                                if value is not None:
                                    candidates.append((family, value['foreground_' + metric]))
                        dcvc = w7_analysis.dcvc_value([tuple(x) for x in base['dcvc_points']], rate) if base['dcvc_points'] else None
                        if dcvc is not None:
                            candidates.append(('dcvc', dcvc['foreground_' + metric] + stress))
                        if candidates:
                            family, score = min(candidates, key=lambda x: x[1])
                            metrics[metric] = {'delta': score - base['hybrid'][metric]['foreground'], 'best_anchor': family}
                        else:
                            metrics[metric] = None
                    whole: dict[str, Any] = {}
                    for metric in ('lpips', 'dists'):
                        index = METRICS.index('full_' + metric)
                        candidates_whole = []
                        if view_name == 'primary':
                            points = [(r['wire_kbps'], -float(np.mean(r['values'][index]) + stress)) for r in subset]
                            quality = _in_bracket_hull(points, rate)
                            if quality is not None:
                                candidates_whole.append(('svt_achievable_hull', -quality))
                        else:
                            for family in w7_analysis.SVT:
                                family_rows = [{'rate_kbps': r['wire_kbps'], 'crf': r['crf'],
                                                'foreground_lpips': float(np.mean(r['values'][1])) + stress,
                                                'foreground_dists': float(np.mean(r['values'][3])) + stress,
                                                'whole_lpips': float(np.mean(r['values'][0])) + stress,
                                                'whole_dists': float(np.mean(r['values'][2])) + stress}
                                               for r in subset if r['policy'] == family]
                                value = w1_roi.policy_at_target(family_rows, rate) if family_rows else None
                                if value is not None:
                                    candidates_whole.append((family, value['whole_' + metric]))
                        if dcvc is not None:
                            candidates_whole.append(('dcvc', dcvc['whole_' + metric] + stress))
                        if candidates_whole:
                            family, score = min(candidates_whole, key=lambda x: x[1])
                            whole[metric] = {'difference': score - base['hybrid'][metric]['whole'], 'best_anchor': family}
                        else:
                            whole[metric] = None
                    family_status = None
                    dominance = None
                    if view_name == 'historical_family_sensitivity':
                        policies = {}
                        for family in w7_analysis.SVT:
                            family_rows = [{'rate_kbps': r['wire_kbps'], 'crf': r['crf'],
                                            'foreground_lpips': float(np.mean(r['values'][1])) + stress,
                                            'foreground_dists': float(np.mean(r['values'][3])) + stress,
                                            'whole_lpips': float(np.mean(r['values'][0])) + stress,
                                            'whole_dists': float(np.mean(r['values'][2])) + stress}
                                           for r in subset if r['policy'] == family]
                            policies[family] = w1_roi.policy_at_target(family_rows, rate) if family_rows else None
                        if dcvc is not None:
                            policies['dcvc'] = {k: v + stress if k.startswith(('whole_', 'foreground_')) else v
                                                for k, v in dcvc.items()}
                        family_status = {family: 'resolved' if value is not None else 'unresolved'
                                         for family, value in policies.items()}
                        if not base['dcvc_points']:
                            family_status['dcvc'] = 'missing'
                        dominance = w1_roi.decide(policies, base['hybrid']) if any(v is not None for v in policies.values()) else None
                    by_window[name] = {'rate_kbps': rate, 'metrics': metrics, 'whole_frame': whole,
                                       'dcvc_missing': not bool(base['dcvc_points']),
                                       'family_status': family_status, 'w1_dominance': dominance}
                groups: dict[str, Any] = {}
                for metric in ('lpips', 'dists'):
                    present = [r['metrics'][metric]['delta'] for r in by_window.values() if r['metrics'][metric] is not None]
                    groups[metric] = {'complete': len(present) == 6,
                                      'interval': w7_analysis.t_interval(present) if len(present) == 6 else None}
                key = f'{horizon}:{stress:+.3f}:{view_name}'
                clusters = {}
                for prefix in ('al', 'med'):
                    cluster = [r for n, r in by_window.items() if n.startswith(prefix)]
                    clusters[prefix] = {m: w7_analysis.t_interval([r['metrics'][m]['delta'] for r in cluster])
                                        if all(r['metrics'][m] is not None for r in cluster) else None
                                        for m in ('lpips', 'dists')}
                eligible = [r for r in by_window.values() if not r['dcvc_missing']]
                without_missing = {m: w7_analysis.t_interval([r['metrics'][m]['delta'] for r in eligible])
                                   if all(r['metrics'][m] is not None for r in eligible) else None
                                   for m in ('lpips', 'dists')}
                output['scenarios'][key] = {'windows': by_window, 'pooled': groups, 'clusters': clusters,
                                             'sensitivity_without_missing_dcvc': without_missing,
                                             'holds': all(groups[m]['complete'] and groups[m]['interval']['excludes_zero_above']
                                                          for m in ('lpips', 'dists')) if all(groups[m]['complete'] for m in groups) else None}
    return output


def main() -> None:
    import argparse
    import copy
    import resource
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pilot', 'final'), required=True)
    parser.add_argument('--historical-inbox', type=Path, required=True)
    parser.add_argument('--pilot-publication', type=Path, required=True)
    parser.add_argument('--new-smoke-publication', type=Path, required=True)
    parser.add_argument('--new-full-publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--gate-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root / 'docs/experiments/w7-inputs'
    started = time.monotonic()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    historical_jobs = json.loads((evidence / 'historical-53-reuse-review-20261105.json').read_text())['historical_jobs']
    archives = {job: args.historical_inbox / job / 'full/published.tar' for job in historical_jobs}
    old, old_review, control_review = load_historical(evidence / 'historical-53-reuse-review-20261105.json',
                                                       evidence / 'historical-18-controls-fairness-review-20261105.json', archives)
    pubs = {('20261105T055056Z-aa3070f0', 'smoke'): args.pilot_publication,
            ('20261105T064410Z-40ee5e97', 'smoke'): args.new_smoke_publication,
            ('20261105T064410Z-40ee5e97', 'full'): args.new_full_publication}
    new, new_review, inventory = load_new(evidence / 'narrow-five-40ee5e97-score-receipt-20261105.json',
                                           evidence / 'mkv-53-inventory-20261105.json',
                                           evidence / 'fps-native-yuv-14f7a33e-delivery-map-20261105.json', pubs)
    views = verified_views(inventory, old + new, old_review, control_review, new_review)
    billing = load_billing(root / 'docs/experiments/w9-inputs/w7-tier1-billing-corrected.json')
    loaded_wall = time.monotonic() - started
    if args.mode == 'pilot':
        fake = copy.deepcopy(views)
        for i, row in enumerate(fake['all_roles']):
            row['values'] = np.full((4, 80), .1 + (i % 7) * .01)
        fake_billing = {name: {'rate_kbps': 20.0, 'billing_correction': {'intended_bank_bytes': 1_000_000},
                               'dcvc_points': [], 'hybrid': {m: {'whole': .5, 'foreground': .5}
                                                            for m in ('lpips', 'dists')}}
                        for name in w7_analysis.WINDOWS}
        durations = []
        for _ in range(10):
            before = time.monotonic()
            decision_report(fake, fake_billing)
            durations.append(time.monotonic() - before)
        after = resource.getrusage(resource.RUSAGE_SELF)
        output = {'scope': 'CPU cost only; synthetic scores, no outcome or admission',
                  'historical_roles': len(old), 'new_roles': len(new), 'primary_streams': len(views['primary']),
                  'authenticated_load_wall_seconds': loaded_wall, 'synthetic_runs': len(durations),
                  'synthetic_run_wall_seconds': durations, 'total_wall_seconds': time.monotonic() - started,
                  'process_cpu_seconds': after.ru_utime + after.ru_stime - cpu.ru_utime - cpu.ru_stime,
                  'full_analysis_basis': 'one complete W7 t-interval report, no bootstrap; external work and reserves require root',
                  'source_sha256': _file_sha(Path(__file__))}
    else:
        if args.gate is None or args.gate_sha256 is None or _file_sha(args.gate) != args.gate_sha256:
            raise ValueError('SHA-pinned root cost gate required for final report')
        gate = json.loads(args.gate.read_text())
        if (gate.get('approved') is not True or gate.get('analysis_source_sha256') != _file_sha(Path(__file__))
                or gate.get('new_score_audit_sha256') != NEW_REVIEW_SHA
                or gate.get('max_cpu_seconds', 0) <= 0 or gate.get('max_cpu_seconds', 0) > 500):
            raise ValueError('root cost gate does not approve this source and evidence')
        resource.setrlimit(resource.RLIMIT_CPU, (int(gate['max_cpu_seconds']), int(gate['max_cpu_seconds'])))
        output = decision_report(views, billing)
        output['evidence'] = {'historical_review_sha256': OLD_REVIEW_SHA,
                              'control_review_sha256': CONTROL_REVIEW_SHA,
                              'new_score_audit_sha256': NEW_REVIEW_SHA, 'billing_sha256': BILLING_SHA,
                              'analysis_source_sha256': _file_sha(Path(__file__))}
        output['limitations'] = ['Historical RGB is inferred, not independently matched.',
                                 'Ada and A6000 scoring differ; ±0.001 is a stress, not an uncertainty bound.',
                                 'Reference-bank reuse and missing operational framing remain assumptions.']
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
