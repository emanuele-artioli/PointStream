"""Build W7 MKV anchor hulls from independently verified score rows.

No archive admission, job dispatch, hybrid bank rebilling or claim decision lives
here. Callers must first independently verify every publication and join against
the approved W7 receipts. Missing points prevent a complete study envelope.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from experiments.background import g5
from experiments.tennis import w1_roi, w7_analysis

METRICS = ('full_lpips', 'foreground_lpips', 'full_dists', 'foreground_dists')


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
    return {'primary': primary, 'historical_family_sensitivity': sensitivity, 'deduplicated': removed}


def hulls(rows: list[dict[str, Any]], stress: float = 0.0) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Return upper concave hulls in quality = negative error; stress is ±.001."""
    if stress not in (-0.001, 0.0, 0.001):
        raise ValueError('stress must be exactly -0.001, 0, or +0.001')
    out = {}
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
