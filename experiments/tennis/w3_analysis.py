"""Pure helpers of W3 (docs/experiments/w3-neural-baseline.md): point selection, decode
exactness and envelope comparison. No torch, no codec; everything here is unit tested."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis.parallel_curves import compare

# Container header of src/codecs/dcvc_uf_worker.py (">4sBBBH").
CONTAINER_HEADER_BYTES = 9
TARGET_KBPS = (15.0, 30.0, 60.0)
# The smoke codes the middle target only; both stages choose each point the same way.
SELECTION_ORDER = (30.0, 15.0, 60.0)
ACCEPT_LEVELS = 1
METRICS = ('full_lpips', 'full_dists', 'foreground_lpips', 'foreground_dists')


def estimate_rate_kbps(nals: list[dict[str, Any]], *, target_frames: int, fps: float) -> float:
    """Rate of a ``target_frames`` stream from a short calibration encode.

    One I frame plus the mean cost of a displayed P frame in the calibration run
    (padding frames are paid for but not displayed, so they stay in the mean).
    """
    intra = [n for n in nals if n['type'] == 'I']
    inter = [n for n in nals if n['type'] == 'P']
    if len(intra) != 1 or not inter or target_frames < 2 or fps <= 0:
        raise ValueError('one I frame, at least one P NAL and a longer target are required')
    intra_bytes = intra[0]['nal_bytes'] + intra[0]['sps_bytes']
    shown = sum(n['display_frames'] for n in inter)
    per_frame = sum(n['nal_bytes'] + n['sps_bytes'] for n in inter) / shown
    total = CONTAINER_HEADER_BYTES + intra_bytes + per_frame * (target_frames - 1)
    return total * 8 * fps / target_frames / 1000


def select_points(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Distinct candidates closest in log-rate to the targets 30, 15, 60 kbps (the first ``count``).

    Rate alone decides. A candidate already taken for an earlier target is skipped;
    exact ties go to the larger resolution (smaller ``scale``), then the smaller QP.
    The result is ordered by target rate.
    """
    if count not in (1, 3):
        raise ValueError('one point (smoke) or three points (full) per window')
    taken: list[dict[str, Any]] = []
    for target in SELECTION_ORDER[:count]:
        free = [c for c in candidates if all((c['scale'], c['qp']) != (t['scale'], t['qp']) for t in taken)]
        if not free:
            raise ValueError('not enough distinct candidates')
        best = min(free, key=lambda c: (abs(math.log(c['est_kbps'] / target)), c['scale'], c['qp']))
        taken.append({**best, 'target_kbps': target})
    return sorted(taken, key=lambda c: c['target_kbps'])


def bracket_ok(points: list[dict[str, Any]], low: float = 20.0, high: float = 45.0) -> bool:
    """Whether estimated rates of the chosen points bracket (most of) 15-60 kbps."""
    rates = [p['est_kbps'] for p in points]
    return min(rates) <= low and max(rates) >= high


def planned_bracket(candidates: list[dict[str, Any]]) -> bool:
    """Whether the three-point plan (15, 30, 60 kbps) that the full run codes brackets 15-60 kbps.

    The smoke codes the 30 kbps point only, so the bracket is judged on the plan, not on the smoke's point."""
    return bracket_ok(select_points(candidates, 3))


def frame_max_diffs(first: Path, second: Path, frame_bytes: int, frames: int) -> list[int]:
    """Per-frame maximum absolute difference of two raw 8-bit files of equal frame size."""
    for path in (first, second):
        if path.stat().st_size != frame_bytes * frames:
            raise ValueError(f'{path} is not {frames} frames of {frame_bytes} bytes')
    a = np.memmap(first, np.uint8, 'r', shape=(frames, frame_bytes))
    b = np.memmap(second, np.uint8, 'r', shape=(frames, frame_bytes))
    return [int(np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).max()) for i in range(frames)]


def exactness(diffs: list[int], levels: int = ACCEPT_LEVELS) -> dict[str, Any]:
    """Acceptance of one comparison; the first frame that differs at all and the first beyond ``levels``."""
    if not diffs:
        raise ValueError('no frames compared')
    first_any = next((i for i, d in enumerate(diffs) if d > 0), None)
    first_bad = next((i for i, d in enumerate(diffs) if d > levels), None)
    return {'frames': len(diffs), 'max_abs_diff': max(diffs), 'frames_nonzero': sum(d > 0 for d in diffs),
            'first_frame_differing': first_any, 'first_frame_beyond_tolerance': first_bad,
            'tolerance_levels': levels, 'passed': first_bad is None}


def luma_psnr_per_frame(decoded: Path, source: Path, width: int, height: int, frames: int) -> list[float]:
    """Luma PSNR of each frame of two raw 4:2:0 files (a drift diagnostic, not a decision input)."""
    size = width * height * 3 // 2
    a = np.memmap(decoded, np.uint8, 'r', shape=(frames, size))[:, :width * height]
    b = np.memmap(source, np.uint8, 'r', shape=(frames, size))[:, :width * height]
    out = []
    for i in range(frames):
        mse = float(np.mean((a[i].astype(np.float64) - b[i].astype(np.float64)) ** 2))
        out.append(99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse))
    return out


def rate_dominance(test: list[tuple[float, float]], anchor: list[tuple[float, float]], samples: int = 200) -> dict[str, Any]:
    """Compare two (rate, quality) curves at every rate both cover, linear in log rate.

    Used when the quality ranges do not overlap, where a BD-rate is undefined.
    The anchor is its upper concave hull; the test curve is its Pareto front.
    """
    from experiments.tennis.k6b_side_feasibility import hull

    front: list[tuple[float, float]] = []
    for rate, quality in sorted(test):
        if not front or quality > front[-1][1]:
            front.append((rate, quality))
    envelope = hull(anchor)
    low, high = max(front[0][0], envelope[0][0]), min(front[-1][0], envelope[-1][0])
    if len(front) < 2 or high <= low:
        return {'status': 'no common rate range', 'rate_range_kbps': None, 'verdict': None}
    grid = np.exp(np.linspace(math.log(low), math.log(high), samples))
    t = np.interp(np.log(grid), np.log([r for r, _ in front]), [q for _, q in front])
    a = np.interp(np.log(grid), np.log([r for r, _ in envelope]), [q for _, q in envelope])
    delta = t - a
    verdict = ('better at every common rate' if (delta > 0).all() else
               'worse at every common rate' if (delta < 0).all() else 'crosses the anchor envelope')
    return {'status': 'common rate range', 'rate_range_kbps': [low, high], 'verdict': verdict,
            'min_quality_gain': float(delta.min()), 'max_quality_gain': float(delta.max())}


def envelope_comparison(test: dict[str, list[tuple[float, float]]], anchor: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    """Per metric: PCHIP BD-rate on overlapping quality (negative = fewer bits at equal quality), else dominance.

    ``test[m]`` / ``anchor[m]`` hold (kbps, distance); distances are negated into quality here.
    """
    out: dict[str, Any] = {}
    for metric in METRICS:
        t = [(r, -d) for r, d in test[metric]]
        a = [(r, -d) for r, d in anchor[metric]]
        result = compare(t, a)
        result['rate_dominance'] = rate_dominance(t, a)
        out[metric] = result
    return out


# W3b: W1's dominance rule applied to the DCVC points (docs/experiments/w3-neural-baseline.md, W3b).
DOMINANCE_MARGIN = 0.001


def interpolate_log_rate(points: list[dict[str, Any]], rate: float, metric: str) -> float | None:
    """Metric at ``rate``, linear in log-rate between the nearest points at or below and at or above it.

    Never extrapolates: None when no point lies on one side. ``points`` need ``rate_kbps`` and the metric."""
    below = [p for p in points if p['rate_kbps'] <= rate]
    above = [p for p in points if p['rate_kbps'] >= rate]
    if not below or not above:
        return None
    lo, hi = max(below, key=lambda p: p['rate_kbps']), min(above, key=lambda p: p['rate_kbps'])
    if hi['rate_kbps'] == lo['rate_kbps']:
        return float(lo[metric])
    weight = (math.log(rate) - math.log(lo['rate_kbps'])) / (math.log(hi['rate_kbps']) - math.log(lo['rate_kbps']))
    return float(lo[metric] + weight * (hi[metric] - lo[metric]))


def dominance(points: list[dict[str, Any]], rate: float, hybrid: dict[str, dict[str, float]],
              margin: float = DOMINANCE_MARGIN) -> dict[str, Any]:
    """W1's rule at ``rate`` for one curve: per distance (lpips, dists) whole <= hybrid whole + margin AND
    foreground <= hybrid foreground + margin; unresolved if either interpolation lacks a bracket.

    ``points`` hold ``rate_kbps`` and ``whole_<m>`` / ``foreground_<m>``; ``hybrid[m]`` has ``whole`` and ``foreground``."""
    out: dict[str, Any] = {}
    for m, ref in hybrid.items():
        whole = interpolate_log_rate(points, rate, f'whole_{m}')
        fg = interpolate_log_rate(points, rate, f'foreground_{m}')
        if whole is None or fg is None:
            out[m] = {'verdict': 'unresolved', 'whole': whole, 'foreground': fg}
            continue
        ok_whole, ok_fg = whole <= ref['whole'] + margin, fg <= ref['foreground'] + margin
        out[m] = {'verdict': 'dominates' if ok_whole and ok_fg else 'does not dominate', 'whole': whole, 'foreground': fg,
                  'whole_margin': ref['whole'] + margin - whole, 'foreground_margin': ref['foreground'] + margin - fg}
    return out
