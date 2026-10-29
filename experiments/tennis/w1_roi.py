"""W1 shared logic: the SVT ROI grid, its rate-only CRF search and the frozen decision rule.

Rule and grid: docs/experiments/w1-roi-pareto.md. Nothing here looks at a quality number
except `decide` and `foreground_at_whole`, which apply the rule written before any run.
Importing this module has no side effect (the K6b native-base module pins the process to
one CPU on import, so none of its constants are imported from there).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

WINDOW = 'alcaraz_hurkacz/031@129'
PACKET_DIR = 'packets/alcaraz_hurkacz__031@129/'
FPS = 25
FRAMES = 240
TICKS = list(range(0, FRAMES, 3))
SOURCE_START = 9  # first source frame of the window inside the 031 clip (k6b_native_base.SOURCES)
SOURCE_SHA = '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76'
SOURCE_YUV_SHA = 'e7b960cd2456b439c5439b337a45e47d2a0a44a2a32025a2de595af3feed4fa2'
TRANSPORT_SHA = '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7'
ANCHOR_SHA = '67444a2a8f56c8700fb073a1184ce207ec3ad38cc003dbafa6710bb33836dff9'
HYBRID_ARCHIVE_SHA = 'e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1'
WEIGHTS = {'lpips': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
           'dists': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}
TOOLS = {'SvtAv1EncApp': '4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4',
         'dav1d': 'e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6',
         'ffmpeg': '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'}

TARGET_KBPS = 22.737962222222222  # the hybrid's total rate (job 20261027T211451Z-f6a754eb)
HYBRID = {'lpips': {'whole': 0.2757866112979842, 'foreground': 0.34587557402445757},
          'dists': {'whole': 0.1696124367415905, 'foreground': 0.2525512419641018}}
ORIGINAL_KPS_BYTES = 4236  # the keypoint packet inside the hybrid's 22.737962 kbps
MARGIN = 0.001
OFFSETS = (0, -20, -40)
#: coded size per scale divisor: the K6b bases code the half resolution as 960x544.
CODED = {1: (1920, 1080), 2: (960, 544), 4: (480, 272), 8: (240, 136)}
CRF_MIN, CRF_MAX = 1, 70  # SvtAv1EncApp 4.2.0 accepts --crf 1..70
CONTROL_CRF = 63
#: file bytes of the existing plain CRF 63 Al bases (k6b_real_base_judge rates, 240 frames).
CONTROL_IVF_BYTES = {8: 12237, 4: 20557, 2: 40422}
#: rows of the hybrid job's scores.npz holding the plain bases (0: scale 8, 2: scale 4, 4: scale 2).
CONTROL_ROW = {8: 0, 4: 2, 2: 4}
SLOPE_PRIOR = 0.07  # d ln(rate) / d CRF, only to place the first guesses
METRICS = ('lpips', 'dists')
CONTROL_TOLERANCE = 1e-4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rate_kbps(nbytes: int, frames: int = FRAMES, fps: int = FPS) -> float:
    """Complete IVF file bytes over the window's duration, as the existing anchors are rated."""
    return nbytes * 8 / (frames / fps) / 1000


def secondary_kbps(new_packet_bytes: int) -> float:
    """The hybrid re-billed with a smaller keypoint packet (same generated pixels): the saving is
    (4236 - B) bytes over the 9.6 s window."""
    return TARGET_KBPS - (ORIGINAL_KPS_BYTES - new_packet_bytes) * 8 / (FRAMES / FPS) / 1000


def policy_name(scale: int, offset: int) -> str:
    return f's{scale}_o{-offset}' if offset else f's{scale}_o0'


# ---------------------------------------------------------------- the ROI map


def roi_text(regions: np.ndarray, width: int, height: int, inside: int) -> str:
    """SVT-AV1 `--roi-map-file`: per frame, one QP offset per 64x64 block, `inside` for every
    block touched by the frozen source R and 0 elsewhere (a negative `inside` spends more
    bits there). Same blocks as parallel_baselines.roi_text; the sign convention differs."""
    if regions.ndim != 3 or regions.dtype != bool or width % 2 or height % 2:
        raise ValueError('Boolean native regions and even coded geometry required')
    if not -63 <= inside <= 63:
        raise ValueError('ROI offset outside the encoder range')
    lines = []
    for t, region in enumerate(regions):
        touched = touched_blocks(region, width, height)
        lines.append(str(t) + ' ' + ' '.join(str(inside) if x else '0' for x in touched.ravel()))
    return '\n'.join(lines) + '\n'


def touched_blocks(region: np.ndarray, width: int, height: int) -> np.ndarray:
    """(rows, cols) bool: 64x64 blocks of the coded frame that any pixel of `region` touches
    (nearest-neighbour resize to the coded size, raster order, row 0 at the top)."""
    import cv2

    rows, cols = (height + 63) // 64, (width + 63) // 64
    resized = cv2.resize(region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    padded = np.zeros((rows * 64, cols * 64), bool)
    padded[:height, :width] = resized
    return padded.reshape(rows, 64, cols, 64).any(axis=(1, 3))


# ---------------------------------------------------------------- the rate-only CRF search


def _flip(known: dict[int, float], target: float) -> int | None:
    """Smallest CRF a with rate(a) >= target > rate(a + 1), both measured."""
    for a in sorted(known):
        if known[a] >= target and a + 1 in known and known[a + 1] < target:
            return a
    return None


def next_crf(known: dict[int, float], target: float, start: int) -> tuple[str, int | None]:
    """Decide the next CRF to encode from measured rates alone (never from quality).

    Returns ('encode', crf) or a terminal status with None:
    'complete' (a bracketing pair of adjacent CRFs plus one neighbour are measured),
    'unbracketed_high' (the rate is still above the target at CRF 70) and
    'unbracketed_low' (the rate is already below it at CRF 1).
    """
    if not known:
        return 'encode', min(CRF_MAX, max(CRF_MIN, start))
    flip = _flip(known, target)
    if flip is not None:
        for neighbour in (flip + 2, flip - 1):
            if CRF_MIN <= neighbour <= CRF_MAX and neighbour not in known:
                return 'encode', neighbour
        return 'complete', None
    above = [c for c in known if known[c] >= target]
    below = [c for c in known if known[c] < target]
    if above and below:
        a = max(above)
        later = [c for c in below if c > a]
        if later:
            b = min(later)
            # Interpolate ln(rate) linearly in CRF to place the next probe strictly between.
            span = math.log(known[a]) - math.log(known[b])
            guess = a + (math.log(known[a]) - math.log(target)) / span * (b - a) if span > 0 else (a + b) / 2
            return 'encode', min(b - 1, max(a + 1, round(guess)))
        # Non-monotone: every lower-rate point sits at a smaller CRF. Probe just above the highest.
        if a + 1 <= CRF_MAX and a + 1 not in known:
            return 'encode', a + 1
        return 'unbracketed_high', None
    slope = SLOPE_PRIOR
    if len(known) >= 2:
        crfs = sorted(known)
        measured = (math.log(known[crfs[0]]) - math.log(known[crfs[-1]])) / (crfs[-1] - crfs[0])
        if measured > 0.01:
            slope = measured
    if above:
        top = max(above)
        if top >= CRF_MAX:
            return 'unbracketed_high', None
        step = max(1, math.ceil((math.log(known[top]) - math.log(target)) / slope))
        return 'encode', min(CRF_MAX, top + step)
    bottom = min(below)
    if bottom <= CRF_MIN:
        return 'unbracketed_low', None
    step = max(1, math.ceil((math.log(target) - math.log(known[bottom])) / slope))
    return 'encode', max(CRF_MIN, bottom - step)


def select_crfs(known: dict[int, float], target: float) -> list[int] | None:
    """The three CRFs to score: the bracketing adjacent pair and one neighbour, or None."""
    flip = _flip(known, target)
    if flip is None:
        return None
    chosen = [flip, flip + 1]
    for neighbour in (flip + 2, flip - 1):
        if neighbour in known:
            chosen.append(neighbour)
            break
    return sorted(chosen)


def start_crf(scale: int, offset: int) -> int:
    """First probe, from the plain CRF 63 rates already measured (half 33.685 kbps; ROI adds rate)."""
    return {4: 60, 2: 67, 1: 70}[scale]


# ---------------------------------------------------------------- the frozen rule


def bracketing_pair(points: list[dict[str, Any]], target: float = TARGET_KBPS) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Nearest scored point below and nearest above the target rate; None if either is missing."""
    below = [p for p in points if p['rate_kbps'] < target]
    above = [p for p in points if p['rate_kbps'] >= target]
    if not below or not above:
        return None
    return max(below, key=lambda p: p['rate_kbps']), min(above, key=lambda p: p['rate_kbps'])


def interpolate_log_rate(target: float, low: tuple[float, float], high: tuple[float, float]) -> float:
    (r0, v0), (r1, v1) = low, high
    weight = (math.log(target) - math.log(r0)) / (math.log(r1) - math.log(r0))
    return v0 + weight * (v1 - v0)


def policy_at_target(points: list[dict[str, Any]], target: float = TARGET_KBPS) -> dict[str, Any] | None:
    """Whole and foreground scores of one policy at the target rate, linear in log-rate between
    its two bracketing points; None (unresolved) when no bracket exists. Never extrapolates."""
    pair = bracketing_pair(points, target)
    if pair is None:
        return None
    low, high = pair
    out: dict[str, Any] = {'crf': [low['crf'], high['crf']], 'rate_kbps': [low['rate_kbps'], high['rate_kbps']]}
    for metric in METRICS:
        for part in ('whole', 'foreground'):
            out[f'{part}_{metric}'] = interpolate_log_rate(
                target, (low['rate_kbps'], low[f'{part}_{metric}']), (high['rate_kbps'], high[f'{part}_{metric}']))
    return out


def dominates(value: dict[str, Any], metric: str, hybrid: dict[str, dict[str, float]] = HYBRID,
              margin: float = MARGIN) -> bool:
    return (value[f'whole_{metric}'] <= hybrid[metric]['whole'] + margin
            and value[f'foreground_{metric}'] <= hybrid[metric]['foreground'] + margin)


def foreground_at_whole(curve: list[tuple[float, float]], whole: float) -> float | None:
    """Best (lowest) foreground score at a given whole-frame score, piecewise linear over the
    policy's (whole, foreground) points; None when `whole` lies outside their range."""
    points = sorted(curve)
    if len(points) < 2 or not points[0][0] <= whole <= points[-1][0]:
        return None
    best = None
    for (w0, f0), (w1, f1) in zip(points, points[1:]):
        if w0 <= whole <= w1:
            value = f0 if w1 == w0 else f0 + (whole - w0) / (w1 - w0) * (f1 - f0)
            best = value if best is None else min(best, value)
    return best


def decide(policies: dict[str, dict[str, Any] | None], hybrid: dict[str, dict[str, float]] = HYBRID,
           margin: float = MARGIN) -> dict[str, Any]:
    """Outcome B if any resolved policy dominates the hybrid on either metric, else A."""
    dominating: dict[str, list[str]] = {m: [] for m in METRICS}
    for name, value in policies.items():
        if value is None:
            continue
        for metric in METRICS:
            if dominates(value, metric, hybrid, margin):
                dominating[metric].append(name)
    resolved = sorted(n for n, v in policies.items() if v is not None)
    return {'outcome': 'B' if any(dominating.values()) else 'A', 'dominating': dominating,
            'resolved': resolved, 'unresolved': sorted(set(policies) - set(resolved)),
            'margin': margin, 'hybrid': hybrid}


def best_foreground(policies: dict[str, dict[str, Any] | None], hybrid: dict[str, dict[str, float]] = HYBRID) -> dict[str, Any]:
    """Per resolution and metric: foreground at the hybrid's whole-frame score, over the offsets."""
    out: dict[str, Any] = {}
    for scale in sorted({int(n.split('_')[0][1:]) for n in policies}):
        for metric in METRICS:
            curve = [(v[f'whole_{metric}'], v[f'foreground_{metric}'])
                     for n, v in policies.items() if v is not None and n.startswith(f's{scale}_')]
            out[f's{scale}_{metric}'] = {
                'foreground_at_hybrid_whole': foreground_at_whole(curve, hybrid[metric]['whole']),
                'hybrid_foreground': hybrid[metric]['foreground'], 'points': sorted(curve)}
    return out


def analyse(by_policy: dict[str, list[dict[str, Any]]], scales: tuple[int, ...], target: float,
            hybrid: dict[str, dict[str, float]] = HYBRID) -> dict[str, Any]:
    """The whole rule at one target rate over the scored points of every policy."""
    policies: dict[str, dict[str, Any] | None] = {name: policy_at_target(rows, target) for name, rows in by_policy.items()}
    for scale in scales:
        for offset in OFFSETS:
            policies.setdefault(policy_name(scale, offset), None)
    return {'target_kbps': target, 'at_target': policies, 'decision': decide(policies, hybrid),
            'best_foreground': best_foreground(policies, hybrid)}
