"""Pose-stream coding for H2: what a hand's MANO parameters cost to send, per technique.

A track is one hand's consecutive frames. Per frame the sender has 51 values: global orientation
(axis-angle, 3), articulation (15 joints axis-angle, 45) and root (image position u, v in pixels and
log depth). Encoder, in order: smoothing, temporal subsampling, articulation subspace (MANO's pose
PCA), uniform quantization, prediction of each quantized symbol from the ones already sent. The
prediction is lossless, so it changes only the rate; the decoder dequantizes, leaves the subspace,
and fills the frames not sent by holding the last one or by linear interpolation to the next.

Rate is H1's estimate: the empirical entropy of each value's prediction residual, pooled over all
tracks, summed over values, plus one presence bit per sent frame. A track's first symbols (its
start) are sent once at a fixed cost and counted apart, as in H1.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

ANGLE_STEP = math.radians(1.0)
PX_STEP = 0.25
LOGZ_STEP = 0.002
ROOT = slice(48, 51)

SMOOTHERS: tuple[tuple[Any, ...], ...] = (
    ("none",),
    *(("one_euro", c, b) for c in (0.5, 1.0, 2.0) for b in (0.0, 0.5)),
    *(("gauss", ms) for ms in (33.3, 66.7, 133.3, 266.7)),
)
SEND_HZ = (None, 15.0, 10.0, 7.5)  # None: every frame
FILLS = ("hold", "linear")
SUBSPACES = (None, 6, 12, 24)
STEP_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
PREDICTIONS = ("previous", "velocity")
BUDGETS_MS = (0.0, 33.3, 100.0, 266.7)


def combinations() -> list[dict[str, Any]]:
    """Every coding combination H2 evaluates (3,080)."""
    out = []
    for smoother, hz, fill, k, scale, prediction in itertools.product(SMOOTHERS, SEND_HZ, FILLS, SUBSPACES,
                                                                      STEP_SCALES, PREDICTIONS):
        if hz is None and fill == "linear":
            continue  # nothing to fill
        out.append({"smoother": smoother, "send_hz": hz, "fill": fill, "subspace": k, "step_scale": scale,
                    "prediction": prediction})
    return out


def combo_key(c: dict[str, Any]) -> str:
    s = c["smoother"]
    smoother = s[0] if s[0] == "none" else f"{s[0]}:{':'.join(f'{v:g}' for v in s[1:])}"
    hz = "all" if c["send_hz"] is None else f"{c['send_hz']:g}Hz-{c['fill']}"
    k = "full" if c["subspace"] is None else f"pca{c['subspace']}"
    return f"{smoother}|{hz}|{k}|q{c['step_scale']:g}|{c['prediction']}"


# ----------------------------------------------------------------- root

def root_params(transl: np.ndarray, focal: float, centre: tuple[float, float]) -> np.ndarray:
    """MANO translation (N, 3) as image position of the origin (px) and log depth."""
    t = np.asarray(transl, float)
    return np.stack([focal * t[:, 0] / t[:, 2] + centre[0], focal * t[:, 1] / t[:, 2] + centre[1], np.log(t[:, 2])], -1)


def root_transl(params: np.ndarray, focal: float, centre: tuple[float, float]) -> np.ndarray:
    z = np.exp(params[:, 2])
    return np.stack([(params[:, 0] - centre[0]) * z / focal, (params[:, 1] - centre[1]) * z / focal, z], -1)


# ----------------------------------------------------------------- smoothing

def one_euro(x: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """Casiez et al.'s One-Euro filter, causal, per column."""
    def alpha(cutoff: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    out = np.array(x, float)
    dx = np.zeros(x.shape[1:])
    for t in range(1, len(x)):
        raw = (x[t] - out[t - 1]) * fps
        dx = dx + alpha(d_cutoff) * (raw - dx)
        a = alpha(min_cutoff + beta * np.abs(dx))
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
    return out


def gaussian(x: np.ndarray, radius: int) -> np.ndarray:
    """Centred Gaussian over ±radius frames (σ = radius/2), renormalised at the track's ends."""
    if radius < 1 or len(x) < 2:
        return np.array(x, float)
    k = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (k / (radius / 2.0)) ** 2)
    n = len(x)
    out = np.zeros_like(x, dtype=float)
    norm = np.zeros(n)
    for offset, weight in zip(k, w):
        lo, hi = max(0, -offset), min(n, n - offset)
        if hi <= lo:
            continue  # the window reaches past a short track
        out[lo:hi] += weight * x[lo + offset:hi + offset]
        norm[lo:hi] += weight
    return out / norm[:, None]


def lookahead_frames(smoother: tuple[Any, ...], fps: float) -> int:
    return int(round(smoother[1] * fps / 1000.0)) if smoother[0] == "gauss" else 0


def smooth(x: np.ndarray, smoother: tuple[Any, ...], fps: float) -> np.ndarray:
    if smoother[0] == "none":
        return np.array(x, float)
    if smoother[0] == "one_euro":
        return one_euro(x, fps, smoother[1], smoother[2])
    return gaussian(x, lookahead_frames(smoother, fps))


# ----------------------------------------------------------------- schedule, subspace, quantization

def send_step(hz: float | None, fps: float) -> int:
    return 1 if hz is None else max(1, int(round(fps / hz)))


def schedule(length: int, step: int) -> np.ndarray:
    """Frames sent: every ``step``-th from the first, and always the last."""
    sent = list(range(0, length, step))
    if sent[-1] != length - 1:
        sent.append(length - 1)
    return np.asarray(sent)


def latency_ms(combo: dict[str, Any], fps: float) -> float:
    frames = lookahead_frames(combo["smoother"], fps)
    if combo["send_hz"] is not None and combo["fill"] == "linear":
        frames += send_step(combo["send_hz"], fps) - 1
    return frames * 1000.0 / fps


@dataclass
class Basis:
    mean: np.ndarray  # (45,)
    components: np.ndarray  # (45, 45), rows

    def encode(self, pose: np.ndarray, k: int) -> np.ndarray:
        return (pose - self.mean) @ np.linalg.pinv(self.components[:k])

    def decode(self, coeffs: np.ndarray, k: int) -> np.ndarray:
        return self.mean + coeffs @ self.components[:k]


def to_coded(x: np.ndarray, k: int | None, basis: Basis) -> np.ndarray:
    if k is None:
        return x
    return np.concatenate([x[:, :3], basis.encode(x[:, 3:48], k), x[:, ROOT]], 1)


def from_coded(y: np.ndarray, k: int | None, basis: Basis) -> np.ndarray:
    if k is None:
        return y
    return np.concatenate([y[:, :3], basis.decode(y[:, 3:3 + k], k), y[:, -3:]], 1)


def steps(k: int | None, scale: float) -> np.ndarray:
    angles = 3 + (45 if k is None else k)
    return np.concatenate([np.full(angles, ANGLE_STEP), [PX_STEP, PX_STEP, LOGZ_STEP]]) * scale


def residuals(symbols: np.ndarray, prediction: str) -> np.ndarray:
    """Prediction residuals of a track's sent symbols (rows after the first, which is the start)."""
    if len(symbols) < 2:
        return np.zeros((0, symbols.shape[1]), np.int64)
    out = symbols[1:] - symbols[:-1]
    if prediction == "velocity" and len(symbols) > 2:
        out[1:] = symbols[2:] - (2 * symbols[1:-1] - symbols[:-2])
    return out


def fill(values: np.ndarray, sent: np.ndarray, length: int, mode: str) -> np.ndarray:
    """Values at every frame from the sent frames' values."""
    if len(sent) == length:
        return values
    idx = np.arange(length)
    if mode == "hold":
        return values[np.searchsorted(sent, idx, side="right") - 1]
    out = np.empty((length, values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.interp(idx, sent, values[:, d])
    return out


def encode(x: np.ndarray, combo: dict[str, Any], fps: float, basis: Basis,
           smoothed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One track: the quantized sent symbols (S, D) and the frames they belong to."""
    xs = smooth(x, combo["smoother"], fps) if smoothed is None else smoothed
    sent = schedule(len(x), send_step(combo["send_hz"], fps))
    y = to_coded(xs[sent], combo["subspace"], basis)
    return np.round(y / steps(combo["subspace"], combo["step_scale"])).astype(np.int64), sent


def decode(symbols: np.ndarray, sent: np.ndarray, length: int, combo: dict[str, Any], basis: Basis) -> np.ndarray:
    y = symbols * steps(combo["subspace"], combo["step_scale"])
    return fill(from_coded(y, combo["subspace"], basis), sent, length, combo["fill"])


def entropy_bits(values: Iterable[int]) -> float:
    counts = np.array(list(Counter(values).values()), float)
    if not counts.size:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def pooled_bits(residual_blocks: list[np.ndarray]) -> float:
    """Bits per sent frame: entropy of each value's residuals pooled over tracks, summed, plus presence."""
    blocks = [b for b in residual_blocks if len(b)]
    if not blocks:
        return 1.0
    stacked = np.concatenate(blocks, 0)
    return float(sum(entropy_bits(stacked[:, d].tolist()) for d in range(stacked.shape[1]))) + 1.0
