"""PointStream's ball mode (PLAN K5b): the ball sent as its positions and a per-window appearance, drawn by the
decoder over the pixel coder's decode.

Payload of the container's ``BALL`` segment (a bit string, zero-padded to whole bytes)::

    on                1 bit; 0 ends the payload (no ball drawn in this window)
    colour            3 x 8 bits (RGB)
    radius            4 bits, index into `RADII`
    exposure          3 bits, index into `EXPOSURES` (0: a sharp disk)
    phase             3 bits, index into `PHASES`
    order             3 bits, the Exp-Golomb order of the position residuals
    first visibility  1 bit; then each visibility run's length - 1 (Exp-Golomb order 0), until the window's
                      frame count (the container's header) is reached
    positions         per visible frame: a run's first position absolute (x in ceil(log2 width) bits, y in
                      ceil(log2 height) bits), then the residual of a constant-velocity prediction from the two
                      previous positions of the run (one previous: zero velocity), x then y, signed Exp-Golomb

Drawing: velocity is the central difference of the decoded positions (one-sided at a run's ends, zero for a
lone frame). The ball is a disk of radius r averaged over its exposure: ``SAMPLES`` instants spread over
``exposure`` of the frame's displacement, centred ``phase`` of that displacement from the coded position.
Alpha is the mean over the instants of an anti-aliased disk's coverage, so a fast ball fades as it lengthens,
as a camera's exposure blurs it; the composite is ``decode * (1 - alpha) + colour * alpha``.

The encoder (`fit`) chooses the appearance by analysis by synthesis over its own decode: for each radius,
exposure and phase on the grids, the colour in closed form (least squares), then the combination that most
lowers the window's squared error; off when none lowers it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

RADII = tuple(1.5 + 0.5 * i for i in range(10))
EXPOSURES = tuple(i / 7 for i in range(8))
PHASES = tuple((i - 3) / 6 for i in range(7))
SAMPLES = 16
MAX_ORDER = 7
#: Crop half-size ceiling: a labelled jump larger than this is drawn clipped (none in TrackNet's windows).
MAX_HALF = 96

#: Coded positions: (x, y) on frames where the ball is sent, None elsewhere.
Track = list[tuple[int, int] | None]


@dataclass(frozen=True)
class Appearance:
    colour: tuple[int, int, int]
    radius: int  # index into RADII
    exposure: int  # index into EXPOSURES
    phase: int  # index into PHASES

    @property
    def values(self) -> tuple[float, float, float]:
        return RADII[self.radius], EXPOSURES[self.exposure], PHASES[self.phase]


# ----------------------------------------------------------------- bits


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def put(self, value: int, width: int) -> None:
        if value < 0 or value >= 1 << width:
            raise ValueError(f"{value} does not fit {width} bits")
        self.bits.extend((value >> (width - 1 - i)) & 1 for i in range(width))

    def ue(self, value: int, order: int = 0) -> None:
        """Unsigned Exp-Golomb of order ``order``."""
        v = value + (1 << order)
        width = v.bit_length()
        self.bits.extend([0] * (width - 1 - order))
        self.put(v, width)

    def se(self, value: int, order: int) -> None:
        self.ue(2 * value - 1 if value > 0 else -2 * value, order)

    def tobytes(self) -> bytes:
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def bit(self) -> int:
        if self.pos >= len(self.data) * 8:
            raise ValueError("truncated ball payload")
        b = (self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1
        self.pos += 1
        return b

    def get(self, width: int) -> int:
        value = 0
        for _ in range(width):
            value = value << 1 | self.bit()
        return value

    def ue(self, order: int = 0) -> int:
        zeros = 0
        while not self.bit():
            zeros += 1
            if zeros > 40:
                raise ValueError("Exp-Golomb code too long")
        return ((1 << (zeros + order)) | self.get(zeros + order)) - (1 << order)

    def se(self, order: int) -> int:
        u = self.ue(order)
        return (u + 1) // 2 if u % 2 else -(u // 2)


def ue_bits(value: int, order: int) -> int:
    return 2 * (value + (1 << order)).bit_length() - 1 - order


def se_bits(value: int, order: int) -> int:
    return ue_bits(2 * value - 1 if value > 0 else -2 * value, order)


# ----------------------------------------------------------------- payload


def residuals(track: Track) -> list[tuple[int, int]]:
    """The constant-velocity residuals the payload carries (a run's first position excluded)."""
    out = []
    prev: list[tuple[int, int]] = []
    for p in track:
        if p is None:
            prev = []
            continue
        if len(prev) == 1:
            out.append((p[0] - prev[0][0], p[1] - prev[0][1]))
        elif prev:
            out.append((p[0] - (2 * prev[1][0] - prev[0][0]), p[1] - (2 * prev[1][1] - prev[0][1])))
        prev = (prev + [p])[-2:]
    return out


def best_order(track: Track) -> int:
    res = [c for r in residuals(track) for c in r]
    return min(range(MAX_ORDER + 1), key=lambda k: (sum(se_bits(c, k) for c in res), k))


def encode(track: Track, appearance: Appearance | None, width: int, height: int) -> bytes:
    w = BitWriter()
    if appearance is None:
        w.put(0, 1)
        return w.tobytes()
    w.put(1, 1)
    for c in appearance.colour:
        w.put(c, 8)
    w.put(appearance.radius, 4)
    w.put(appearance.exposure, 3)
    w.put(appearance.phase, 3)
    order = best_order(track)
    w.put(order, 3)
    seen = [p is not None for p in track]
    w.put(int(seen[0]) if seen else 0, 1)
    run = 0
    for t, s in enumerate(seen):
        run += 1
        if t + 1 == len(seen) or seen[t + 1] != s:
            w.ue(run - 1)
            run = 0
    bx, by = max(1, math.ceil(math.log2(width))), max(1, math.ceil(math.log2(height)))
    prev: list[tuple[int, int]] = []
    for p in track:
        if p is None:
            prev = []
            continue
        if not 0 <= p[0] < width or not 0 <= p[1] < height:
            raise ValueError(f"position {p} outside {width}x{height}")
        if not prev:
            w.put(p[0], bx)
            w.put(p[1], by)
        else:
            px, py = prev[0] if len(prev) == 1 else (2 * prev[1][0] - prev[0][0], 2 * prev[1][1] - prev[0][1])
            w.se(p[0] - px, order)
            w.se(p[1] - py, order)
        prev = (prev + [p])[-2:]
    return w.tobytes()


def decode(payload: bytes, frames: int, width: int, height: int) -> tuple[Track, Appearance | None]:
    r = BitReader(payload)
    if not r.bit():
        return [None] * frames, None
    colour = (r.get(8), r.get(8), r.get(8))
    appearance = Appearance(colour, r.get(4), r.get(3), r.get(3))
    if appearance.radius >= len(RADII) or appearance.exposure >= len(EXPOSURES) or appearance.phase >= len(PHASES):
        raise ValueError("appearance index out of range")
    order = r.get(3)
    state = bool(r.bit())
    seen: list[bool] = []
    while len(seen) < frames:
        seen.extend([state] * (r.ue() + 1))
        state = not state
    if len(seen) != frames:
        raise ValueError("visibility runs overrun the window")
    bx, by = max(1, math.ceil(math.log2(width))), max(1, math.ceil(math.log2(height)))
    track: Track = []
    prev: list[tuple[int, int]] = []
    for s in seen:
        if not s:
            track.append(None)
            prev = []
            continue
        if not prev:
            p = (r.get(bx), r.get(by))
        else:
            px, py = prev[0] if len(prev) == 1 else (2 * prev[1][0] - prev[0][0], 2 * prev[1][1] - prev[0][1])
            p = (px + r.se(order), py + r.se(order))
        track.append(p)
        prev = (prev + [p])[-2:]
    return track, appearance


# ----------------------------------------------------------------- drawing


def velocities(track: Track) -> np.ndarray:
    """(frames, 2) float: central difference of the positions within a run, one-sided at its ends, 0 alone."""
    v = np.zeros((len(track), 2))
    for t, p in enumerate(track):
        if p is None:
            continue
        a = track[t - 1] if t > 0 else None
        b = track[t + 1] if t + 1 < len(track) else None
        if a is not None and b is not None:
            v[t] = ((b[0] - a[0]) / 2, (b[1] - a[1]) / 2)
        elif b is not None:
            v[t] = (b[0] - p[0], b[1] - p[1])
        elif a is not None:
            v[t] = (p[0] - a[0], p[1] - a[1])
    return v


def half_size(speed: float) -> int:
    """Crop half-size around a coded position that holds the drawn ball at any grid setting, the judge's 8 px
    disk and the 8 px band around the frame's displacement."""
    return min(MAX_HALF, int(math.ceil(max(RADII[-1] + 1.0 + speed, 8.0 + speed / 2.0))) + 1)


def coverage(dx: np.ndarray, dy: np.ndarray, velocity: np.ndarray, radius: float, exposure: float, phase: float,
             samples: int = SAMPLES) -> np.ndarray:
    """Alpha on offsets (dx, dy) from the coded position, broadcast against ``velocity[..., None, None]``:
    an anti-aliased disk's coverage averaged over the exposure's instants."""
    vx, vy = velocity[..., 0, None, None], velocity[..., 1, None, None]
    n = samples if exposure > 0 else 1
    acc = None
    for i in range(n):
        s = phase + ((i + 0.5) / n - 0.5) * exposure
        d = np.hypot(dx - s * vx, dy - s * vy)
        a = np.clip(radius + 0.5 - d, 0.0, 1.0)
        acc = a if acc is None else acc + a
    assert acc is not None
    return (acc / n).astype(np.float32)


def draw(frame: np.ndarray, position: tuple[int, int], velocity: np.ndarray, appearance: Appearance) -> np.ndarray:
    """The frame (h, w, 3) uint8 with the ball composited at ``position``."""
    radius, exposure, phase = appearance.values
    h, w = frame.shape[:2]
    half = half_size(float(np.hypot(*velocity)))
    x, y = position
    x0, x1, y0, y1 = max(0, x - half), min(w, x + half + 1), max(0, y - half), min(h, y + half + 1)
    out = frame.copy()
    if x0 >= x1 or y0 >= y1:
        return out
    dy, dx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = coverage(dx, dy, np.asarray(velocity, np.float32), radius, exposure, phase)[..., None]
    crop = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(np.round(crop * (1 - a) + np.asarray(appearance.colour, np.float32) * a), 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------- the encoder's fit


@dataclass
class Crops:
    """Stacked crops around every coded position of a window, grouped by crop size: per group the frame
    indices, the offsets grid, the velocities, a validity mask (inside the frame) and the source's pixels."""

    groups: list[dict[str, Any]]

    @staticmethod
    def build(track: Track, source: Any, velocity: np.ndarray) -> Crops:
        h, w = source.shape[1:3]
        by_half: dict[int, list[int]] = {}
        for t, p in enumerate(track):
            if p is not None:
                by_half.setdefault(bucket(half_size(float(np.hypot(*velocity[t])))), []).append(t)
        groups = []
        for half, frames in sorted(by_half.items()):
            dy, dx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
            centres = np.array([track[t] for t in frames])
            valid = ((centres[:, 0, None, None] + dx >= 0) & (centres[:, 0, None, None] + dx < w)
                     & (centres[:, 1, None, None] + dy >= 0) & (centres[:, 1, None, None] + dy < h))
            groups.append({"half": half, "frames": frames, "dx": dx, "dy": dy, "centres": centres,
                           "velocity": velocity[frames].astype(np.float32), "valid": valid.astype(np.float32),
                           "source": crops_of(source, frames, centres, half)})
        return Crops(groups)


def bucket(half: int) -> int:
    """Crop half-sizes rounded up to a few classes, so frames stack."""
    for b in (12, 16, 24, 32, 48, 64):
        if half <= b:
            return b
    return MAX_HALF


def crops_of(frames_rgb: Any, frames: list[int], centres: np.ndarray, half: int) -> np.ndarray:
    """(n, 2 half + 1, 2 half + 1, 3) float32 crops, zero outside the frame."""
    h, w = frames_rgb.shape[1:3]
    size = 2 * half + 1
    out = np.zeros((len(frames), size, size, 3), np.float32)
    for i, (t, (x, y)) in enumerate(zip(frames, centres)):
        x0, x1, y0, y1 = max(0, x - half), min(w, x + half + 1), max(0, y - half), min(h, y + half + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        out[i, y0 - (y - half):y1 - (y - half), x0 - (x - half):x1 - (x - half)] = np.asarray(frames_rgb[t, y0:y1, x0:x1])
    return out


def terms(crops: Crops, decodes: list[list[np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """Per crop group, the decodes' fixed least-squares terms over valid pixels, as columns (decode, channel):
    with e = source - decode, ``alpha`` columns e and decode * e, ``alpha2`` columns decode and decode^2;
    and per decode and channel sum(e^2)."""
    out = []
    for g, group in enumerate(crops.groups):
        valid = group["valid"].reshape(-1, 1)
        src = group["source"].reshape(-1, 3)
        a_cols, a2_cols, ee = [], [], []
        for dec in decodes:
            d = dec[g].reshape(-1, 3)
            e = (src - d) * valid
            dv = d * valid
            a_cols += [e, dv * e]
            a2_cols += [dv, dv * dv]
            ee.append((e.astype(np.float64) ** 2).sum(0))
        out.append({"alpha": np.concatenate(a_cols, 1), "alpha2": np.concatenate(a2_cols, 1), "ee": np.array(ee)})
    return out


def sums(crops: Crops, fixed: list[dict[str, np.ndarray]], radius: float, exposure: float, phase: float) -> np.ndarray:
    """Per decode, the least-squares terms of one appearance setting: (decodes, 3 channels, 3) holding
    sum(alpha^2), sum(alpha * w), sum(w^2) with w = source - decode * (1 - alpha), over valid pixels."""
    n = len(fixed[0]["ee"])
    out = np.zeros((n, 3, 3))
    for group, f in zip(crops.groups, fixed):
        a = (coverage(group["dx"], group["dy"], group["velocity"], radius, exposure, phase) * group["valid"]).reshape(-1)
        a2 = a * a
        pa = (a @ f["alpha"]).astype(np.float64).reshape(n, 2, 3)  # sum(a e), sum(a d e)
        pa2 = (a2 @ f["alpha2"]).astype(np.float64).reshape(n, 2, 3)  # sum(a^2 d), sum(a^2 d^2)
        out[:, :, 0] += float(a2.astype(np.float64).sum())
        out[:, :, 1] += pa[:, 0] + pa2[:, 0]
        out[:, :, 2] += f["ee"] + 2 * pa[:, 1] + pa2[:, 1]
    return out


def base_error(crops: Crops, decode: list[np.ndarray]) -> float:
    return float(sum((((g["source"] - d) * g["valid"][..., None]) ** 2).sum() for g, d in zip(crops.groups, decode)))


def fit(crops: Crops, decodes: list[list[np.ndarray]], settings: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """Per decode (its crops as `crops_of` gives them, per group), the appearance among ``settings`` (radius,
    exposure, phase indices) with the rounded least-squares colour that most lowers the squared error, or None
    when none lowers it; with the error change of each."""
    fixed = terms(crops, decodes)
    base = [float(b) for b in np.sum([f["ee"] for f in fixed], 0).sum(1)]
    best: list[dict[str, Any]] = [{"appearance": None, "delta_se": 0.0, "base_se": b} for b in base]
    for ri, ei, pi in settings:
        s = sums(crops, fixed, RADII[ri], EXPOSURES[ei], PHASES[pi])
        for k in range(len(decodes)):
            aa, aw, ww = s[k, :, 0], s[k, :, 1], s[k, :, 2]
            colour = np.clip(np.round(np.where(aa > 0, aw / np.maximum(aa, 1e-12), 0.0)), 0, 255)
            delta = float((colour * colour * aa - 2 * colour * aw + ww).sum() - base[k])
            if delta < best[k]["delta_se"] - 1e-6:
                best[k] = {"appearance": Appearance((int(colour[0]), int(colour[1]), int(colour[2])), ri, ei, pi),
                           "delta_se": delta, "base_se": base[k]}
    return best


def grid(kind: str) -> list[tuple[int, int, int]]:
    """The settings each renderer searches: the disk over radius (no exposure, no phase); the streak over all."""
    centre = PHASES.index(0.0)
    if kind == "disk":
        return [(r, 0, centre) for r in range(len(RADII))]
    if kind == "streak":
        return [(r, e, p) for r in range(len(RADII)) for e in range(len(EXPOSURES)) for p in range(len(PHASES))
                if e or p == centre]
    raise ValueError(kind)
