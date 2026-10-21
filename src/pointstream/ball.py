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


def _put_track(w: BitWriter, track: Track, order: int, width: int, height: int) -> None:
    """Visibility runs then positions, as the module docstring lays them out."""
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


def _get_track(r: BitReader, frames: int, order: int, width: int, height: int) -> Track:
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
    return track


def _put_appearance(w: BitWriter, a: Appearance) -> None:
    for c in a.colour:
        w.put(c, 8)
    w.put(a.radius, 4)
    w.put(a.exposure, 3)
    w.put(a.phase, 3)


def _get_appearance(r: BitReader) -> Appearance:
    colour = (r.get(8), r.get(8), r.get(8))
    a = Appearance(colour, r.get(4), r.get(3), r.get(3))
    _check(a)
    return a


def _check(a: Appearance) -> None:
    if not (0 <= a.radius < len(RADII) and 0 <= a.exposure < len(EXPOSURES) and 0 <= a.phase < len(PHASES)
            and all(0 <= c < 256 for c in a.colour)):
        raise ValueError("appearance index out of range")


def encode(track: Track, appearance: Appearance | None, width: int, height: int) -> bytes:
    w = BitWriter()
    if appearance is None:
        w.put(0, 1)
        return w.tobytes()
    w.put(1, 1)
    _put_appearance(w, appearance)
    order = best_order(track)
    w.put(order, 3)
    _put_track(w, track, order, width, height)
    return w.tobytes()


def decode(payload: bytes, frames: int, width: int, height: int) -> tuple[Track, Appearance | None]:
    r = BitReader(payload)
    if not r.bit():
        return [None] * frames, None
    appearance = _get_appearance(r)
    order = r.get(3)
    return _get_track(r, frames, order, width, height), appearance


# ----------------------------------------------------------------- per-frame payload (K5b 1b)


def _appearance_deltas(appearances: list[Appearance]) -> list[tuple[int, ...]]:
    """Per visible frame after the first, the differences from the previous one: radius, exposure, phase, colour."""
    return [(b.radius - a.radius, b.exposure - a.exposure, b.phase - a.phase, *(cb - ca for ca, cb in zip(a.colour, b.colour)))
            for a, b in zip(appearances, appearances[1:])]


def encode_frames(track: Track, appearances: list[Appearance | None], width: int, height: int) -> bytes:
    """The per-frame payload (K5b 1b)::

        on              1 bit; 0 ends the payload
        orders          3 bits position-residual order, 3 bits colour-difference order
        track           visibility runs and positions, as `encode`
        appearances     the first visible frame's absolute (as `encode`); each later visible frame's radius,
                        exposure and phase differences from the previous visible frame (signed Exp-Golomb order 0)
                        and its colour differences (signed Exp-Golomb, the colour order)

    ``appearances[t]`` is None exactly where ``track[t]`` is."""
    if len(appearances) != len(track) or any((a is None) != (p is None) for a, p in zip(appearances, track)):
        raise ValueError("an appearance per visible frame, none elsewhere")
    shown = [a for a in appearances if a is not None]
    w = BitWriter()
    if not shown:
        w.put(0, 1)
        return w.tobytes()
    w.put(1, 1)
    deltas = _appearance_deltas(shown)
    colour = [c for d in deltas for c in d[3:]]
    corder = min(range(MAX_ORDER + 1), key=lambda k: (sum(se_bits(c, k) for c in colour), k))
    order = best_order(track)
    w.put(order, 3)
    w.put(corder, 3)
    _put_track(w, track, order, width, height)
    _put_appearance(w, shown[0])
    for d in deltas:
        for v in d[:3]:
            w.se(v, 0)
        for v in d[3:]:
            w.se(v, corder)
    return w.tobytes()


def decode_frames(payload: bytes, frames: int, width: int, height: int) -> tuple[Track, list[Appearance | None]]:
    r = BitReader(payload)
    if not r.bit():
        return [None] * frames, [None] * frames
    order, corder = r.get(3), r.get(3)
    track = _get_track(r, frames, order, width, height)
    out: list[Appearance | None] = []
    prev: Appearance | None = None
    for p in track:
        if p is None:
            out.append(None)
            continue
        if prev is None:
            a = _get_appearance(r)
        else:
            dr, de, dp = r.se(0), r.se(0), r.se(0)
            colour = tuple(c + r.se(corder) for c in prev.colour)
            a = Appearance((colour[0], colour[1], colour[2]), prev.radius + dr, prev.exposure + de, prev.phase + dp)
            _check(a)
        out.append(a)
        prev = a
    return track, out


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


def patch(frame: np.ndarray, position: tuple[int, int], velocity: np.ndarray, appearance: Appearance
          ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The region (y0, y1, x0, x1) the ball changes and its composited pixels (uint8), or None off the frame."""
    radius, exposure, phase = appearance.values
    h, w = frame.shape[:2]
    half = half_size(float(np.hypot(*velocity)))
    x, y = position
    x0, x1, y0, y1 = max(0, x - half), min(w, x + half + 1), max(0, y - half), min(h, y + half + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    dy, dx = np.mgrid[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = coverage(dx, dy, np.asarray(velocity, np.float32), radius, exposure, phase)[..., None]
    crop = np.asarray(frame[y0:y1, x0:x1]).astype(np.float32)
    return (y0, y1, x0, x1), np.clip(np.round(crop * (1 - a) + np.asarray(appearance.colour, np.float32) * a), 0, 255).astype(np.uint8)


def draw(frame: np.ndarray, position: tuple[int, int], velocity: np.ndarray, appearance: Appearance) -> np.ndarray:
    """The frame (h, w, 3) uint8 with the ball composited at ``position``."""
    out = np.array(frame, copy=True)
    got = patch(frame, position, velocity, appearance)
    if got is not None:
        (y0, y1, x0, x1), pixels = got
        out[y0:y1, x0:x1] = pixels
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


# ----------------------------------------------------------------- the encoder's per-frame fit (K5b 1b)

#: Position refinement: each coded position moves at most SHIFT px per axis from the tracker's.
SHIFT = 4
#: Appearance and position steps alternate: appearance, then ROUNDS times (position, appearance).
ROUNDS = 2
#: A frame's ball is sent only when it lowers the crop's squared error by more than this (float rounding of the sums).
MIN_GAIN = 0.5
#: Alpha grid half-sizes of the fit, so frames stack.
FIT_BUCKETS = (8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, MAX_HALF)
#: Settings per batched product in the appearance step (bounds memory).
CHUNK = 64


def fit_half(speed: float) -> int:
    """Half-size of the alpha grid holding the ball at any setting: radius, anti-aliasing, and up to one frame's
    displacement (exposure / 2 + |phase| <= 1)."""
    need = int(math.ceil(RADII[-1] + 1.0 + speed)) + 1
    return next(b for b in FIT_BUCKETS if need <= b or b == MAX_HALF)


@dataclass
class FrameCrops:
    """Crops around a tracker's positions with a SHIFT margin, grouped by the alpha grid's size: per group the frame
    indices, the start centres, the start velocities, the alpha grid (dx, dy) of side 2 half + 1, and on the
    margin grid (side 2 (half + SHIFT) + 1) the validity, the source and every decode."""

    groups: list[dict[str, Any]]
    frames: int

    @staticmethod
    def build(start: Track, source: Any, decodes: list[Any]) -> FrameCrops:
        h, w = source.shape[1:3]
        velocity = velocities(start)
        by_half: dict[int, list[int]] = {}
        for t, p in enumerate(start):
            if p is not None:
                by_half.setdefault(fit_half(float(np.hypot(*velocity[t]))), []).append(t)
        groups = []
        for half, frames in sorted(by_half.items()):
            m = half + SHIFT
            centres = np.array([start[t] for t in frames])
            yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
            valid = ((centres[:, 0, None, None] + xx >= 0) & (centres[:, 0, None, None] + xx < w)
                     & (centres[:, 1, None, None] + yy >= 0) & (centres[:, 1, None, None] + yy < h)).astype(np.float32)
            dy, dx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
            groups.append({"half": half, "frames": frames, "centres": centres, "velocity": velocity[frames].astype(np.float32),
                           "dx": dx, "dy": dy, "valid": valid, "source": crops_of(source, frames, centres, m),
                           "decodes": np.stack([crops_of(d, frames, centres, m) for d in decodes])})
        return FrameCrops(groups, len(start))


def alpha_table(g: dict[str, Any], settings: list[tuple[int, int, int]]) -> np.ndarray:
    """(settings, n, P) float32: each frame's alpha at each setting, on the group's grid (P = (2 half + 1)^2)."""
    n = len(g["frames"])
    out = np.empty((len(settings), n, g["dx"].size), np.float32)
    for j, (ri, ei, pi) in enumerate(settings):
        out[j] = coverage(g["dx"], g["dy"], g["velocity"], RADII[ri], EXPOSURES[ei], PHASES[pi]).reshape(n, -1)
    return out


def _window(a: np.ndarray, off: np.ndarray, side: int) -> np.ndarray:
    """Per frame i the (side x side) window of ``a[i]`` (n, S, S, ...) at offset ``off[i]`` (x, y) from the centre."""
    n = len(a)
    rows = SHIFT + off[:, 1, None] + np.arange(side)[None]
    cols = SHIFT + off[:, 0, None] + np.arange(side)[None]
    return a[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]


def _appearance_step(g: dict[str, Any], table: np.ndarray, off: np.ndarray) -> dict[str, np.ndarray]:
    """Per decode and frame of the group, at the given offsets (D, n, 2), the setting (an index into the table) and
    rounded least-squares colour that most lower the crop's squared error, with that change (positive when no
    setting helps). Every setting at once: (n, settings, P) @ (n, P, columns)."""
    side = 2 * g["half"] + 1
    D, n = g["decodes"].shape[:2]
    cols1, cols2 = [], []
    for k in range(D):
        v = _window(g["valid"], off[k], side)[..., None]
        d = _window(g["decodes"][k], off[k], side) * v
        e = _window(g["source"], off[k], side) * v - d
        cols1 += [e, d * e]
        cols2 += [d, d * d, v]
    x1 = np.concatenate(cols1, -1).reshape(n, side * side, -1)  # per decode: e (3), d e (3)
    x2 = np.concatenate(cols2, -1).reshape(n, side * side, -1)  # per decode: d (3), d^2 (3), v (1)
    best_delta = np.full((D, n), np.inf)
    best = np.zeros((D, n), np.int64)
    best_colour = np.zeros((D, n, 3), np.int64)
    for c0 in range(0, len(table), CHUNK):
        a = np.ascontiguousarray(np.transpose(table[c0:c0 + CHUNK], (1, 0, 2)))  # (n, C, P)
        p1 = np.matmul(a, x1).astype(np.float64).reshape(n, -1, D, 2, 3)
        p2 = np.matmul(a * a, x2).astype(np.float64).reshape(n, -1, D, 7)
        aa = p2[..., 6:7]
        aw = p1[..., 0, :] + p2[..., 0:3]
        colour = np.clip(np.round(np.where(aa > 0, aw / np.maximum(aa, 1e-12), 0.0)), 0, 255)
        delta = (colour * colour * aa - 2 * colour * aw + 2 * p1[..., 1, :] + p2[..., 3:6]).sum(-1)  # (n, C, D)
        j = delta.argmin(1)  # (n, D)
        low = np.take_along_axis(delta, j[:, None], 1)[:, 0].T  # (D, n)
        better = low < best_delta - 1e-6
        best_delta = np.where(better, low, best_delta)
        best = np.where(better, c0 + j.T, best)
        chosen = np.take_along_axis(colour, j[:, None, :, None], 1)[:, 0]  # (n, D, 3)
        best_colour = np.where(better[..., None], np.transpose(chosen, (1, 0, 2)), best_colour).astype(np.int64)
    return {"setting": best, "colour": best_colour, "delta": best_delta}


def _offset_delta(g: dict[str, Any], k: int, a: np.ndarray, colour: np.ndarray, ox: int, oy: int) -> np.ndarray:
    side = 2 * g["half"] + 1
    sl = np.s_[:, SHIFT + oy:SHIFT + oy + side, SHIFT + ox:SHIFT + ox + side]
    v = g["valid"][sl][..., None]
    d = g["decodes"][k][sl]
    e = (g["source"][sl] - d) * v
    cd = (colour - d) * v
    return (a * a * cd * cd - 2 * a * e * cd).sum((1, 2, 3)).astype(np.float64)


def _position_step(g: dict[str, Any], table: np.ndarray, step: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Per decode and frame the offset within SHIFT (x, y) that most lowers the crop's error with its appearance:
    every second offset first, then the 8 neighbours of the best."""
    side = 2 * g["half"] + 1
    D, n = g["decodes"].shape[:2]
    h, w = shape
    cx, cy = g["centres"][:, 0], g["centres"][:, 1]
    best = np.zeros((D, n, 2), np.int64)
    for k in range(D):
        a = table[step["setting"][k], np.arange(n)].reshape(n, side, side, 1)
        colour = step["colour"][k][:, None, None, :].astype(np.float32)
        low = np.full(n, np.inf)

        def consider(ox: Any, oy: Any, delta: np.ndarray) -> None:
            inside = (cx + ox >= 0) & (cx + ox < w) & (cy + oy >= 0) & (cy + oy < h)
            better = inside & (delta < low - 1e-6)
            low[better] = delta[better]
            best[k, better, 0] = np.broadcast_to(ox, n)[better]
            best[k, better, 1] = np.broadcast_to(oy, n)[better]

        for oy in range(-SHIFT, SHIFT + 1, 2):
            for ox in range(-SHIFT, SHIFT + 1, 2):
                consider(ox, oy, _offset_delta(g, k, a, colour, ox, oy))
        coarse = best[k].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                o = np.clip(coarse + (dx, dy), -SHIFT, SHIFT)
                windows = {"source": _window(g["source"], o, side), "valid": _window(g["valid"], o, side)[..., None],
                           "decode": _window(g["decodes"][k], o, side)}
                e = (windows["source"] - windows["decode"]) * windows["valid"]
                cd = (colour - windows["decode"]) * windows["valid"]
                consider(o[:, 0], o[:, 1], (a * a * cd * cd - 2 * a * e * cd).sum((1, 2, 3)).astype(np.float64))
    return best


def fit_frames(crops: FrameCrops, settings: list[tuple[int, int, int]], shape: tuple[int, int],
               rounds: int = ROUNDS) -> list[dict[str, Any]]:
    """Per decode the encoder's per-frame ball: positions refined within SHIFT of the tracker's and an appearance per
    frame, alternating the two steps, each frame shown only where its ball lowers the crop's error (by the fit's
    approximation: the velocities of the tracker's positions). Returns per decode the track, the appearances and
    the fitted error change."""
    D = crops.groups[0]["decodes"].shape[0] if crops.groups else 0
    tracks: list[Track] = [[None] * crops.frames for _ in range(D)]
    apps: list[list[Appearance | None]] = [[None] * crops.frames for _ in range(D)]
    deltas = np.zeros(D)
    for g in crops.groups:
        n = len(g["frames"])
        table = alpha_table(g, settings)
        off = np.zeros((D, n, 2), np.int64)
        step = _appearance_step(g, table, off)
        for _ in range(rounds):
            off = _position_step(g, table, step, shape)
            step = _appearance_step(g, table, off)
        for k in range(D):
            for i, t in enumerate(g["frames"]):
                if step["delta"][k, i] < -MIN_GAIN:
                    ri, ei, pi = settings[int(step["setting"][k, i])]
                    c = step["colour"][k, i]
                    x, y = g["centres"][i] + off[k, i]
                    tracks[k][t] = (int(x), int(y))
                    apps[k][t] = Appearance((int(c[0]), int(c[1]), int(c[2])), ri, ei, pi)
                    deltas[k] += step["delta"][k, i]
    return [{"track": tracks[k], "appearances": apps[k], "delta_se": float(deltas[k])} for k in range(D)]


def render(frame: np.ndarray, t: int, track: Track, velocity: np.ndarray, appearances: list[Appearance | None]) -> np.ndarray:
    """The decoder's frame ``t``: the decode with the per-frame ball drawn, when one is sent."""
    p, a = track[t], appearances[t]
    return draw(frame, p, velocity[t], a) if p is not None and a is not None else frame
