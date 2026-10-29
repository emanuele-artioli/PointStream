"""W2: how cheap can the tennis player keypoints be? CPU-only coders with real round trips.

Decision rule and definitions: docs/experiments/w2-pose-rate.md. The reference is the
decoded original KPS packet (240 frames per player, a 1-pixel grid). Every coder here
writes an actual byte stream (header plus an adaptive binary range coder) and is
measured on what a decoder reconstructs from those bytes alone.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRANSPORT_SHA256 = '6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7'
WINDOWS = {
    'Al': {'id': 'alcaraz_hurkacz/031@129',
           'packet_sha256': '38c29a952e59e2aeb72de71284286dc2855df3cd0de948571c493bc23349ed50'},
    'Med': {'id': 'medvedev_struff/014@194',
            'packet_sha256': '0246c32cafb59ce16a9b0514e58b615b6714f07f00cee3285515075efe77f915'},
}
FRAME = (1920, 1080)
MAGIC = 0xA7
JOINTS = 17
CONF = 0.5
MEAN_LIMIT = 0.0066
P95_LIMIT = 0.0132
RATE_LIMIT_KBPS = 0.5
LEVELS = (0.0033, 0.0066, 0.0132, 0.0264)

# ----------------------------------------------------------------------------- range coder

_TOP = 1 << 24
_PROB_BITS = 12
_COUNT_LIMIT = 255
_UNARY_MAX = 14


class _Model:
    """Count-based adaptive bit probabilities, identical in the encoder and the decoder."""

    def __init__(self) -> None:
        self.counts: dict[Any, list[int]] = {}

    def p0(self, ctx: Any) -> int:
        n0, n1 = self.counts.get(ctx, (0, 0))
        p = ((2 * n0 + 1) << _PROB_BITS) // (2 * (n0 + n1) + 2)
        return min(max(p, 1), (1 << _PROB_BITS) - 1)

    def update(self, ctx: Any, bit: int) -> None:
        c = self.counts.setdefault(ctx, [0, 0])
        c[bit] += 1
        if c[0] + c[1] > _COUNT_LIMIT:
            c[0] = (c[0] + 1) // 2
            c[1] = (c[1] + 1) // 2


class Encoder:
    """LZMA-style carry-propagating range encoder; the leading zero byte and trailing zeros are dropped."""

    encoding = True

    def __init__(self) -> None:
        self.model = _Model()
        self.low = 0
        self.range = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()
        self.section = 'other'
        self.cost: dict[str, float] = {}

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low >= (1 << 32):
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def _encode(self, bit: int, p0: int) -> None:
        bound = (self.range >> _PROB_BITS) * p0
        if bit:
            self.low += bound
            self.range -= bound
        else:
            self.range = bound
        while self.range < _TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self._shift_low()

    def bit(self, ctx: Any, value: int | None) -> int:
        if value not in (0, 1):
            raise ValueError('The encoder needs a bit')
        p0 = self.model.p0(ctx)
        p = p0 if value == 0 else (1 << _PROB_BITS) - p0
        self.cost[self.section] = self.cost.get(self.section, 0.0) - math.log2(p / (1 << _PROB_BITS))
        self._encode(value, p0)
        self.model.update(ctx, value)
        return value

    def raw(self, value: int | None) -> int:
        if value not in (0, 1):
            raise ValueError('The encoder needs a bit')
        self.cost[self.section] = self.cost.get(self.section, 0.0) + 1.0
        self._encode(value, 1 << (_PROB_BITS - 1))
        return value

    def finish(self) -> bytes:
        top = self.low + self.range - 1
        for shift in range(33, -1, -1):  # the value in [low, low+range) with most trailing zeros
            value = (top >> shift) << shift
            if value >= self.low:
                self.low = value
                break
        for _ in range(5):
            self._shift_low()
        if self.out[0] != 0:
            raise AssertionError('Range coder leading byte must be zero')
        data = bytes(self.out[1:])
        return data.rstrip(b'\x00')


class Decoder:
    encoding = False

    def __init__(self, data: bytes) -> None:
        self.model = _Model()
        self.data = data
        self.pos = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        self.section = 'other'
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self) -> int:
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def _decode(self, p0: int) -> int:
        bound = (self.range >> _PROB_BITS) * p0
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        while self.range < _TOP:
            self.range = (self.range << 8) & 0xFFFFFFFF
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit

    def bit(self, ctx: Any, value: int | None = None) -> int:
        bit = self._decode(self.model.p0(ctx))
        self.model.update(ctx, bit)
        return bit

    def raw(self, value: int | None = None) -> int:
        return self._decode(1 << (_PROB_BITS - 1))


Coder = Encoder | Decoder


def _raw_bits(io_: Coder, value: int | None, width: int) -> int:
    out = 0
    for i in range(width - 1, -1, -1):
        out = (out << 1) | io_.raw(None if value is None else (value >> i) & 1)
    return out


def _exp_golomb(io_: Coder, r: int | None) -> int:
    """r >= 0: adaptive unary length of r+1, then raw mantissa bits."""
    k = 0 if r is None else (r + 1).bit_length() - 1
    n = 0
    while True:
        more = io_.bit(('egk', min(n, 15)), None if r is None else int(k > n))
        if not more:
            break
        n += 1
    low = _raw_bits(io_, None if r is None else (r + 1) - (1 << n), n)
    return (1 << n) + low - 1


def code_signed(io_: Coder, v: int | None, kind: Any, magnitude_ctx: int, sign_ctx: Any) -> int:
    """Zero flag, sign, unary magnitude with an Exp-Golomb escape; v is None when decoding."""
    nz = io_.bit((kind, 'z', magnitude_ctx), None if v is None else int(v != 0))
    if not nz:
        return 0
    neg = io_.bit((kind, 's', sign_ctx), None if v is None else int(v < 0))
    m = None if v is None else abs(v) - 1
    i = 0
    while i < _UNARY_MAX:
        more = io_.bit((kind, 'm', min(i, 7), magnitude_ctx), None if m is None else int(m > i))
        if not more:
            break
        i += 1
    else:
        i = _UNARY_MAX + _exp_golomb(io_, None if m is None else m - _UNARY_MAX)
    return -(i + 1) if neg else i + 1


# ----------------------------------------------------------------------------- the coders


@dataclass(frozen=True)
class Setting:
    q: int = 4
    key: int = 1
    predictor: str = 'velocity'  # 'previous' | 'velocity'
    interp: str = 'linear'  # 'linear' | 'cubic'
    vis: str = 'all'  # 'all' every frame | 'keys' on keyframes only (held between)

    def __post_init__(self) -> None:
        if (self.q < 1 or self.q > 255 or self.key < 1 or self.key > 255
                or self.predictor not in ('previous', 'velocity')
                or self.interp not in ('linear', 'cubic') or self.vis not in ('all', 'keys')):
            raise ValueError(f'Invalid setting {self}')

    @property
    def label(self) -> str:
        return f'q{self.q}-k{self.key}-{self.predictor[:3]}-{self.interp[:3]}-vis{self.vis}'


def key_times(n: int, key: int) -> list[int]:
    """Every key-th frame plus the last one, so interpolation never has to hold or extrapolate."""
    if n < 1:
        raise ValueError('Empty run')
    times = list(range(0, n, key))
    if times[-1] != n - 1:
        times.append(n - 1)
    return times


def split_runs(t: np.ndarray) -> list[tuple[int, int]]:
    """(start frame, length) of the maximal runs of consecutive frames."""
    t = np.asarray(t, np.int64)
    if len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Frame times must be strictly increasing')
    cuts = [0, *(np.flatnonzero(np.diff(t) != 1) + 1).tolist(), len(t)]
    return [(int(t[a]), b - a) for a, b in zip(cuts, cuts[1:])]


def quantize_keys(xy: np.ndarray, setting: Setting) -> np.ndarray:
    """Grid indices of the key frames: the nearest multiple of q, independent of the predictor."""
    kt = key_times(len(xy), setting.key)
    return np.floor(xy[kt] / setting.q + 0.5).astype(np.int64)


def interpolate(keys: np.ndarray, n: int, setting: Setting) -> np.ndarray:
    """Decoded positions (n, 17, 2) from key positions (m, 17, 2) in pixels."""
    kt = key_times(n, setting.key)
    if len(kt) != len(keys):
        raise ValueError('Key count does not match the run length')
    out = np.empty((n, JOINTS, 2), np.float64)
    if len(kt) == 1:
        out[:] = keys[0]
        return out
    for i in range(len(kt) - 1):
        a, b = kt[i], kt[i + 1]
        u = ((np.arange(a, b + 1) - a) / (b - a))[:, None, None]
        p1, p2 = keys[i], keys[i + 1]
        if setting.interp == 'linear':
            seg = p1 + u * (p2 - p1)  # exact where the keys are equal
        else:
            # Missing neighbours are linear extrapolations, so a straight line is reproduced exactly.
            p0 = keys[i - 1] if i > 0 else 2 * keys[0] - keys[1]
            p3 = keys[i + 2] if i + 2 < len(kt) else 2 * keys[-1] - keys[-2]
            seg = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                         + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3)
        out[a:b + 1] = seg
    return out


def _vis_times(n: int, setting: Setting) -> list[int]:
    return list(range(n)) if setting.vis == 'all' else key_times(n, setting.key)


def expand_vis(coded: np.ndarray, n: int, setting: Setting) -> np.ndarray:
    """Visibility for every frame; with keyframe-only visibility each frame holds the previous key's."""
    if setting.vis == 'all':
        return coded.astype(bool)
    kt = key_times(n, setting.key)
    out = np.empty((n, JOINTS), bool)
    for i in range(len(kt) - 1):
        out[kt[i]:kt[i + 1]] = coded[i]
    out[n - 1] = coded[-1]
    return out


def _code_run(io_: Coder, n: int, setting: Setting, q_in: np.ndarray | None,
              vis_in: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Code (or decode) one run; returns key grid indices (m, 17, 2) and coded visibility."""
    kt = key_times(n, setting.key)
    vt = _vis_times(n, setting)
    vis = np.zeros((len(vt), JOINTS), bool)
    io_.section = 'visibility'
    for i in range(len(vt)):
        for j in range(JOINTS):
            prev_t = 2 if i == 0 else int(vis[i - 1, j])
            prev_j = 2 if j == 0 else int(vis[i, j - 1])
            prev2 = 2 if i < 2 or prev_t == 2 else int(vis[i - 2, j])
            value = None if vis_in is None else int(vis_in[vt[i], j])
            vis[i, j] = bool(io_.bit(('v', prev_t, prev_j, prev2), value))
    io_.section = 'coordinates'
    m = len(kt)
    q = np.zeros((m, JOINTS, 2), np.int64)
    res = np.zeros((m, JOINTS, 2), np.int64)
    for a in (0, 1):  # the first key: joint 0 against the frame centre, the others against their predecessor
        centre = FRAME[a] // 2 // setting.q
        d0 = code_signed(io_, None if q_in is None else int(q_in[0, 0, a]) - centre, ('abs', a), 0, 1)
        q[0, 0, a] = centre + d0
    for j in range(1, JOINTS):
        for a in (0, 1):
            d = code_signed(io_, None if q_in is None else int(q_in[0, j, a] - q[0, j - 1, a]),
                            ('first', a), 0, 1)
            q[0, j, a] = q[0, j - 1, a] + d
    vis_index = (lambda k: kt[k]) if setting.vis == 'all' else (lambda k: k)
    for k in range(1, m):
        seen = vis[vis_index(k)]
        for j in range(JOINTS):
            for a in (0, 1):
                if k == 1 or setting.predictor == 'previous':
                    pred = q[k - 1, j, a]
                else:
                    pred = 2 * q[k - 1, j, a] - q[k - 2, j, a]
                above = abs(int(res[k, j - 1, a])) if j else 0
                before = abs(int(res[k - 1, j, a])) if k >= 2 else 0
                s = above + before
                mag = (0 if s == 0 else 1 if s <= 2 else 2 if s <= 5 else 3) + 4 * int(seen[j])
                sign = int(np.sign(res[k, j - 1, a])) + 1 if j else 1
                r = code_signed(io_, None if q_in is None else int(q_in[k, j, a] - pred),
                                ('c', a), mag, (a, sign))
                res[k, j, a] = r
                q[k, j, a] = pred + r
    return q, vis


def _put_varint(buf: bytearray, v: int) -> None:
    if v < 0:
        raise ValueError('Negative varint')
    while v >= 128:
        buf.append((v & 127) | 128)
        v >>= 7
    buf.append(v)


def _get_varint(data: bytes, pos: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        if pos >= len(data) or shift > 28:
            raise ValueError('Truncated or oversized varint')
        b = data[pos]
        pos += 1
        out |= (b & 127) << shift
        if b < 128:
            return out, pos
        shift += 7


@dataclass
class Encoded:
    blob: bytes
    header_bytes: int
    cost_bits: dict[str, float]
    expected: list[dict[str, Any]]  # what the encoder knows the decoder will output


def encode_window(units: list[dict[str, Any]], setting: Setting) -> Encoded:
    """Units: {'t': frame indices, 'kp': (n, 17, 3) with confidence in the last column}."""
    header = bytearray([MAGIC, (setting.predictor == 'velocity') | ((setting.interp == 'cubic') << 1)
                        | ((setting.vis == 'keys') << 2), setting.q, setting.key, len(units)])
    plans = []
    for u in units:
        runs = split_runs(u['t'])
        _put_varint(header, len(runs))
        for start, length in runs:
            _put_varint(header, start)
            _put_varint(header, length)
        plans.append(runs)
    enc = Encoder()
    expected = []
    for u, runs in zip(units, plans):
        kp = np.asarray(u['kp'], np.float64)
        xy_all, conf_all = kp[:, :, :2], kp[:, :, 2] >= CONF
        cursor = 0
        out_kp: list[np.ndarray] = []
        out_t: list[int] = []
        for start, length in runs:
            xy, conf = xy_all[cursor:cursor + length], conf_all[cursor:cursor + length]
            q_in = quantize_keys(xy, setting)
            q, vis = _code_run(enc, length, setting, q_in, conf)
            dec = interpolate(q * float(setting.q), length, setting)
            out_kp.append(np.concatenate(
                [dec, expand_vis(vis, length, setting)[:, :, None].astype(np.float64)], 2))
            out_t.extend(range(start, start + length))
            cursor += length
        expected.append({'t': np.asarray(out_t, np.int64), 'kp': np.concatenate(out_kp)})
    payload = enc.finish()
    return Encoded(bytes(header) + payload, len(header), dict(enc.cost), expected)


def decode_window(blob: bytes) -> list[dict[str, Any]]:
    if len(blob) < 5 or blob[0] != MAGIC:
        raise ValueError('Not a W2 keypoint stream')
    flags, q, key, n_units = blob[1], blob[2], blob[3], blob[4]
    if flags & ~7 or n_units == 0:
        raise ValueError('Invalid stream header')
    setting = Setting(q=q, key=key, predictor='velocity' if flags & 1 else 'previous',
                      interp='cubic' if flags & 2 else 'linear', vis='keys' if flags & 4 else 'all')
    pos = 5
    plans = []
    for _ in range(n_units):
        n_runs, pos = _get_varint(blob, pos)
        runs = []
        for _ in range(n_runs):
            start, pos = _get_varint(blob, pos)
            length, pos = _get_varint(blob, pos)
            if not 0 < length <= 100_000:
                raise ValueError('Invalid run length')
            runs.append((start, length))
        if not runs:
            raise ValueError('A unit without runs')
        plans.append(runs)
    dec = Decoder(blob[pos:])
    units = []
    for runs in plans:
        out_kp: list[np.ndarray] = []
        out_t: list[int] = []
        for start, length in runs:
            qk, vis = _code_run(dec, length, setting, None, None)
            xy = interpolate(qk * float(setting.q), length, setting)
            out_kp.append(np.concatenate(
                [xy, expand_vis(vis, length, setting)[:, :, None].astype(np.float64)], 2))
            out_t.extend(range(start, start + length))
        units.append({'t': np.asarray(out_t, np.int64), 'kp': np.concatenate(out_kp)})
    return units


# ----------------------------------------------------------------------------- measurement


def extent_height(kp: np.ndarray) -> float:
    """The player's box height: median over frames of the confident joints' vertical extent."""
    heights = []
    for frame in kp:
        c = frame[:, 2] >= CONF
        if c.sum() >= 2:
            heights.append(float(np.ptp(frame[c, 1])))
    if not heights:
        raise ValueError('A player without two confident joints in any frame')
    return float(np.median(heights))


def crop_height(kp: np.ndarray, margin: float = 1.4, percentile: float = 90,
                gen: tuple[int, int] = (512, 784)) -> float:
    """The generator's crop height rule (k6b.needed_height, 90th percentile over frames)."""
    heights = []
    for frame in kp:
        c = frame[:, 2] >= CONF
        if c.sum() >= 2:
            h, w = float(np.ptp(frame[c, 1])), float(np.ptp(frame[c, 0]))
            heights.append(margin * max(h, w * gen[1] / gen[0], 8.0))
    return float(np.percentile(heights, percentile)) if heights else 64.0


def _second_difference(xy: np.ndarray, conf: np.ndarray, runs: list[tuple[int, int]],
                       t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mags, masks = [], []
    cursor = 0
    for _, length in runs:
        p = xy[cursor:cursor + length]
        c = conf[cursor:cursor + length]
        if length >= 3:
            mags.append(np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=2))
            masks.append(c[2:] & c[1:-1] & c[:-2])
        cursor += length
    if not mags:
        return np.zeros(0), np.zeros(0, bool)
    return np.concatenate(mags).ravel(), np.concatenate(masks).ravel()


def player_error(ref: dict[str, Any], dec: dict[str, Any]) -> dict[str, float]:
    """Displacement and jitter of one decoded player against its reference."""
    if not np.array_equal(ref['t'], dec['t']):
        raise ValueError('Decoded frame coverage differs from the reference')
    r, d = np.asarray(ref['kp'], np.float64), np.asarray(dec['kp'], np.float64)
    conf = r[:, :, 2] >= CONF
    height, crop = extent_height(r), crop_height(r)
    dist = np.linalg.norm(d[:, :, :2] - r[:, :, :2], axis=2)
    runs = split_runs(ref['t'])
    s_ref, mask = _second_difference(r[:, :, :2], conf, runs, ref['t'])
    s_dec, _ = _second_difference(d[:, :, :2], conf, runs, ref['t'])
    jitter_ref = float(s_ref[mask].mean() / height) if mask.any() else float('nan')
    jitter_dec = float(s_dec[mask].mean() / height) if mask.any() else float('nan')
    return {
        'height_px': height, 'crop_height_px': crop, 'frames': int(len(r)),
        'confident_joints': int(conf.sum()),
        'mean': float(dist[conf].mean() / height), 'p95': float(np.percentile(dist[conf], 95) / height),
        'mean_all17': float(dist.mean() / height), 'p95_all17': float(np.percentile(dist, 95) / height),
        'mean_crop': float(dist[conf].mean() / crop), 'p95_crop': float(np.percentile(dist[conf], 95) / crop),
        'max_px': float(dist[conf].max()),
        'jitter_ref': jitter_ref, 'jitter_dec': jitter_dec,
        'jitter_ratio': jitter_dec / jitter_ref if jitter_ref else float('nan'),
        'visibility_mismatch': float(((d[:, :, 2] >= CONF) != conf).mean()),
    }


def passes(error: dict[str, float], mean_limit: float = MEAN_LIMIT, p95_limit: float = P95_LIMIT) -> bool:
    return error['mean'] <= mean_limit and error['p95'] <= p95_limit


def run_setting(units: list[dict[str, Any]], setting: Setting, seconds: float) -> dict[str, Any]:
    """Encode, decode from the bytes alone, check the decoder equals the encoder's expectation, measure."""
    began = time.monotonic()
    enc = encode_window(units, setting)
    dec = decode_window(enc.blob)
    for e, d in zip(enc.expected, dec):
        if not (np.array_equal(e['t'], d['t']) and np.array_equal(e['kp'], d['kp'])):
            raise ValueError(f'Round trip mismatch for {setting.label}')
    if len(dec) != len(units):
        raise ValueError('Unit count changed')
    players = [player_error(u, d) for u, d in zip(units, dec)]
    return {
        'setting': asdict(setting), 'label': setting.label, 'bytes': len(enc.blob),
        'header_bytes': enc.header_bytes, 'kbps': len(enc.blob) * 8 / seconds / 1000,
        'estimated_bytes_by_section': {k: v / 8 for k, v in enc.cost_bits.items()},
        'sha256': hashlib.sha256(enc.blob).hexdigest(), 'round_trip_exact': True,
        'players': players, 'passes_error': all(passes(p) for p in players),
        'seconds': time.monotonic() - began,
    }


def stage1_settings(predictor: str) -> dict[str, list[Setting]]:
    return {
        'a': [Setting(4, 1, p) for p in ('previous', 'velocity')],
        'b': [Setting(8, 1, predictor), Setting(16, 1, predictor)],
        'c': [Setting(4, k, predictor, i) for k in (2, 4) for i in ('linear', 'cubic')],
    }


def stage2_settings() -> list[Setting]:
    out = []
    for q in (1, 2, 4, 8, 16, 32):
        for key in (1, 2, 4, 8):
            for p in ('previous', 'velocity'):
                for interp in (('linear',) if key == 1 else ('linear', 'cubic')):
                    for vis in (('all',) if key == 1 else ('all', 'keys')):
                        out.append(Setting(q, key, p, interp, vis))
    return out


def best_at_levels(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per window and error level: the lowest-rate setting whose every player passes (p95 <= 2 x mean limit)."""
    out: dict[str, dict[str, Any]] = {}
    for window, results in rows.items():
        for level in LEVELS:
            ok = [r for r in results
                  if all(p['mean'] <= level and p['p95'] <= 2 * level for p in r['players'])]
            best = min(ok, key=lambda r: r['bytes'], default=None)
            out.setdefault(f'{level * 100:.2f}%', {})[window] = (
                None if best is None else {'label': best['label'], 'bytes': best['bytes'],
                                           'kbps': best['kbps'], 'visibility_mismatch': max(
                                               p['visibility_mismatch'] for p in best['players'])})
    return out


# ----------------------------------------------------------------------------- inputs


def sha256_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_packets(transport: Path) -> dict[str, dict[str, Any]]:
    """Verify the archive SHA256 before reading any member, then read the two KPS packets in memory."""
    from experiments.tennis.k6b_rate_streams import decode_keypoints

    digest = sha256_file(transport)
    if digest != TRANSPORT_SHA256:
        raise ValueError('Transport archive SHA256 mismatch')
    out = {}
    with tarfile.open(transport) as tar:
        for name, w in WINDOWS.items():
            member = f'packets/{w["id"].replace("/", "__")}/keypoints.kps'
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f'Missing member {member}')
            packet = io.BytesIO(stream.read()).getvalue()
            if hashlib.sha256(packet).hexdigest() != w['packet_sha256']:
                raise ValueError(f'{name} packet SHA256 mismatch')
            meta, units = decode_keypoints(packet)
            frames, fps = meta['native_frames'], meta['fps'][0] / meta['fps'][1]
            out[name] = {'packet_bytes': len(packet), 'units': units, 'frames': frames, 'fps': fps,
                         'seconds': frames / fps, 'window': meta['window']}
    return out


def side_terms(window_seconds: float = 9.6) -> dict[str, Any]:
    """The other side terms for Al, from existing reports only (no new measurement)."""
    kps_bytes = 4236
    other_kbps = 2.0771288888888884  # k6b-side-feasibility-2026-10-28.json
    bootstrap_bytes = 444  # s4-bootstrap.json of the actual-base job 20261027T211451Z-f6a754eb
    adapter_kbps = 1.65  # assessment 2026-10-29: about 1.65 kbps over 2 h
    boot_kbps = bootstrap_bytes * 8 / window_seconds / 1000
    return {
        'allowance_total_side_kbps': 0.668734, 'kps_kbps': kps_bytes * 8 / window_seconds / 1000,
        'other_side_kbps': other_kbps, 'adapter_guider_kbps_approx': adapter_kbps,
        'bootstrap_kbps': boot_kbps,
        'match_bank_kbps_by_subtraction': other_kbps - adapter_kbps - boot_kbps,
        'remaining_with_free_keypoints_kbps': 0.668734 - other_kbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['1', '2', 'both'], default='both')
    args = parser.parse_args()
    began = time.monotonic()
    data = load_packets(args.transport)
    record: dict[str, Any] = {
        'scope': 'CPU coders on the decoded original keypoints of two development windows; no images.',
        'input_sha256': {'transport_published_tar': TRANSPORT_SHA256,
                         **{n: w['packet_sha256'] for n, w in WINDOWS.items()}},
        'windows': {n: {k: v for k, v in d.items() if k != 'units'} for n, d in data.items()},
        'rule': {'mean': MEAN_LIMIT, 'p95': P95_LIMIT, 'kbps': RATE_LIMIT_KBPS},
        'original_packet_kbps': {n: d['packet_bytes'] * 8 / d['seconds'] / 1000 for n, d in data.items()},
        'side_terms_al': side_terms(),
    }
    stage1: dict[str, dict[str, list[dict[str, Any]]]] = {}
    chosen = 'velocity'
    if args.stage in ('1', 'both'):
        # The predictor for (b) and (c) is the (a) winner by total bytes over both windows.
        a_total = {}
        for setting in stage1_settings('velocity')['a']:
            a_total[setting.predictor] = sum(
                run_setting(d['units'], setting, d['seconds'])['bytes'] for d in data.values())
        chosen = min(a_total, key=lambda p: a_total[p])
        record['stage1_predictor_choice'] = {'total_bytes_family_a': a_total, 'chosen': chosen}
        for family, settings in stage1_settings(chosen).items():
            stage1[family] = {n: [run_setting(d['units'], s, d['seconds']) for s in settings]
                              for n, d in data.items()}
        record['stage1'] = stage1
        flat = [r for fam in stage1.values() for rows in fam.values() for r in rows]
        by_setting: dict[str, list[dict[str, Any]]] = {}
        for r in flat:
            by_setting.setdefault(r['label'], []).append(r)
        record['stage1_candidates'] = [
            label for label, rs in by_setting.items() if len(rs) >= 2 and all(
                r['passes_error'] and r['kbps'] <= RATE_LIMIT_KBPS for r in rs)]
    if args.stage in ('2', 'both'):
        grid: dict[str, list[dict[str, Any]]] = {n: [] for n in data}
        for setting in stage2_settings():
            for n, d in data.items():
                grid[n].append(run_setting(d['units'], setting, d['seconds']))
        record['stage2'] = grid
        record['best_at_error_level'] = best_at_levels(grid)
        labels = {r['label'] for r in grid['Al']}
        record['stage2_candidates'] = sorted(
            label for label in labels if all(
                (lambda r: r['passes_error'] and r['kbps'] <= RATE_LIMIT_KBPS)(
                    next(x for x in grid[n] if x['label'] == label)) for n in data))
    record['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['wall_seconds'] = time.monotonic() - began
    with args.out.open('x') as f:
        json.dump(record, f, indent=1, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.out), 'wall_seconds': record['wall_seconds'],
                      'stage1_candidates': record.get('stage1_candidates'),
                      'stage2_candidates': record.get('stage2_candidates')}))


if __name__ == '__main__':
    main()
