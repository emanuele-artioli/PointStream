"""Versioned, self-delimiting native video plus per-frame BALL symbols.

PSBL retains the complete native container verbatim. All header, identifier,
native and BALL bytes count on the wire. Installed decoders and model weights
are separate resources; the identifier names them, it does not include them.
SHA-256 detects corruption, not malicious replacement or authenticity.
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from src.pointstream import ball

MAGIC = b"PSBL"
VERSION = 1
# magic, version, geometry, frame rate, count, codec, identifier length, payload lengths
_HEADER = struct.Struct(">4sBHHIIIBHII")
_DIGEST_SIZE = 32
MAX_NATIVE = 512 * 1024 * 1024
MAX_BALL = 8 * 1024 * 1024
MAX_FRAMES = 65535
MAX_DIMENSION = 8192
MAX_IDENTIFIER = 128
_CODECS = {"svt-av1-ivf": 1, "dcvc-uf": 2}
_IDENTIFIER = re.compile(r"[A-Za-z0-9._:+@/-]+\Z")


@dataclass(frozen=True)
class Stream:
    width: int
    height: int
    fps: Fraction
    frames: int
    codec: str
    resource_id: str
    native: bytes
    ball_payload: bytes

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frames, 1) / self.fps


def _validate(stream: Stream) -> None:
    if not (1 <= stream.width <= MAX_DIMENSION and 1 <= stream.height <= MAX_DIMENSION):
        raise ValueError("invalid geometry")
    if not (1 <= stream.frames <= MAX_FRAMES):
        raise ValueError("invalid frame count")
    if (not isinstance(stream.fps, Fraction) or stream.fps <= 0
            or stream.fps.numerator > 0xffffffff or stream.fps.denominator > 0xffffffff
            or not Fraction(1, 1000) <= stream.fps <= 1000):
        raise ValueError("invalid constant frame rate")
    if stream.codec not in _CODECS:
        raise ValueError("unknown codec")
    if (not _IDENTIFIER.fullmatch(stream.resource_id)
            or len(stream.resource_id) > MAX_IDENTIFIER):
        raise ValueError("invalid installed resource identifier")
    if not isinstance(stream.native, bytes) or not 1 <= len(stream.native) <= MAX_NATIVE:
        raise ValueError("invalid native stream length")
    if not isinstance(stream.ball_payload, bytes) or not 1 <= len(stream.ball_payload) <= MAX_BALL:
        raise ValueError("invalid BALL payload length")
    _decode_ball(stream)


def _decode_ball(stream: Stream) -> tuple[ball.Track, list[ball.Appearance | None]]:
    # Preflight visibility runs before the legacy decoder allocates their lengths.
    # A corrupt Exp-Golomb run can otherwise request a huge list.
    reader = ball.BitReader(stream.ball_payload)
    if reader.bit():
        reader.get(6)  # position and colour orders
        reader.bit()  # initial visibility
        remaining = stream.frames
        while remaining:
            run = reader.ue() + 1
            if run > remaining:
                raise ValueError("visibility runs overrun frame count")
            remaining -= run
    track, apps = ball.decode_frames(stream.ball_payload, stream.frames, stream.width, stream.height)
    # Re-encoding checks geometry, appearance values, padding and trailing bytes.
    if ball.encode_frames(track, apps, stream.width, stream.height) != stream.ball_payload:
        raise ValueError("noncanonical BALL payload")
    return track, apps


def pack(stream: Stream) -> bytes:
    """Serialize one complete window, including exact native-container bytes."""
    _validate(stream)
    identifier = stream.resource_id.encode("ascii")
    header = _HEADER.pack(MAGIC, VERSION, stream.width, stream.height,
                          stream.fps.numerator, stream.fps.denominator, stream.frames,
                          _CODECS[stream.codec], len(identifier), len(stream.native), len(stream.ball_payload))
    body = header + identifier + stream.native + stream.ball_payload
    return body + hashlib.sha256(body).digest()


def unpack(data: bytes) -> Stream:
    """Parse exactly one window; reject malformed, noncanonical and trailing data."""
    if len(data) < _HEADER.size + _DIGEST_SIZE:
        raise ValueError("truncated PSBL header")
    magic, version, width, height, num, den, frames, codec, n_id, n_native, n_ball = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported PSBL version")
    if not (1 <= n_id <= MAX_IDENTIFIER and 1 <= n_native <= MAX_NATIVE and 1 <= n_ball <= MAX_BALL):
        raise ValueError("invalid PSBL lengths")
    expected = _HEADER.size + n_id + n_native + n_ball + _DIGEST_SIZE
    if len(data) != expected:
        raise ValueError("truncated PSBL or trailing bytes")
    if hashlib.sha256(data[:-_DIGEST_SIZE]).digest() != data[-_DIGEST_SIZE:]:
        raise ValueError("PSBL checksum mismatch")
    names = {v: k for k, v in _CODECS.items()}
    if codec not in names or not den:
        raise ValueError("invalid codec or frame rate")
    if num == 0 or Fraction(num, den).numerator != num or Fraction(num, den).denominator != den:
        raise ValueError("frame rate must be positive and reduced")
    start = _HEADER.size
    try:
        identifier = data[start:start + n_id].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid identifier encoding") from exc
    start += n_id
    stream = Stream(width, height, Fraction(num, den), frames, names[codec], identifier,
                    data[start:start + n_native], data[start + n_native:-_DIGEST_SIZE])
    _validate(stream)
    return stream


def render_from_native_decoded_frames(stream: Stream, frames: np.ndarray) -> np.ndarray:
    """Render decoded RGB uint8 frames using only serialized BALL symbols.

    The caller supplies the native decoder's pixels; this function does not
    establish process isolation or perform native decoding.
    """
    _validate(stream)
    if frames.shape != (stream.frames, stream.height, stream.width, 3) or frames.dtype != np.uint8:
        raise ValueError("native decoded frames differ from declared RGB geometry")
    track, appearances = _decode_ball(stream)
    velocity = ball.velocities(track)
    return np.stack([ball.render(frame, i, track, velocity, appearances) for i, frame in enumerate(frames)])
