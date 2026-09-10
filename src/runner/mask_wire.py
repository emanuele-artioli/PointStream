"""Lossless compact mask wire for the independent client envelope.

Wire representation (schema version 1)
-------------------------------------
Each mask is one binary blob, stored as a uint8 vector in the NumPy ``npz``
envelope under ``mask_{i}``. The blob is little-endian, self-describing, and
does not depend on encoder-side source pixels.

Header (10 bytes + ``ndim * 4`` shape words), all multi-byte fields little-endian::

    magic[4]     = b"PSM1"
    version u8   = 1
    ndim    u8   = 2 or 3
    dtype   u8   = 0  (packed binary, reconstructed as uint8 0/1)
    endian  u8   = 0  (little)
    packing u8   = 0  (LSB-first within each byte, row-major in the rectangle)
    reserved u8  = 0
    shape   u32 * ndim
    n_spans u32

Then ``n_spans`` records, omitting empty frames::

    frame_index u32   (0 when ndim == 2)
    y0 u32, x0 u32, y1 u32, x1 u32   half-open bounding rectangle
    n_packed u32
    packed bits of the rectangle (``ceil((y1-y0)*(x1-x0)/8)`` bytes)

Pixels outside every span are 0. This is exact for binary masks: the
rectangle carries every set bit, not a bbox approximation.
"""

from __future__ import annotations

import struct
from typing import Any, Final

import numpy as np

MAGIC: Final[bytes] = b"PSM1"
SCHEMA_VERSION: Final[int] = 1
DTYPE_PACKED_BINARY: Final[int] = 0
ENDIAN_LITTLE: Final[int] = 0
PACKING_LSB_ROW_MAJOR: Final[int] = 0

ENCODING_NAME: Final[str] = "sparse-rect-packed-bits"
DTYPE_NAME: Final[str] = "uint8"
ENDIAN_NAME: Final[str] = "little"
PACKING_NAME: Final[str] = "lsb-first-row-major"

_U32 = struct.Struct("<I")
_HEADER = struct.Struct("<4sBBBBBB")


def wire_declaration() -> dict[str, Any]:
    """JSON metadata declaring the mask blob format."""
    return {
        "schema_version": SCHEMA_VERSION,
        "encoding": ENCODING_NAME,
        "dtype": DTYPE_NAME,
        "endian": ENDIAN_NAME,
        "packing": PACKING_NAME,
        "empty_frames": "omitted",
    }


def encode_mask(mask: np.ndarray) -> bytes:
    """Encode a 2-D or 3-D mask as sparse rectangles plus packed bits."""
    array = np.asarray(mask)
    if array.ndim not in (2, 3):
        raise ValueError(f"mask must be (H, W) or (T, H, W); got shape {array.shape!r}")
    if any(int(dim) < 0 for dim in array.shape):
        raise ValueError(f"mask shape must be non-negative; got {array.shape!r}")
    binary = np.ascontiguousarray(array != 0, dtype=np.uint8)
    parts = bytearray()
    parts += _HEADER.pack(
        MAGIC,
        SCHEMA_VERSION,
        int(binary.ndim),
        DTYPE_PACKED_BINARY,
        ENDIAN_LITTLE,
        PACKING_LSB_ROW_MAJOR,
        0,
    )
    parts += struct.pack("<" + "I" * binary.ndim, *[int(dim) for dim in binary.shape])

    spans: list[bytes] = []
    if binary.ndim == 2:
        packed = _encode_span(0, binary)
        if packed is not None:
            spans.append(packed)
    else:
        for frame_index, frame in enumerate(binary):
            packed = _encode_span(int(frame_index), frame)
            if packed is not None:
                spans.append(packed)
    parts += _U32.pack(len(spans))
    for span in spans:
        parts += span
    return bytes(parts)


def decode_mask(blob: bytes) -> np.ndarray:
    """Decode a mask blob to uint8 0/1. Corrupt or truncated input raises."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("mask payload must be bytes")
    reader = _Reader(bytes(blob))
    magic, version, ndim, dtype, endian, packing, reserved = _HEADER.unpack(reader.take(_HEADER.size))
    if magic != MAGIC:
        raise ValueError("corrupt mask payload: bad magic")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported mask wire schema version {version}")
    if ndim not in (2, 3):
        raise ValueError(f"unsupported mask ndim {ndim}")
    if dtype != DTYPE_PACKED_BINARY:
        raise ValueError(f"unsupported mask dtype code {dtype}")
    if endian != ENDIAN_LITTLE:
        raise ValueError(f"unsupported mask endian code {endian}")
    if packing != PACKING_LSB_ROW_MAJOR:
        raise ValueError(f"unsupported mask packing code {packing}")
    if reserved != 0:
        raise ValueError("corrupt mask payload: reserved field must be 0")

    shape = tuple(reader.u32() for _ in range(ndim))
    if any(int(dim) < 0 for dim in shape):
        raise ValueError(f"corrupt mask payload: negative shape {shape}")
    height = int(shape[-2]) if ndim == 2 else int(shape[1])
    width = int(shape[-1]) if ndim == 2 else int(shape[2])
    n_frames = 1 if ndim == 2 else int(shape[0])
    out = np.zeros(shape, dtype=np.uint8)
    n_spans = reader.u32()
    if n_spans > n_frames:
        raise ValueError("corrupt mask payload: more spans than frames")
    seen: set[int] = set()
    for _ in range(n_spans):
        frame_index = reader.u32()
        y0, x0, y1, x1 = reader.u32(), reader.u32(), reader.u32(), reader.u32()
        n_packed = reader.u32()
        packed = reader.take(n_packed)
        if ndim == 2 and frame_index != 0:
            raise ValueError("corrupt mask payload: 2-D frame_index must be 0")
        if frame_index in seen or frame_index < 0 or frame_index >= n_frames:
            raise ValueError("corrupt mask payload: invalid or duplicate frame_index")
        seen.add(frame_index)
        rect = _unpack_rect(packed, y0=y0, x0=x0, y1=y1, x1=x1, height=height, width=width)
        if ndim == 2:
            out[y0:y1, x0:x1] = rect
        else:
            out[frame_index, y0:y1, x0:x1] = rect
    if reader.remaining:
        raise ValueError("corrupt mask payload: trailing bytes")
    return out


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self._offset + count > len(self._data):
            raise ValueError("truncated mask payload")
        start = self._offset
        self._offset += count
        return self._data[start : self._offset]

    def u32(self) -> int:
        return int(_U32.unpack(self.take(_U32.size))[0])

    @property
    def remaining(self) -> int:
        return len(self._data) - self._offset


def _encode_span(frame_index: int, frame: np.ndarray) -> bytes | None:
    rect = _nonzero_rect(frame)
    if rect is None:
        return None
    y0, x0, y1, x1 = rect
    packed = _pack_bits(frame[y0:y1, x0:x1])
    return (
        _U32.pack(frame_index)
        + _U32.pack(y0)
        + _U32.pack(x0)
        + _U32.pack(y1)
        + _U32.pack(x1)
        + _U32.pack(len(packed))
        + packed
    )


def _nonzero_rect(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(frame != 0, axis=1)
    cols = np.any(frame != 0, axis=0)
    if not bool(rows.any()):
        return None
    y_idx = np.flatnonzero(rows)
    x_idx = np.flatnonzero(cols)
    return int(y_idx[0]), int(x_idx[0]), int(y_idx[-1]) + 1, int(x_idx[-1]) + 1


def _pack_bits(rect: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(rect, dtype=np.uint8).reshape(-1)
    return np.packbits(flat, bitorder="little").tobytes()


def _unpack_rect(
    packed: bytes,
    *,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    height: int,
    width: int,
) -> np.ndarray:
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("corrupt mask payload: rectangle out of bounds")
    n_bits = (y1 - y0) * (x1 - x0)
    expected = (n_bits + 7) // 8
    if len(packed) != expected:
        raise ValueError("corrupt mask payload: packed length mismatch")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    if bits.size < n_bits:
        raise ValueError("truncated mask payload")
    return bits[:n_bits].reshape(y1 - y0, x1 - x0).astype(np.uint8, copy=False)


__all__ = [
    "DTYPE_NAME",
    "ENCODING_NAME",
    "ENDIAN_NAME",
    "MAGIC",
    "PACKING_NAME",
    "SCHEMA_VERSION",
    "decode_mask",
    "encode_mask",
    "wire_declaration",
]
