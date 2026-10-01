"""PSR1 binary mask wire primitives, byte-compatible with retained E06 packs.

This module depends only on NumPy and the standard library; container packing
and experimental evaluation are deliberately outside this installed codec.
"""
from __future__ import annotations

import struct
import numpy as np

RLE_MAGIC = b"PSR1"
RLE_VERSION = 1
MODE_INTRA = 0
MODE_XOR_KEY = 1
KEY_PERIOD = 4

_U16 = struct.Struct("<H")


def _runs(flat: np.ndarray) -> list[tuple[int, int]]:
    values = np.ascontiguousarray(flat, dtype=np.uint8).reshape(-1)
    out: list[tuple[int, int]] = []
    index = 0
    n = int(values.size)
    while index < n:
        value = int(values[index])
        stop = index + 1
        while stop < n and int(values[stop]) == value:
            stop += 1
        length = stop - index
        while length > 0:
            chunk = min(length, 65535)
            out.append((value, chunk))
            length -= chunk
        index = stop
    return out


def encode_rle_frame(mask: np.ndarray) -> bytes:
    binary = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    runs = _runs(binary)
    if len(runs) > 65535:
        raise ValueError("too many RLE runs")
    parts = bytearray(_U16.pack(len(runs)))
    for value, count in runs:
        parts.append(value & 0xFF)
        parts += _U16.pack(count)
    return bytes(parts)


def decode_rle_frame(
    blob: bytes, *, height: int, width: int, offset: int = 0
) -> tuple[np.ndarray, int]:
    if offset + 2 > len(blob):
        raise ValueError("truncated mask RLE")
    n_runs = int(_U16.unpack_from(blob, offset)[0])
    cursor = offset + 2
    expected = height * width
    pixels = np.empty(expected, dtype=np.uint8)
    filled = 0
    for _ in range(n_runs):
        if cursor + 3 > len(blob):
            raise ValueError("truncated mask RLE run")
        value = int(blob[cursor])
        count = int(_U16.unpack_from(blob, cursor + 1)[0])
        cursor += 3
        if value not in (0, 1) or count < 1:
            raise ValueError("corrupt mask RLE run")
        if filled + count > expected:
            raise ValueError("mask RLE overflow")
        pixels[filled : filled + count] = value
        filled += count
    if filled != expected:
        raise ValueError("mask RLE underfill")
    return pixels.reshape(height, width), cursor


def encode_mask_stack(masks: np.ndarray, *, mode: int, key_period: int = KEY_PERIOD) -> bytes:
    stack = np.ascontiguousarray(masks != 0, dtype=np.uint8)
    if stack.ndim != 3:
        raise ValueError(f"mask stack must be (T, H, W); got {stack.shape}")
    n_frames, height, width = (int(dim) for dim in stack.shape)
    if mode not in (MODE_INTRA, MODE_XOR_KEY):
        raise ValueError(f"unsupported RLE mode {mode}")
    parts = bytearray(RLE_MAGIC)
    parts.append(RLE_VERSION)
    parts.append(mode)
    parts += _U16.pack(height)
    parts += _U16.pack(width)
    parts += _U16.pack(n_frames)
    parts.append(key_period & 0xFF)
    previous: np.ndarray | None = None
    for index, frame in enumerate(stack):
        intra = mode == MODE_INTRA or index % key_period == 0 or previous is None
        if intra:
            coded = frame
        else:
            assert previous is not None
            coded = np.bitwise_xor(frame, previous)
        parts.append(1 if intra else 0)
        parts += encode_rle_frame(coded)
        previous = frame
    return bytes(parts)


def decode_mask_stack(blob: bytes) -> np.ndarray:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("mask RLE payload must be bytes")
    data = bytes(blob)
    if len(data) < 13 or data[:4] != RLE_MAGIC:
        raise ValueError("corrupt mask RLE magic")
    version = int(data[4])
    mode = int(data[5])
    if version != RLE_VERSION:
        raise ValueError(f"unsupported mask RLE version {version}")
    if mode not in (MODE_INTRA, MODE_XOR_KEY):
        raise ValueError(f"unsupported mask RLE mode {mode}")
    height = int(_U16.unpack_from(data, 6)[0])
    width = int(_U16.unpack_from(data, 8)[0])
    n_frames = int(_U16.unpack_from(data, 10)[0])
    key_period = int(data[12])
    if height < 1 or width < 1 or n_frames < 1 or key_period < 1:
        raise ValueError("corrupt mask RLE header")
    cursor = 13
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    previous: np.ndarray | None = None
    for index in range(n_frames):
        if cursor >= len(data):
            raise ValueError("truncated mask RLE stack")
        intra = int(data[cursor]) == 1
        cursor += 1
        coded, cursor = decode_rle_frame(data, height=height, width=width, offset=cursor)
        if intra:
            frame = coded
        else:
            if previous is None:
                raise ValueError("XOR-delta without a previous mask")
            frame = np.bitwise_xor(coded, previous)
        frames[index] = frame
        previous = frame
    if cursor != len(data):
        raise ValueError("trailing bytes in mask RLE")
    return frames


