"""Experimental lossless binary shape coder (H5p 4d), not MPEG-4 compliant.

Adaptive binary arithmetic coding with five causal spatial pixels and five
motion-predicted temporal pixels. Motion is blockwise integer translation,
sent as an LZMA stream; the decoder never sees the encoder's target masks.
"""
from __future__ import annotations

import lzma
import struct

import numpy as np

HEADER = struct.Struct('>4sHHHBII')
HALF, QUARTER, LIMIT = 1 << 31, 1 << 30, (1 << 32) - 1


def prediction(previous: np.ndarray, vectors: np.ndarray, block: int = 32) -> np.ndarray:
    h, w = previous.shape
    yy, xx = np.indices((h, w))
    dy = vectors[yy // block, xx // block, 0].astype(int)
    dx = vectors[yy // block, xx // block, 1].astype(int)
    sy, sx = yy + dy, xx + dx
    valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
    return previous[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)] * valid


def motion(current: np.ndarray, previous: np.ndarray, block: int = 32) -> np.ndarray:
    """Full integer search ±4 reduced pixels; vectors are transmitted, not free."""
    import cv2
    h, w = current.shape
    result = np.zeros((-(-h // block), -(-w // block), 2), np.int8)
    padded = np.pad(previous.astype(np.float32), 4)
    for y in range(0, h, block):
        for x in range(0, w, block):
            target = current[y:y + block, x:x + block].astype(np.float32)
            th, tw = target.shape
            area = padded[y:y + th + 8, x:x + tw + 8]
            costs = cv2.matchTemplate(area, target, cv2.TM_SQDIFF)
            best = np.unravel_index(int(costs.argmin()), costs.shape)
            # Prefer no motion on ties, including all-background blocks.
            if costs[4, 4] <= costs[best] + 0.01:
                best = (4, 4)
            result[y // block, x // block] = (best[0] - 4, best[1] - 4)
    return result


def contexts(frame: np.ndarray, predicted: np.ndarray):
    """Causal raster contexts. Both arrays have a two-pixel zero border."""
    h, w = frame.shape[0] - 4, frame.shape[1] - 4
    for y in range(2, h + 2):
        for x in range(2, w + 2):
            c = (int(frame[y, x - 1]) | int(frame[y, x - 2]) << 1 |
                 int(frame[y - 1, x - 1]) << 2 | int(frame[y - 1, x]) << 3 |
                 int(frame[y - 1, x + 1]) << 4 | int(predicted[y, x]) << 5 |
                 int(predicted[y, x - 1]) << 6 | int(predicted[y, x + 1]) << 7 |
                 int(predicted[y - 1, x]) << 8 | int(predicted[y + 1, x]) << 9)
            yield y, x, c


def encode(masks: np.ndarray, mode: int = 2) -> bytes:
    """mode 0: intra, 1: previous frame, 2: block motion; exact binary output."""
    if masks.ndim != 3 or not all(0 < d <= 65535 for d in masks.shape) or mode not in (0, 1, 2):
        raise ValueError('invalid binary shape or mode')
    masks = (masks > 0).astype(np.uint8)
    n, h, w = masks.shape
    vectors = np.zeros((n, -(-h // 32), -(-w // 32), 2), np.int8)
    bits: list[int] = []
    low, high, pending = 0, LIMIT, 0
    counts = [[1, 1] for _ in range(1024)]
    def emit(bit: int) -> None:
        nonlocal pending
        bits.append(bit)
        bits.extend([1 - bit] * pending)
        pending = 0
    previous = np.zeros((h, w), np.uint8)
    for t, mask in enumerate(masks):
        if mode == 2 and t:
            vectors[t] = motion(mask, previous)
        pred = prediction(previous, vectors[t]) if mode else np.zeros_like(previous)
        frame, padded = np.pad(mask, 2), np.pad(pred, 2)
        for y, x, c in contexts(frame, padded):
            a, b = counts[c]
            split = low + (high - low + 1) * a // (a + b) - 1
            bit = int(frame[y, x])
            if bit:
                low = split + 1
            else:
                high = split
            while True:
                if high < HALF:
                    emit(0)
                elif low >= HALF:
                    emit(1)
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < 3 * QUARTER:
                    pending += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low, high = low * 2, high * 2 + 1
            counts[c][bit] += 1
            if sum(counts[c]) >= 4096:
                counts[c] = [max(1, (v + 1) // 2) for v in counts[c]]
        previous = mask
    pending += 1
    emit(int(low >= QUARTER))
    mv = lzma.compress(vectors.tobytes()) if mode == 2 else b''
    raw = np.packbits(np.asarray(bits, np.uint8)).tobytes()
    return HEADER.pack(b'SHP1', n, h, w, mode, len(mv), len(bits)) + mv + raw


def decode(payload: bytes) -> np.ndarray:
    magic, n, h, w, mode, mv_size, bit_count = HEADER.unpack_from(payload)
    if magic != b'SHP1' or mode not in (0, 1, 2) or not min(n, h, w):
        raise ValueError('invalid shape header')
    if len(payload) != HEADER.size + mv_size + -(-bit_count // 8):
        raise ValueError('truncated shape payload')
    shape = (n, -(-h // 32), -(-w // 32), 2)
    vectors = np.zeros(shape, np.int8)
    if mode == 2:
        vectors = np.frombuffer(lzma.decompress(payload[HEADER.size:HEADER.size + mv_size]), np.int8).reshape(shape)
    bits = np.unpackbits(np.frombuffer(payload[HEADER.size + mv_size:], np.uint8)).tolist()
    offset = 0
    def read() -> int:
        nonlocal offset
        bit = bits[offset] if offset < bit_count else 0
        offset += 1
        return int(bit)
    value = 0
    for _ in range(32):
        value = value * 2 + read()
    low, high = 0, LIMIT
    counts = [[1, 1] for _ in range(1024)]
    masks = np.zeros((n, h, w), np.uint8)
    previous = np.zeros((h, w), np.uint8)
    for t in range(n):
        pred = prediction(previous, vectors[t]) if mode else np.zeros_like(previous)
        frame = np.zeros((h + 4, w + 4), np.uint8)
        for y, x, c in contexts(frame, np.pad(pred, 2)):
            a, b = counts[c]
            split = low + (high - low + 1) * a // (a + b) - 1
            bit = int(value > split)
            frame[y, x] = bit
            if bit:
                low = split + 1
            else:
                high = split
            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    value -= HALF
                elif low >= QUARTER and high < 3 * QUARTER:
                    low -= QUARTER
                    high -= QUARTER
                    value -= QUARTER
                else:
                    break
                low, high, value = low * 2, high * 2 + 1, value * 2 + read()
            counts[c][bit] += 1
            if sum(counts[c]) >= 4096:
                counts[c] = [max(1, (v + 1) // 2) for v in counts[c]]
        masks[t] = frame[2:-2, 2:-2]
        previous = masks[t]
    return masks
