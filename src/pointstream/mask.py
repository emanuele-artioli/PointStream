"""F's class map for PointStream's decoder, coded at a reduced size (docs/experiments.md, H5m).

The encoder reduces each frame's class map (0 background, `HAND`, `OBJECT`) to 1/``scale`` of its size by area: a
cell is in F when at least half of it is F, and a hand when its hand pixels outnumber its object pixels. The two
bit planes (F, hand) of every frame are XOR-ed with the previous frame's and compressed with lzma, losslessly.
The client decodes them, upsamples F bilinearly and thresholds it at 0.5, and gives each pixel of F̂ the class
of the nearest reduced cell in F.

Payload: scale u8, frames u16, reduced width u16, reduced height u16, then the lzma stream.
"""

from __future__ import annotations

import lzma
import struct

import numpy as np

from src.pointstream.composite import HAND, OBJECT

HEADER = struct.Struct(">BHHH")


def reduced_size(width: int, height: int, scale: int) -> tuple[int, int]:
    return -(-width // scale), -(-height // scale)


def reduce(classes: np.ndarray, scale: int) -> np.ndarray:
    """(n, h, w) class maps to (n, ⌈h/scale⌉, ⌈w/scale⌉) by area (edges padded with background)."""
    if scale == 1:
        return classes.astype(np.uint8)
    n, h, w = classes.shape
    rw, rh = reduced_size(w, h, scale)
    padded = np.zeros((n, rh * scale, rw * scale), np.uint8)
    padded[:, :h, :w] = classes
    cells = padded.reshape(n, rh, scale, rw, scale)
    f = (cells > 0).mean(axis=(2, 4))
    hand = (cells == HAND).mean(axis=(2, 4))
    obj = (cells == OBJECT).mean(axis=(2, 4))
    out = np.zeros((n, rh, rw), np.uint8)
    out[f >= 0.5] = OBJECT
    out[(f >= 0.5) & (hand > obj)] = HAND
    return out


def encode(reduced: np.ndarray, scale: int) -> bytes:
    """The payload of a MASK segment."""
    n, rh, rw = reduced.shape
    planes = np.stack([np.packbits(reduced > 0, axis=-1), np.packbits(reduced == HAND, axis=-1)], axis=1)
    delta = planes.copy()
    delta[1:] ^= planes[:-1]
    return HEADER.pack(scale, n, rw, rh) + lzma.compress(delta.tobytes(), preset=9 | lzma.PRESET_EXTREME)


def decode(payload: bytes) -> tuple[np.ndarray, int]:
    """The reduced class maps and the scale."""
    scale, n, rw, rh = HEADER.unpack_from(payload)
    packed_w = -(-rw // 8)
    delta = np.frombuffer(lzma.decompress(payload[HEADER.size:]), np.uint8).reshape(n, 2, rh, packed_w)
    planes = np.bitwise_xor.accumulate(delta, axis=0)
    f = np.unpackbits(planes[:, 0], axis=-1)[..., :rw].astype(bool)
    hand = np.unpackbits(planes[:, 1], axis=-1)[..., :rw].astype(bool)
    out = np.zeros((n, rh, rw), np.uint8)
    out[f] = OBJECT
    out[hand & f] = HAND
    return out, scale


def upsample(reduced: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The client's class maps at ``size`` (w, h): F bilinear and thresholded at 0.5, each F pixel the class of
    the nearest reduced cell in F."""
    import cv2

    w, h = size
    out = np.zeros((len(reduced), h, w), np.uint8)
    for t, frame in enumerate(reduced):
        if frame.shape == (h, w):
            out[t] = frame
            continue
        f = cv2.resize((frame > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
        if not f.any():
            continue
        labels = np.where(frame > 0, frame, 0).astype(np.uint8)
        if (labels > 0).any():  # nearest reduced F cell's class, through a distance transform on the reduced grid
            _, idx = cv2.distanceTransformWithLabels((labels == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            lut = np.zeros(int(idx.max()) + 1, np.uint8)
            lut[idx[labels > 0]] = labels[labels > 0]
            nearest = cv2.resize(lut[idx], (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            nearest = np.full((h, w), OBJECT, np.uint8)
        out[t][f] = np.where(nearest[f] > 0, nearest[f], OBJECT)
    return out
