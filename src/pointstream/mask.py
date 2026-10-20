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


# ------------------------------------------- F as a video (H5p 2): a smooth plane a motion-compensated codec codes

#: Signed distance to F's boundary, in coded pixels, clipped to ±`SDF_CLIP` and stored as 128 + `SDF_GAIN`·d
#: (inside positive), so the client's F̂ is the bilinear upsample ≥ 128: a ramp the codec codes cheaply instead of
#: a step, and the boundary at sub-pixel precision from the zero crossing.
SDF_CLIP = 8
SDF_GAIN = 15
NEUTRAL = 128  # chroma outside F; hands 255, objects 0


def signed_distance(f: np.ndarray) -> np.ndarray:
    """Signed distance (pixels, positive inside) from each pixel to the boundary of the binary frame ``f``."""
    import cv2

    inside = cv2.distanceTransform(f.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~f).astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(f, inside - 0.5, 0.5 - outside).astype(np.float32)


def padded_size(width: int, height: int, scale: int) -> tuple[int, int]:
    """The coded frame for 1/``scale``: the reduced size rounded up to a multiple of 8 (the encoder's 4:2:0)."""
    rw, rh = reduced_size(width, height, scale)
    return -(-rw // 8) * 8, -(-rh // 8) * 8


def video_planes(classes: np.ndarray, scale: int, representation: str) -> np.ndarray:
    """(n, h, w) class maps to raw 8-bit 4:2:0 frames at 1/``scale`` (padded by edge replication), flattened per
    frame as Y, U, V. ``sdf``: Y is F's clipped signed distance; ``binary``: Y is 255 on F's reduced cells, 0 off
    them. U: 255 on hand cells, 0 on object cells, 128 elsewhere (at the chroma's half size); V: 128."""
    import cv2

    n, h, w = classes.shape
    rw, rh = reduced_size(w, h, scale)
    pw, ph = padded_size(w, h, scale)
    out = np.empty((n, ph * pw * 3 // 2), np.uint8)
    chroma = reduce(classes, 2 * scale)
    for t in range(n):
        if representation == "sdf":
            d = signed_distance(classes[t] > 0)
            small = cv2.resize(d, (rw, rh), interpolation=cv2.INTER_AREA) / scale
            y = np.clip(NEUTRAL + SDF_GAIN * np.clip(small, -SDF_CLIP, SDF_CLIP), 0, 255)
        elif representation == "binary":
            y = np.where(reduce(classes[t:t + 1], scale)[0] > 0, 255, 0)
        else:
            raise ValueError(f"unknown representation {representation!r}")
        y = np.pad(np.round(y).astype(np.uint8), ((0, ph - rh), (0, pw - rw)), mode="edge")
        c = np.where(chroma[t] == HAND, 255, np.where(chroma[t] == OBJECT, 0, NEUTRAL)).astype(np.uint8)
        c = np.pad(c, ((0, ph // 2 - c.shape[0]), (0, pw // 2 - c.shape[1])), mode="edge")
        out[t] = np.concatenate([y.ravel(), c.ravel(), np.full(c.size, NEUTRAL, np.uint8)])
    return out


def video_classes(planes: np.ndarray, scale: int, size: tuple[int, int]) -> np.ndarray:
    """The client's class maps at ``size`` (w, h) from decoded 4:2:0 frames: F̂ where the bilinear upsample of Y
    is ≥ 128 (both representations), a hand where U's is ≥ 128, else an object."""
    import cv2

    w, h = size
    rw, rh = reduced_size(w, h, scale)
    pw, ph = padded_size(w, h, scale)
    out = np.zeros((len(planes), h, w), np.uint8)
    for t, frame in enumerate(planes):
        y = frame[:pw * ph].reshape(ph, pw)[:rh, :rw].astype(np.float32)
        u = frame[pw * ph:pw * ph + pw * ph // 4].reshape(ph // 2, pw // 2)
        u = u[:-(-rh // 2), :-(-rw // 2)].astype(np.float32)
        f = cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR) >= NEUTRAL
        hand = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) >= NEUTRAL
        out[t][f] = OBJECT
        out[t][f & hand] = HAND
    return out


def accuracy(truth: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    """F̂ against F over a clip: IoU of the union, F's pixels wrong as a share of F, the mean symmetric boundary
    distance (pixels), and the share of F ∩ F̂ given the right class."""
    import cv2

    inter = union = wrong = 0
    distances: list[float] = []
    right = both = 0
    for a, b in zip(truth, decoded):
        fa, fb = a > 0, b > 0
        inter += int((fa & fb).sum())
        union += int((fa | fb).sum())
        wrong += int((fa ^ fb).sum())
        common = fa & fb
        both += int(common.sum())
        right += int((a[common] == b[common]).sum())
        edges = [cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                 for f in (fa, fb)]
        if edges[0].any() and edges[1].any():
            for x, y in ((0, 1), (1, 0)):
                field = cv2.distanceTransform((~edges[y]).astype(np.uint8), cv2.DIST_L2, 5)
                distances.append(float(field[edges[x]].mean()))
    area = int((truth > 0).sum())
    return {"iou": inter / union if union else 1.0, "error_share": wrong / area if area else 0.0,
            "boundary_px": float(np.mean(distances)) if distances else 0.0,
            "class_accuracy": right / both if both else 1.0}
