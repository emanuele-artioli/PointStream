"""PointStream's compositor and the foreground stream's input (PLAN H5, interface agreed in G6).

The background decodes the whole frame with F filled. The foreground stream codes F and a ring around it; the
client upsamples it, runs each class's enhancer slot, and blends it over the background: alpha is 1 on F and
falls linearly to 0 over `RING` px outside it (Euclidean distance), so the background shows elsewhere.

The foreground's encoder keeps the source only on K = F dilated by the ring plus a guard for the upsampling
filter (`keep_radius`) and replaces the rest by a smooth push-pull extrapolation of K's own pixels
(`push_pull`), as MPEG-4's arbitrary-shape coding padded outside the shape: the codec spends little outside
F and no edge forms at K's border.

Class maps hold 0 (background), `HAND` and `OBJECT`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RING = 8
HAND, OBJECT = 1, 2
#: An enhancer slot: (upsampled RGB frames (n, h, w, 3) uint8, its class's pixels (n, h, w) bool) -> frames.
Enhancer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def keep_radius(scale: int, ring: int = RING) -> int:
    """K's dilation of F in pixels at full size: the ring, plus two coded pixels (a bicubic tap's reach) of
    guard for the upsampling."""
    return ring + 2 * scale


def disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """A (h, w) or (n, h, w) boolean mask dilated by a Euclidean disk."""
    import cv2

    if radius <= 0:
        return mask.astype(bool)
    kernel = disk(radius)
    if mask.ndim == 2:
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return np.stack([cv2.dilate(m.astype(np.uint8), kernel) for m in mask]).astype(bool)


def push_pull(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """``image`` (h, w, c) with its unknown pixels replaced by a smooth extrapolation of the known ones: area
    averages of the known pixels pulled down a pyramid, then pushed back up, each level filling only what its
    finer level lacks. Known pixels are kept exactly. An image with no known pixel is returned unchanged."""
    import cv2

    known = known.astype(bool)
    if known.all() or not known.any():
        return image.copy()
    weight = known.astype(np.float32)
    colour = image.astype(np.float32) * weight[..., None]
    levels = [(colour, weight)]
    while min(weight.shape) > 1:
        h, w = weight.shape
        ph, pw = h + h % 2, w + w % 2
        c = np.pad(colour, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
        wt = np.pad(weight, ((0, ph - h), (0, pw - w)), mode="edge")
        colour = c.reshape(ph // 2, 2, pw // 2, 2, -1).mean(axis=(1, 3))
        weight = wt.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))
        levels.append((colour, weight))
    colour, weight = levels[-1]
    value = colour / np.maximum(weight, 1e-12)[..., None]
    for colour, weight in reversed(levels[:-1]):
        h, w = weight.shape
        up = cv2.resize(value, (w, h), interpolation=cv2.INTER_LINEAR).reshape(h, w, -1)
        coverage = np.clip(weight, 0.0, 1.0)[..., None]
        mean = colour / np.maximum(weight, 1e-12)[..., None]
        value = np.where(coverage > 0, coverage * mean + (1 - coverage) * up, up)
    out = np.where(known[..., None], image.astype(np.float32), value.reshape(image.shape))
    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    return out.astype(image.dtype)


def foreground_input(frames: np.ndarray, foreground: np.ndarray, scale: int, size: tuple[int, int],
                     ring: int = RING) -> np.ndarray:
    """The foreground stream's input: each RGB frame (n, h, w, 3) area-downscaled to ``size`` (w, h), with
    everything outside K (`keep_radius`) replaced by `push_pull`."""
    import cv2

    radius = keep_radius(scale, ring)
    out = np.empty((len(frames), size[1], size[0], 3), np.uint8)
    for t, (frame, f) in enumerate(zip(frames, foreground)):
        keep = dilate(f, radius)
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if frame.shape[1::-1] != size else frame
        known = cv2.resize(keep.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        out[t] = push_pull(small, known)
    return out


def alpha_and_classes(classes: np.ndarray, ring: int = RING) -> tuple[np.ndarray, np.ndarray]:
    """For one (h, w) class map: the blend weight (1 on F, 1 − d/(ring + 1) at distance d outside, 0 beyond
    the ring) and the class of the nearest F pixel at every pixel within the ring (0 elsewhere)."""
    import cv2

    f = classes > 0
    if not f.any():
        return np.zeros(classes.shape, np.float32), np.zeros_like(classes)
    distance, labels = cv2.distanceTransformWithLabels((~f).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    alpha = np.clip(1.0 - distance / (ring + 1.0), 0.0, 1.0).astype(np.float32)
    lut = np.zeros(int(labels.max()) + 1, classes.dtype)
    lut[labels[f]] = classes[f]
    nearest = np.where(alpha > 0, lut[labels], 0).astype(classes.dtype)
    return alpha, nearest


def apply_slots(upsampled: np.ndarray, nearest: np.ndarray, slots: dict[int, Enhancer]) -> np.ndarray:
    """Each class's enhancer on the upsampled foreground, taken on the pixels nearest that class (F and its
    ring); a class without a slot keeps the upsampled pixels (the identity)."""
    out = upsampled.copy()
    for cls, enhancer in slots.items():
        where = nearest == cls
        if where.any():
            enhanced = enhancer(upsampled, where)
            out[where] = enhanced[where]
    return out


def composite(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """α·fg + (1 − α)·bg per pixel, rounded to uint8; (n, h, w, 3) frames and (n, h, w) weights."""
    a = alpha[..., None].astype(np.float32)
    out = a * foreground.astype(np.float32) + (1.0 - a) * background.astype(np.float32)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
