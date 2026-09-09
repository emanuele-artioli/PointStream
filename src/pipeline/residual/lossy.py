"""Lossy representations of the corrective residual signal.

Provides:
- Clipped mode (legacy +128 offset; saturates outside [-128, 127])
- Full-range mode (sign-magnitude scaled representation covering [-255, 255] in uint8)
- Auditable spatial gating, background downscaling, and chroma subsampling
- Invert helper with explicit saturation/overflow bounds checking
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

OFFSET = 128
ResidualMode = Literal["clipped", "full_range", "raw_int16"]


def encode_lossy(
    signed: np.ndarray,
    *,
    mode: str = "clipped",
    scale: float = 1.0,
    offset: float = 128.0,
) -> np.ndarray:
    """Encode signed differences into uint8 representation.

    Args:
        signed: (T, H, W, C) signed int16 differences in [-255, 255].
        mode:
            - "clipped": bias by +128 and clip to [0, 255]. Differences outside
              [-128, 127] saturate.
            - "full_range": map [-255, 255] into [0, 255] preserving both
              +/-255 extrema and exact 0.
        scale: Optional scale multiplier when custom scaling is specified.
        offset: Optional offset bias.
    """
    arr = np.asarray(signed, dtype=np.int16)
    if mode == "clipped":
        return np.clip(arr + int(round(offset)), 0, 255).astype(np.uint8)

    if mode == "full_range":
        u = np.full(arr.shape, int(round(offset)), dtype=np.uint8)
        pos = arr > 0
        neg = arr < 0
        u[pos] = np.clip(offset + np.rint(arr[pos] * (127.0 / 255.0)), 128, 255).astype(np.uint8)
        u[neg] = np.clip(offset - np.rint(np.abs(arr[neg]) * (128.0 / 255.0)), 0, 128).astype(np.uint8)
        return u

    if mode == "raw_int16":
        return arr

    raise ValueError(f"Unknown residual representation mode: {mode!r}")


def decode_lossy(
    encoded: np.ndarray,
    *,
    mode: str = "clipped",
    scale: float = 1.0,
    offset: float = 128.0,
) -> np.ndarray:
    """Decode uint8 representation back into signed int16 differences."""
    u = np.asarray(encoded)
    if mode == "raw_int16":
        return u.astype(np.int16)

    if mode == "clipped":
        return u.astype(np.int16) - int(round(offset))

    if mode == "full_range":
        u_int = u.astype(np.int16)
        r = np.zeros(u.shape, dtype=np.int16)
        pos = u_int > int(round(offset))
        neg = u_int < int(round(offset))
        r[pos] = np.rint((u_int[pos] - offset) * (255.0 / 127.0)).astype(np.int16)
        r[neg] = -np.rint((offset - u_int[neg]) * (255.0 / 128.0)).astype(np.int16)
        return r

    raise ValueError(f"Unknown residual representation mode: {mode!r}")


def block_activity_gate(
    residual: np.ndarray,
    *,
    block_size: int,
    threshold: float,
) -> np.ndarray:
    """Zero blocks whose mean absolute residual is below ``threshold``.

    Threshold is in pixel units: 2.0 drops blocks whose mean error is below
    two grey levels. ``block_size <= 1`` or ``threshold <= 0`` is a no-op.
    """
    if block_size <= 1 or threshold <= 0.0:
        return residual
    if residual.ndim != 4:
        raise ValueError(f"residual must be (T, H, W, C); got {residual.shape}.")
    frames, height, width, channels = residual.shape
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    padded = np.pad(residual, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_h, padded_w = padded.shape[1], padded.shape[2]
    n_h, n_w = padded_h // block_size, padded_w // block_size
    blocks = padded.reshape(frames, n_h, block_size, n_w, block_size, channels)
    activity = np.abs(blocks).mean(axis=(2, 4, 5))
    keep = activity >= float(threshold)
    mask = np.repeat(np.repeat(keep[:, :, None, :, None], block_size, axis=2), block_size, axis=4)
    mask = mask.reshape(frames, padded_h, padded_w)
    gated = padded * mask[..., None]
    return gated[:, :height, :width, :]


def downscale_background(
    residual: np.ndarray,
    actor_mask: np.ndarray | None,
    *,
    factor: int,
) -> np.ndarray:
    """Keep object residual at full resolution; coarsen the background.

    ``factor <= 1`` is a no-op. A mask covering every pixel leaves the
    residual untouched — there is no background to coarsen.
    """
    if factor <= 1:
        return residual
    if residual.ndim != 4:
        raise ValueError(f"residual must be (T, H, W, C); got {residual.shape}.")
    frames, height, width, _channels = residual.shape
    if actor_mask is None:
        object_pixels = np.zeros((frames, height, width), dtype=bool)
    else:
        object_pixels = _align_mask(actor_mask, frames, height, width)
    if bool(np.all(object_pixels)):
        return residual

    down_h = max(1, int(math.ceil(height / float(factor))))
    down_w = max(1, int(math.ceil(width / float(factor))))
    coarsened = np.empty_like(residual)
    for index in range(frames):
        small = cv2.resize(
            residual[index].astype(np.float32),
            (down_w, down_h),
            interpolation=cv2.INTER_AREA,
        )
        coarsened[index] = cv2.resize(
            small, (width, height), interpolation=cv2.INTER_NEAREST
        )
    keep = object_pixels[..., None]
    return np.where(keep, residual, coarsened)


def subsample_chroma(frames: np.ndarray, *, pix_fmt: str = "yuv420p") -> np.ndarray:
    """Explicit pixel-domain chroma subsampling for auditable loss modelling."""
    if pix_fmt in {"rgb24", "yuv444p"}:
        return frames
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"frames must be (T, H, W, 3); got {frames.shape}")

    count, height, width, _ = frames.shape
    out = np.empty_like(frames)
    sub_h = max(1, height // 2)
    sub_w = max(1, width // 2)

    for i in range(count):
        ycbcr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        cr_sub = cv2.resize(cr, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        cb_sub = cv2.resize(cb, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        cr_up = cv2.resize(cr_sub, (width, height), interpolation=cv2.INTER_LINEAR)
        cb_up = cv2.resize(cb_sub, (width, height), interpolation=cv2.INTER_LINEAR)
        ycbcr_sub = cv2.merge([y, cr_up, cb_up])
        out[i] = cv2.cvtColor(ycbcr_sub, cv2.COLOR_YCrCb2RGB)
    return out


def invert_clipped_addition(
    delivered: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert delivered = clip(base + residual, 0, 255) with explicit saturation detection.

    Returns:
        (base_estimate, saturation_mask)
        saturation_mask is True where delivered was clamped to 0 or 255,
        indicating that base cannot be uniquely inverted without prior knowledge.
    """
    deliv_int = delivered.astype(np.int16)
    res_int = residual.astype(np.int16)
    diff = deliv_int - res_int
    saturated_high = (deliv_int == 255) & (res_int > 0)
    saturated_low = (deliv_int == 0) & (res_int < 0)
    saturation_mask = saturated_high | saturated_low
    base_estimate = np.clip(diff, 0, 255).astype(np.uint8)
    return base_estimate, saturation_mask


def _align_mask(mask: np.ndarray, frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if array.shape != (height, width):
            raise ValueError(
                f"actor mask shape {array.shape} does not match frame {(height, width)}."
            )
        return np.broadcast_to(array, (frames, height, width))
    if array.shape != (frames, height, width):
        raise ValueError(
            f"actor mask shape {array.shape} does not match clip {(frames, height, width)}."
        )
    return array
