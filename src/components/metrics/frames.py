"""Clip layout shared by every metric backend.

A clip is ``(T, H, W, 3)`` RGB. A lone frame is promoted to ``T=1``. Values are
float64 in ``[0, peak]`` with ``peak=255``, so PSNR and SSIM share one
conversion and a uint8/float mismatch cannot silently rescale one side.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Clip = NDArray[np.float64]


def to_clip(frames: np.ndarray) -> Clip:
    """Promote a frame or clip to ``(T, H, W, 3)`` float64 in ``[0, 255]``."""
    array = np.asarray(frames)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim == 3:
        if array.shape[-1] in (1, 3):
            array = array[None, ...]
        else:
            array = array[..., None]
    if array.ndim != 4:
        raise ValueError(
            f"expected a frame (H, W, C) or clip (T, H, W, C), got shape {array.shape}"
        )
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"expected 1 or 3 channels, got shape {array.shape}")

    values = array.astype(np.float64, copy=False)
    if np.issubdtype(array.dtype, np.floating) and (array.size == 0 or array.max() <= 1.0):
        values = values * 255.0
    return np.clip(values, 0.0, 255.0)


def paired(reference: np.ndarray, predicted: np.ndarray) -> tuple[Clip, Clip]:
    """Two clips of identical shape, or a mismatch that must not be scored."""
    ref = to_clip(reference)
    pred = to_clip(predicted)
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and predicted clips must share shape; got {ref.shape} vs {pred.shape}"
        )
    return ref, pred
