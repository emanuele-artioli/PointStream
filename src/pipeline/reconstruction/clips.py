"""Clip layout for reconstruction and residual.

A clip is ``(T, H, W, 3)`` uint8. A lone frame is promoted to ``T=1``. Shapes
are checked, never broadcast: a silent reshape is how a wrong reconstruction
scores as if it were right.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Clip = NDArray[np.uint8]


def as_clip(frames: np.ndarray, *, path: str = "frames") -> Clip:
    """Promote a frame or clip to ``(T, H, W, 3)`` uint8.

    Raises:
        ValueError: If the array cannot be a 3-channel clip.
    """
    array = np.asarray(frames)
    if array.size == 0:
        raise ValueError(f"{path} is empty; a reconstruction of nothing cannot be scored.")
    if array.ndim == 3 and array.shape[-1] == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"{path} must be a frame (H, W, 3) or clip (T, H, W, 3); got shape {array.shape}."
        )
    if array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"{path} has a zero extent: {array.shape}.")
    return np.asarray(array, dtype=np.uint8)


def require_same_shape(left: np.ndarray, right: np.ndarray, *, path: str) -> None:
    """Refuse a pair that would otherwise be compared by broadcasting."""
    if left.shape != right.shape:
        raise ValueError(
            f"{path}: shapes {left.shape} and {right.shape} must match. "
            "Resampling here would hide an upstream size bug."
        )
