"""Causal, one-step smoothing of generated player crops for the W6b pilot."""
from __future__ import annotations

import numpy as np


def smooth_crops(current: np.ndarray, previous: np.ndarray | None, weights: tuple[float, ...]) -> list[np.ndarray]:
    """Warp the previous raw crop once, then blend each fixed profile independently."""
    import cv2

    from experiments.visor.h3 import dis_flow, warp

    if current.dtype != np.uint8 or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError('current must be uint8 RGB')
    if any(not 0 <= weight <= 1 for weight in weights):
        raise ValueError('weight must be in [0, 1]')
    if previous is None or not any(weights):
        return [current.copy() for _ in weights]
    if previous.dtype != np.uint8 or previous.shape != current.shape:
        raise ValueError('predecessor must match current crop')
    target = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    source = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    aligned = warp(previous, dis_flow(target, source))
    return [current.copy() if weight == 0 else
            np.rint((1 - weight) * current.astype(np.float32) + weight * aligned.astype(np.float32)).astype(np.uint8)
            for weight in weights]


def smooth_crop(current: np.ndarray, previous: np.ndarray | None, weight: float) -> np.ndarray:
    """Blend with the warped raw predecessor; a missing predecessor resets."""
    return smooth_crops(current, previous, (weight,))[0]
