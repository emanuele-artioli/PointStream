"""Signed plate differences, shifted into uint8 so a still-image sidecar can carry them.

A uint8 image only represents per-channel deltas in [-128, 127] exactly.
Larger jumps clip. That is an accepted limit: consecutive sub-chunks of one
scene are expected to drift (scoreboard digits, crowd) rather than jump.
"""

from __future__ import annotations

import numpy as np


def _as_bgr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected a BGR plate [H, W, 3], got {tuple(array.shape)}.")
    return array


def compute_delta(current_bgr: np.ndarray, previous_bgr: np.ndarray) -> np.ndarray:
    """``current - previous``, offset by 128 and clipped to uint8."""
    current = _as_bgr(current_bgr).astype(np.int16)
    previous = _as_bgr(previous_bgr).astype(np.int16)
    if current.shape != previous.shape:
        raise ValueError(
            f"Panorama delta shape mismatch: current={current.shape}, previous={previous.shape}."
        )
    return np.clip(current - previous + 128, 0, 255).astype(np.uint8)


def apply_delta(previous_bgr: np.ndarray, diff_bgr: np.ndarray) -> np.ndarray:
    """Inverse of ``compute_delta``."""
    previous = _as_bgr(previous_bgr).astype(np.int16)
    diff = _as_bgr(diff_bgr).astype(np.int16)
    if previous.shape != diff.shape:
        raise ValueError(
            f"Panorama delta shape mismatch: previous={previous.shape}, diff={diff.shape}."
        )
    return np.clip(previous + diff - 128, 0, 255).astype(np.uint8)
