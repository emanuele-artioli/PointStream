"""Write RGBA PNG sequences for overlayable maps (never counted as payload)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def as_bgra(frame: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(frame)
    if arr.ndim == 2:
        bgra = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
        bgra[:, :, 3] = np.where(arr > 0, 220, 0).astype(np.uint8)
        return bgra
    if arr.ndim != 3:
        raise ValueError(f"expected HxW or HxWxC, got {arr.shape}")
    channels = arr.shape[2]
    if channels == 4:
        return arr
    if channels == 3:
        bgra = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        luma = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        bgra[:, :, 3] = np.where(luma > 0, 220, 0).astype(np.uint8)
        return bgra
    raise ValueError(f"unsupported channel count {channels}")


def write_rgba_png_sequence(frames: Sequence[np.ndarray], preview_dir: Path) -> int:
    """Write BGRA PNGs named ``000000.png`` …. Returns total bytes."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, frame in enumerate(frames):
        path = preview_dir / f"{index:06d}.png"
        vis = as_bgra(frame)
        if not cv2.imwrite(str(path), vis):
            raise RuntimeError(f"failed to write preview {path}")
        total += path.stat().st_size
    return total
