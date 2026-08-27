"""Frame shaping shared by anything that hands pixels to an encoder.

These two helpers lived in ``experiments/headroom/remove.py``. The runner's
codec stage needs them too, and `src/` may not import from `experiments/`, so
they live here and `experiments` imports down to them. One definition, not a
copy — `python -m src.contracts.layers` enforces the direction.
"""

from __future__ import annotations

import numpy as np


def even_size(frames: np.ndarray) -> np.ndarray:
    """Crop to even width and height so 4:2:0 y4m is well-defined."""
    clip = np.asarray(frames)
    height = clip.shape[1] - (clip.shape[1] % 2)
    width = clip.shape[2] - (clip.shape[2] % 2)
    if height < 2 or width < 2:
        raise ValueError(f"clip {tuple(clip.shape)} is too small for 4:2:0")
    return clip[:, :height, :width]


def rgb_to_luma(frames: np.ndarray) -> np.ndarray:
    """BT.601 luma, uint8, shape ``(T, H, W)``."""
    clip = np.asarray(frames, dtype=np.float64)
    luma = 0.299 * clip[..., 0] + 0.587 * clip[..., 1] + 0.114 * clip[..., 2]
    return np.clip(luma, 0, 255).astype(np.uint8)
