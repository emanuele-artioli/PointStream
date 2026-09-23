"""Choose one unwarped frame as a background still.

``still_frame0`` remains the control: it always repeats frame 0. The best-frame
arm searches every frame and keeps the one that minimizes background error when
pasted in place, with no homography.
"""

from __future__ import annotations

import numpy as np


def background_paste_mse(
    candidate: np.ndarray,
    frames: np.ndarray,
    foreground: np.ndarray,
) -> float:
    """Mean squared error of pasting ``candidate`` onto every frame, background only.

    ``frames`` is ``(T, H, W, C)`` uint8. ``foreground`` is ``(T, H, W)`` and
    True on the player. Scoring uses the complement, so a player painted into
    the still counts as error wherever a later frame's background covers those
    pixels. An empty background raises ValueError.

    A caller relies on an identical candidate and an all-background mask scoring
    0, and on a one-level change over the whole background scoring 1.
    """
    stack = np.asarray(frames)
    mask = np.asarray(foreground, dtype=bool)
    if stack.ndim != 4:
        raise ValueError(f"frames must be (T, H, W, C), got {stack.shape}")
    if mask.shape != stack.shape[:3]:
        raise ValueError(f"foreground shape {mask.shape} does not match frames {stack.shape[:3]}")
    background = ~mask
    if not np.any(background):
        raise ValueError("background mask is empty")
    still = np.asarray(candidate)
    if still.shape != stack.shape[1:]:
        raise ValueError(f"candidate shape {still.shape} does not match a frame {stack.shape[1:]}")
    diff = stack.astype(np.float64) - still.astype(np.float64)
    return float(np.mean(np.square(diff[background])))


def select_best_background_frame(
    frames: np.ndarray,
    foreground: np.ndarray,
) -> tuple[int, float]:
    """Return ``(index, mse)`` for the unwarped frame with the lowest background MSE.

    Every frame is a candidate, including frame 0. Ties take the lowest index,
    so a flat window selects frame 0 and the control and the best-frame arm
    coincide. An empty clip raises ValueError.
    """
    stack = np.asarray(frames)
    if stack.ndim != 4 or stack.shape[0] < 1:
        raise ValueError(f"frames must be a non-empty (T, H, W, C) clip, got {stack.shape}")
    errors = [
        background_paste_mse(stack[index], stack, foreground)
        for index in range(stack.shape[0])
    ]
    best = int(np.argmin(errors))
    return best, errors[best]
