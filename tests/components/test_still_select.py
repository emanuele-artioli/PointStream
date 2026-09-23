"""Best-frame background selection. No encodes."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.still import background_paste_mse, select_best_background_frame


def _clip() -> tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    frames[1] = 10
    frames[2] = 10
    foreground = np.zeros((3, 2, 2), dtype=bool)
    foreground[:, 0, 0] = True
    return frames, foreground


def test_best_frame_prefers_the_repeated_background_and_breaks_ties_early() -> None:
    frames, foreground = _clip()
    index, mse = select_best_background_frame(frames, foreground)
    assert index == 1
    assert mse == background_paste_mse(frames[1], frames, foreground)
    assert mse < background_paste_mse(frames[0], frames, foreground)


def test_identical_frames_select_frame_zero() -> None:
    frames = np.full((4, 2, 2, 3), 7, dtype=np.uint8)
    foreground = np.zeros(frames.shape[:3], dtype=bool)
    index, mse = select_best_background_frame(frames, foreground)
    assert index == 0
    assert mse == 0.0


def test_one_level_background_change_scores_one() -> None:
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    candidate = np.ones((2, 2, 3), dtype=np.uint8)
    foreground = np.zeros((1, 2, 2), dtype=bool)
    assert background_paste_mse(candidate, frames, foreground) == 1.0


def test_empty_background_is_rejected() -> None:
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    foreground = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        select_best_background_frame(frames, foreground)
