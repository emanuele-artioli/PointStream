"""Clipped-residual saturation fraction. No encodes."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.residual.lossy import residual_clip_fraction


def test_difference_of_127_does_not_clip_and_128_does() -> None:
    signed = np.zeros((1, 2, 2, 3), dtype=np.int16)
    signed[0, 0, 0] = 127
    signed[0, 0, 1] = 128
    mask = np.ones((1, 2, 2), dtype=bool)
    assert residual_clip_fraction(signed, mask) == 0.25


def test_one_channel_saturates_the_pixel() -> None:
    signed = np.zeros((1, 1, 1, 3), dtype=np.int16)
    signed[0, 0, 0, 2] = -129
    mask = np.ones((1, 1, 1), dtype=bool)
    assert residual_clip_fraction(signed, mask) == 1.0


def test_empty_foreground_is_rejected() -> None:
    signed = np.zeros((1, 2, 2), dtype=np.int16)
    mask = np.zeros((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        residual_clip_fraction(signed, mask)
