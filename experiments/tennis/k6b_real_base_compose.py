"""Receiver composition over decoded base using only predicted player masks."""
from __future__ import annotations

import numpy as np

from experiments.tennis.k6b import paste


def compose(base: np.ndarray, crops: dict[str, np.ndarray], boxes: dict[str, np.ndarray],
            masks: dict[str, np.ndarray], order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed far-to-near order; missing crop or box leaves the paid base intact."""
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError('base must be uint8 RGB')
    if len(set(order)) != len(order) or set(masks) != set(order):
        raise ValueError('player order or prediction set changed')
    canvas = base.copy()
    written = np.zeros(base.shape[:2], dtype=bool)
    for player in order:
        mask = masks[player]
        if mask.dtype != bool or mask.shape != base.shape[:2]:
            raise ValueError('invalid predicted mask')
        if player not in crops or player not in boxes:
            continue
        written |= paste(canvas, crops[player], boxes[player], mask)
    if not np.array_equal(canvas[~written], base[~written]):
        raise AssertionError('base fallback changed outside predicted support')
    return canvas, written
