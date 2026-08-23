"""Small synthetic scenes that look enough like the two domains to test the pipeline."""

from __future__ import annotations

import numpy as np


def tennis_clip(
    *,
    n_frames: int = 12,
    height: int = 96,
    width: int = 128,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mostly-static court, two small striped players. Player area ~8–12%."""
    rng = np.random.default_rng(seed)
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[..., 1] = 90
    court[..., 0] = 40
    court[..., 2] = 50
    yy, xx = np.indices((height, width))
    lines = ((xx % 32) < 2) | ((yy % 40) < 1)
    court[lines] = (220, 220, 220)
    noise = rng.integers(0, 12, size=court.shape, dtype=np.uint16)
    court = np.clip(court.astype(np.uint16) + noise, 0, 255).astype(np.uint8)

    frames = np.repeat(court[np.newaxis, ...], n_frames, axis=0)
    masks = np.zeros((n_frames, height, width), dtype=bool)
    pw, ph = max(8, width // 10), max(16, height // 4)
    for t in range(n_frames):
        x1 = 16 + t * 3
        y1 = height // 3
        x2 = width - pw - 20 - t * 2
        y2 = height // 2
        _paint_player(frames[t], masks[t], x1, y1, pw, ph, stripe=t)
        _paint_player(frames[t], masks[t], x2, y2, pw, ph, stripe=t + 3)
    return frames, masks


def handheld_clip(
    *,
    n_frames: int = 12,
    height: int = 96,
    width: int = 128,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Translating background (parallax-like) plus one person. Panorama is invalid."""
    rng = np.random.default_rng(seed)
    canvas_w = width + n_frames * 4
    canvas = rng.integers(20, 180, size=(height, canvas_w, 3), dtype=np.uint8)
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    masks = np.zeros((n_frames, height, width), dtype=bool)
    pw, ph = max(10, width // 8), max(20, height // 3)
    for t in range(n_frames):
        frames[t] = canvas[:, t * 4 : t * 4 + width]
        x = width // 3
        y = height // 4
        _paint_player(frames[t], masks[t], x, y, pw, ph, stripe=t)
    return frames, masks


def _paint_player(
    frame: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    stripe: int,
) -> None:
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    yy, xx = np.indices(region.shape[:2])
    region[:] = (30, 30, 180)
    region[(xx + stripe) % 4 < 2] = (220, 40, 40)
    mask[y0:y1, x0:x1] = True
