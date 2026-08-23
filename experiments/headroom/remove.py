"""Remove players from a clip: plate inpaint (best) and flat fill (lower bound)."""

from __future__ import annotations

import cv2
import numpy as np

from src.components.background.plate import build_plate

_FLAT = 128


def even_size(frames: np.ndarray) -> np.ndarray:
    """Crop to even width and height so 4:2:0 y4m is well-defined."""
    clip = np.asarray(frames)
    height = clip.shape[1] - (clip.shape[1] % 2)
    width = clip.shape[2] - (clip.shape[2] % 2)
    if height < 2 or width < 2:
        raise ValueError(f"clip {tuple(clip.shape)} is too small for 4:2:0")
    return clip[:, :height, :width]


def as_mask(masks: np.ndarray, n_frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(masks)
    if array.ndim == 2:
        array = np.repeat(array[np.newaxis, ...], n_frames, axis=0)
    if array.shape != (n_frames, height, width):
        raise ValueError(f"mask shape {array.shape} does not match {(n_frames, height, width)}")
    return array.astype(bool)


def rgb_to_luma(frames: np.ndarray) -> np.ndarray:
    """BT.601 luma, uint8, shape ``(T, H, W)``."""
    clip = np.asarray(frames, dtype=np.float64)
    luma = 0.299 * clip[..., 0] + 0.587 * clip[..., 1] + 0.114 * clip[..., 2]
    return np.clip(luma, 0, 255).astype(np.uint8)


def flat_fill(frames: np.ndarray, masks: np.ndarray, value: int = _FLAT) -> np.ndarray:
    """Replace masked pixels with a constant. Cheaper than any real background."""
    clip = np.asarray(frames).copy()
    mask = as_mask(masks, clip.shape[0], clip.shape[1], clip.shape[2])
    clip[mask] = np.array((value, value, value), dtype=clip.dtype)
    return clip


def plate_fill(frames: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...]]:
    """Replace masked pixels with the warped background plate.

    Returns ``(filled_clip, plate_bgr, homographies)``. Homographies map each
    frame into plate coordinates, matching ``build_plate``.
    """
    clip = even_size(np.asarray(frames))
    mask = as_mask(masks, clip.shape[0], clip.shape[1], clip.shape[2])
    bgr = clip[:, :, :, ::-1]
    exclusion = mask.astype(np.uint8) * 255
    plate, packed = build_plate(bgr, masks=exclusion)
    filled = clip.copy()
    height, width = clip.shape[1], clip.shape[2]
    for index in range(clip.shape[0]):
        matrix = np.asarray(packed[index], dtype=np.float64).reshape(3, 3)
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        warped = cv2.warpPerspective(plate, inverse, (width, height))
        rgb = warped[:, :, ::-1]
        filled[index][mask[index]] = rgb[mask[index]]
    return filled, plate, packed


def player_fraction(masks: np.ndarray) -> float:
    mask = np.asarray(masks, dtype=bool)
    if mask.size == 0:
        return 0.0
    return float(mask.mean())
