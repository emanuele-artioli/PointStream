"""Remove players from a clip: plate inpaint (best) and flat fill (lower bound)."""

from __future__ import annotations

import cv2
from dataclasses import dataclass

import numpy as np

from src.components.background.plate import build_plate
from src.components.codec.frames import even_size

_FLAT = 128



def as_mask(masks: np.ndarray, n_frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(masks)
    if array.ndim == 2:
        array = np.repeat(array[np.newaxis, ...], n_frames, axis=0)
    if array.shape != (n_frames, height, width):
        raise ValueError(f"mask shape {array.shape} does not match {(n_frames, height, width)}")
    return array.astype(bool)



def flat_fill(frames: np.ndarray, masks: np.ndarray, value: int = _FLAT) -> np.ndarray:
    """Replace masked pixels with a constant. Cheaper than any real background."""
    clip = np.asarray(frames).copy()
    mask = as_mask(masks, clip.shape[0], clip.shape[1], clip.shape[2])
    clip[mask] = np.array((value, value, value), dtype=clip.dtype)
    return clip


def court_median_fill(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Replace masked pixels with the per-frame median of the unmasked court.

    BP13's grey hole (value 128) sat in a green court and the encoder spent
    bits on the edge. A court-coloured hole is the other bracket: still a
    hole, but not a high-contrast object.
    """
    clip = np.asarray(frames).copy()
    mask = as_mask(masks, clip.shape[0], clip.shape[1], clip.shape[2])
    for index in range(clip.shape[0]):
        background = clip[index][~mask[index]]
        if background.size == 0:
            continue
        color = np.median(background.reshape(-1, 3), axis=0)
        clip[index][mask[index]] = color.astype(clip.dtype, copy=False)
    return clip


@dataclass(frozen=True)
class Fills:
    """The three removals, plus what the plate cost to build.

    A dict of ``ndarray | tuple`` used to carry these, which made every caller
    unable to tell an image from a homography list without checking. Three
    arms are compared against one another here; giving them names the type
    system understands is worth the ten lines.
    """

    plate: np.ndarray
    """Background-plate inpaint. The honest reconstruction."""
    flat: np.ndarray
    """Constant fill. Written down as an upper bracket and measured as a
    *lower* one on both synthetic and real footage — a flat hole is a
    high-contrast object and the encoder spends bits on its edges."""
    median: np.ndarray
    """Temporal median of the region. Sits between the other two."""
    plate_bgr: np.ndarray
    """The panorama plate itself, for the background accounting."""
    homographies: tuple[tuple[float, ...], ...]
    """Per-frame warps, for the background payload."""


def prepare_fills(frames: np.ndarray, masks: np.ndarray) -> Fills:
    """Plate, flat, and court-median fills from one plate construction."""
    filled, plate_bgr, homographies = plate_fill(frames, masks)
    return Fills(
        plate=filled,
        flat=flat_fill(frames, masks),
        median=court_median_fill(frames, masks),
        plate_bgr=plate_bgr,
        homographies=homographies,
    )


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
