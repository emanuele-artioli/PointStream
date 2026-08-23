"""Build a warpable background plate from a span of frames.

A tennis broadcast camera is pan-tilt-zoom, so successive frames relate by a
homography and a median composite after warping is a sound plate. Foreground
masks, when present, are excluded from the median so a player who stands still
for half the chunk does not burn into the background.

This is a rewrite, not a wrap of ``src.encoder.background_modeler``. The
pre-rewrite stitcher is prior art to read, not a foundation to trust.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

MAX_CANVAS_SCALE = 4


def _as_frames(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected frames [N, H, W, 3], got {tuple(array.shape)}.")
    if array.shape[0] == 0:
        raise ValueError("Cannot build a background plate from zero frames.")
    return np.asarray(array, dtype=np.uint8)


def estimate_homographies(frames: np.ndarray) -> list[np.ndarray]:
    """One 3x3 map per frame taking that frame into frame-0 coordinates."""
    stack = _as_frames(frames)
    n_frames = int(stack.shape[0])
    identity = np.eye(3, dtype=np.float64)
    gray0 = cv2.cvtColor(stack[0], cv2.COLOR_BGR2GRAY)
    points0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points0 is None or len(points0) < 4:
        return [identity.copy() for _ in range(n_frames)]

    homographies: list[np.ndarray] = [identity.copy()]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    for index in range(1, n_frames):
        gray = cv2.cvtColor(stack[index], cv2.COLOR_BGR2GRAY)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            gray0,
            gray,
            points0,
            points0.copy(),
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if tracked is None or status is None:
            homographies.append(identity.copy())
            continue
        valid = status.reshape(-1) == 1
        src = tracked.reshape(-1, 2)[valid]
        dst = points0.reshape(-1, 2)[valid]
        if src.shape[0] < 4:
            homographies.append(identity.copy())
            continue
        mapped, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if mapped is None:
            homographies.append(identity.copy())
        else:
            homographies.append(np.asarray(mapped, dtype=np.float64))
    return homographies


def _canvas(
    homographies: Sequence[np.ndarray], width: int, height: int
) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    ).reshape(1, -1, 2)
    projected: list[np.ndarray] = []
    for matrix in homographies:
        warped = cv2.perspectiveTransform(corners, matrix)
        projected.append(warped.reshape(-1, 2))
    stacked = np.vstack(projected)
    min_xy = stacked.min(axis=0)
    max_xy = stacked.max(axis=0)
    canvas_w = int(np.ceil(max_xy[0] - min_xy[0]))
    canvas_h = int(np.ceil(max_xy[1] - min_xy[1]))
    if canvas_w < 1 or canvas_h < 1:
        return None
    if canvas_w > MAX_CANVAS_SCALE * width or canvas_h > MAX_CANVAS_SCALE * height:
        return None
    offset = np.array(
        [[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    adjusted = [offset @ np.asarray(matrix, dtype=np.float64) for matrix in homographies]
    return adjusted, (canvas_w, canvas_h)


def _nanmedian(stacked: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(stacked, axis=0)


def build_plate(
    frames: np.ndarray,
    masks: np.ndarray | Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    """Median-composite a plate, excluding ``masks`` (nonzero = foreground).

    Returns the uint8 BGR plate and the per-frame homographies as 9-tuples.
    """
    stack = _as_frames(frames)
    n_frames, height, width, _ = stack.shape
    homographies = estimate_homographies(stack)
    canvas = _canvas(homographies, width, height)
    if canvas is None:
        homographies = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]
        canvas_w, canvas_h = width, height
        maps = homographies
    else:
        maps, (canvas_w, canvas_h) = canvas

    exclusion = _normalize_masks(masks, n_frames, height, width)

    masked_stack: list[np.ndarray] = []
    unmasked_stack: list[np.ndarray] = []
    valid_src = np.full((height, width), 255, dtype=np.uint8)
    for index in range(n_frames):
        warped = cv2.warpPerspective(
            stack[index].astype(np.float32),
            maps[index],
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_valid = cv2.warpPerspective(
            valid_src,
            maps[index],
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
        invalid = warped_valid < 127
        unmasked = warped.copy()
        unmasked[invalid] = np.nan
        unmasked_stack.append(unmasked)

        masked = warped.copy()
        if exclusion is not None:
            warped_excl = cv2.warpPerspective(
                exclusion[index],
                maps[index],
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )
            masked[invalid | (warped_excl > 0)] = np.nan
        else:
            masked[invalid] = np.nan
        masked_stack.append(masked)

    filled = _nanmedian(np.stack(masked_stack, axis=0))
    fallback = _nanmedian(np.stack(unmasked_stack, axis=0))
    filled = np.where(np.isnan(filled), fallback, filled)
    filled = np.nan_to_num(filled, nan=0.0, posinf=255.0, neginf=0.0)
    plate = np.asarray(np.clip(filled, 0.0, 255.0), dtype=np.uint8)
    packed = tuple(tuple(float(v) for v in matrix.reshape(-1)) for matrix in maps)
    return plate, packed


def _normalize_masks(
    masks: np.ndarray | Sequence[np.ndarray] | None,
    n_frames: int,
    height: int,
    width: int,
) -> np.ndarray | None:
    if masks is None:
        return None
    array = np.asarray(masks)
    if array.ndim == 2:
        array = np.repeat(array[np.newaxis, ...], n_frames, axis=0)
    if array.shape[0] != n_frames:
        raise ValueError(f"Got {array.shape[0]} masks for {n_frames} frames.")
    if array.shape[1] != height or array.shape[2] != width:
        raise ValueError(
            f"Mask size {array.shape[1:]} does not match frame size {(height, width)}."
        )
    return np.asarray(array, dtype=np.uint8)
