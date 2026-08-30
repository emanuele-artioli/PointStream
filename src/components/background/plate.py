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
import warnings

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

MAX_CANVAS_SCALE = 4

#: RANSAC reprojection threshold, in pixels, for the frame -> frame-0 fit, with
#: the iteration budget that threshold needs.
#:
#: This started at 3.0 with the OpenCV default iteration count, and on a panning
#: 4K broadcast shot that fit was wrong in a way that looked plausible: the
#: median tracked point on `federer_djokovic/scene_003` frame 7 moved 37.7 px
#: while the fitted homography moved the frame centre only 21.2 px. A loose
#: threshold lets a 500-point consensus spend the pan on a spurious ~0.2%
#: per-frame zoom instead, because the scene is not a plane -- court, players
#: and stands sit at different depths, so no homography fits all of them and the
#: slack goes somewhere.
#:
#: Measured mean background alignment error over frames 1-7, players excluded
#: (`outputs/bp29-panorama/motion-model-comparison.json`):
#:
#: | model | dynamic clip | static clip |
#: |---|---|---|
#: | identity (no registration) | 12.32 | 0.411 |
#: | homography, threshold 3.0 | 4.73 | 0.413 |
#: | homography, threshold 1.0 | **3.24** | 0.411 |
#: | affine, 6 DOF | 6.37 | 0.412 |
#: | similarity, 4 DOF | 4.08 | 0.411 |
#:
#: So the fault was the threshold, not the model class: a lower-DOF model does
#: not beat the tightened homography. On the static clip every candidate ties
#: with identity, which is the control that says this is fitting real camera
#: motion rather than fitting noise harder.
RANSAC_REPROJ_PX = 1.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.999


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
        mapped, _ = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_PX,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )
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
    """Median along frame 0, leaving All-NaN pixels as NaN.

    ``np.nanmedian`` warns on an All-NaN slice (a canvas column masked in
    every frame, or a warp margin that never saw a source pixel). The warning
    is suppressed here because ``_nearest_finite_fill`` is the documented
    follow-up — not a silent zero.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(stacked, axis=0)


def _nearest_finite_fill(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace remaining NaN pixels with the nearest finite pixel (Euclidean).

    This is the explicit fill for holes ``nanmedian`` cannot score: an
    all-masked column, or a canvas margin that never saw a source pixel.
    A plate with no finite pixel at all is a recorded hole, not a zero fill.
    """
    out = np.asarray(image, dtype=np.float64).copy()
    if out.ndim != 3:
        raise ValueError(f"expected HxWxC plate, got {tuple(out.shape)}")
    hole = np.isnan(out).any(axis=-1)
    n_holes = int(hole.sum())
    if n_holes == 0:
        return out, 0
    if not np.any(~hole):
        raise RuntimeError(
            "background plate has no finite pixels; nearest-valid fill cannot "
            "run. This is a recorded hole, not a silent zero."
        )
    _dist, indices = distance_transform_edt(hole, return_indices=True)
    del _dist
    out[hole] = out[indices[0][hole], indices[1][hole]]
    return out, n_holes


def build_plate(
    frames: np.ndarray,
    masks: np.ndarray | Sequence[np.ndarray] | None = None,
    *,
    register: bool = True,
) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    """Median-composite a plate, excluding ``masks`` (nonzero = foreground).

    Pixels that are masked (or off-canvas) in every frame have no median.
    Those holes are filled with the nearest finite plate pixel, not with
    the player colour and not with a silent zero. A plate that is NaN
    everywhere raises rather than encoding black.

    Args:
        frames: ``(N, H, W, 3)`` uint8 BGR.
        masks: Foreground to keep out of the median; nonzero is foreground.
        register: When False, every homography is the identity — the frames
            are median-composited where they lie, on a frame-sized canvas.
            This is **the control**, not a mode to ship. A plate does two
            separable things: it compensates camera motion, and it averages
            away whatever differs between frames (sensor noise, a masked
            player's hole, compression dither). Turning registration off
            leaves only the second, so the two can be told apart instead of
            being credited to whichever one the story prefers.

    Returns the uint8 BGR plate and the per-frame homographies as 9-tuples.
    """
    stack = _as_frames(frames)
    n_frames, height, width, _ = stack.shape
    if register:
        homographies = estimate_homographies(stack)
    else:
        homographies = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]
    canvas = _canvas(homographies, width, height)
    if canvas is None:
        homographies = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]
        canvas_w, canvas_h = width, height
        maps = homographies
    else:
        maps, (canvas_w, canvas_h) = canvas

    exclusion = _normalize_masks(masks, n_frames, height, width)

    masked_stack: list[np.ndarray] = []
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
    filled, _n_nearest = _nearest_finite_fill(filled)
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
