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
from dataclasses import dataclass
import warnings

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

MAX_CANVAS_SCALE = 4

#: Mid-grey fill for canvas pixels no source frame ever covered. Constant so a
#: later scene's padding predicts from the same value, and so padding cost is
#: a measurement rather than a texture accident. Not used on the independent
#: local-canvas path, which keeps nearest-finite fill for warp margins.
PAD_FILL = 128

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

#: Inliers required before two scene frame-0 images are treated as the same
#: camera geometry. Below this, the pair is unaligned and the canvas falls
#: back to pad-to-max rather than a shared origin.
_ALIGN_MIN_INLIERS = 8


@dataclass(frozen=True)
class HomographyBounds:
    """One scene's camera-motion envelope, in that scene's frame-0 coordinates.

    ``homographies`` map each source frame into frame 0 *before* any canvas
    origin shift. ``min_xy`` / ``max_xy`` are the axis-aligned envelope of the
    warped frame corners in that same space.
    """

    frame_width: int
    frame_height: int
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]
    homographies: tuple[np.ndarray, ...]

    @property
    def local_width(self) -> int:
        return max(1, int(np.ceil(self.max_xy[0] - self.min_xy[0])))

    @property
    def local_height(self) -> int:
        return max(1, int(np.ceil(self.max_xy[1] - self.min_xy[1])))

    @property
    def local_area(self) -> int:
        return self.local_width * self.local_height


@dataclass(frozen=True)
class CanonicalCanvas:
    """Shared even-sized canvas for one background context.

    ``origin_xy`` is the top-left of the union in the shared coordinate
    system (the first scene's frame 0 when alignment succeeded). This mode
    is offline: the union sees every scene in the context before any
    background is encoded. Causal canvas growth is a different product.
    """

    context_id: str
    origin_xy: tuple[float, float]
    width: int
    height: int
    aligned: bool = True

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)


def even_up(value: int) -> int:
    """Smallest even integer ``>= value``. Video frames in a sequence need this."""
    number = int(value)
    if number < 1:
        raise ValueError(f"canvas edge must be positive, got {number}")
    return number if number % 2 == 0 else number + 1


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


def _frame_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    ).reshape(1, -1, 2)


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def collect_bounds(
    homographies: Sequence[np.ndarray], width: int, height: int
) -> HomographyBounds | None:
    """Envelope of warped frame corners in frame-0 coordinates, or None if unusable."""
    projected: list[np.ndarray] = []
    corners = _frame_corners(width, height)
    packed: list[np.ndarray] = []
    for matrix in homographies:
        array = np.asarray(matrix, dtype=np.float64)
        packed.append(array)
        warped = cv2.perspectiveTransform(corners, array)
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
    return HomographyBounds(
        frame_width=int(width),
        frame_height=int(height),
        min_xy=(float(min_xy[0]), float(min_xy[1])),
        max_xy=(float(max_xy[0]), float(max_xy[1])),
        homographies=tuple(packed),
    )


def transform_bounds(bounds: HomographyBounds, alignment: np.ndarray) -> HomographyBounds:
    """Rewrite a scene's envelope into another frame-0 coordinate system."""
    matrix = np.asarray(alignment, dtype=np.float64)
    corners = np.array(
        [
            [bounds.min_xy[0], bounds.min_xy[1]],
            [bounds.max_xy[0], bounds.min_xy[1]],
            [bounds.max_xy[0], bounds.max_xy[1]],
            [bounds.min_xy[0], bounds.max_xy[1]],
        ],
        dtype=np.float64,
    ).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    min_xy = mapped.min(axis=0)
    max_xy = mapped.max(axis=0)
    return HomographyBounds(
        frame_width=bounds.frame_width,
        frame_height=bounds.frame_height,
        min_xy=(float(min_xy[0]), float(min_xy[1])),
        max_xy=(float(max_xy[0]), float(max_xy[1])),
        homographies=bounds.homographies,
    )


def union_canvas(
    bounds: Sequence[HomographyBounds],
    *,
    context_id: str,
    aligned: bool = True,
) -> CanonicalCanvas:
    """One even-sized canvas covering every scene envelope in shared coordinates."""
    if not bounds:
        raise ValueError("a canonical canvas needs at least one scene's bounds")
    min_x = min(item.min_xy[0] for item in bounds)
    min_y = min(item.min_xy[1] for item in bounds)
    max_x = max(item.max_xy[0] for item in bounds)
    max_y = max(item.max_xy[1] for item in bounds)
    width = even_up(int(np.ceil(max_x - min_x)))
    height = even_up(int(np.ceil(max_y - min_y)))
    frame_w = max(item.frame_width for item in bounds)
    frame_h = max(item.frame_height for item in bounds)
    if width > MAX_CANVAS_SCALE * frame_w or height > MAX_CANVAS_SCALE * frame_h:
        raise ValueError(
            f"canonical canvas {width}x{height} exceeds {MAX_CANVAS_SCALE}x "
            f"the largest source frame {frame_w}x{frame_h}"
        )
    return CanonicalCanvas(
        context_id=context_id,
        origin_xy=(float(min_x), float(min_y)),
        width=width,
        height=height,
        aligned=aligned,
    )


def unaligned_canvas(
    bounds: Sequence[HomographyBounds], *, context_id: str
) -> CanonicalCanvas:
    """Pad each local plate to max(width) x max(height), origin at each local min.

    Used when two scenes were declared the same context but their frame-0
    images could not be registered. Equal size is enough for the encoder;
    reconstruction stays aligned because each scene's maps still start at
    its own origin. Predictive coding will be weaker than an aligned union.
    """
    if not bounds:
        raise ValueError("a canonical canvas needs at least one scene's bounds")
    width = even_up(max(item.local_width for item in bounds))
    height = even_up(max(item.local_height for item in bounds))
    return CanonicalCanvas(
        context_id=context_id,
        origin_xy=(0.0, 0.0),
        width=width,
        height=height,
        aligned=False,
    )


def estimate_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Homography mapping ``source`` pixel coords into ``target`` pixel coords.

    Returns None when the pair cannot be registered — not an identity, which
    would silently treat an unrelated camera as a match.
    """
    src = np.asarray(source, dtype=np.uint8)
    dst = np.asarray(target, dtype=np.uint8)
    if src.shape != dst.shape or src.ndim != 3:
        return None
    gray_t = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    points_t = cv2.goodFeaturesToTrack(
        gray_t,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points_t is None or len(points_t) < 4:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_t,
        gray_s,
        points_t,
        points_t.copy(),
        winSize=(21, 21),
        maxLevel=3,
        criteria=criteria,
    )
    if tracked is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    src_pts = tracked.reshape(-1, 2)[valid]
    dst_pts = points_t.reshape(-1, 2)[valid]
    if src_pts.shape[0] < 4:
        return None
    mapped, inliers = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_PX,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )
    if mapped is None or inliers is None:
        return None
    if int(np.asarray(inliers).reshape(-1).sum()) < _ALIGN_MIN_INLIERS:
        return None
    return np.asarray(mapped, dtype=np.float64)


def canvas_maps(
    bounds: HomographyBounds,
    canvas: CanonicalCanvas,
    alignment: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Per-frame maps taking source frames onto ``canvas``."""
    if canvas.aligned:
        align = (
            np.eye(3, dtype=np.float64)
            if alignment is None
            else np.asarray(alignment, dtype=np.float64)
        )
        offset = _translation(-canvas.origin_xy[0], -canvas.origin_xy[1])
        return [offset @ align @ np.asarray(matrix, dtype=np.float64) for matrix in bounds.homographies]
    local_origin = _translation(-bounds.min_xy[0], -bounds.min_xy[1])
    return [local_origin @ np.asarray(matrix, dtype=np.float64) for matrix in bounds.homographies]


def prepare_canonical_context(
    scenes: Sequence[np.ndarray],
    *,
    context_id: str,
    register: bool = True,
) -> tuple[CanonicalCanvas, tuple[np.ndarray, ...], tuple[HomographyBounds, ...]]:
    """Offline prepass: bounds, alignments, and one canvas for a compatible group.

    The first scene's frame 0 is the shared origin when every later scene
    registers to it. This sees future scenes, so it is a buffered codec mode.
    """
    if not scenes:
        raise ValueError("a background context needs at least one scene")
    collected: list[HomographyBounds] = []
    for frames in scenes:
        stack = _as_frames(frames)
        if register:
            homographies = estimate_homographies(stack)
        else:
            homographies = [np.eye(3, dtype=np.float64) for _ in range(stack.shape[0])]
        bounds = collect_bounds(homographies, int(stack.shape[2]), int(stack.shape[1]))
        if bounds is None:
            identity = [np.eye(3, dtype=np.float64) for _ in range(stack.shape[0])]
            bounds = HomographyBounds(
                frame_width=int(stack.shape[2]),
                frame_height=int(stack.shape[1]),
                min_xy=(0.0, 0.0),
                max_xy=(float(stack.shape[2]), float(stack.shape[1])),
                homographies=tuple(identity),
            )
        collected.append(bounds)

    alignments: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    aligned = True
    reference = _as_frames(scenes[0])[0]
    for frames in scenes[1:]:
        mapped = estimate_alignment(_as_frames(frames)[0], reference) if register else np.eye(3)
        if mapped is None:
            aligned = False
            break
        alignments.append(mapped)

    if aligned:
        shared = [collected[0]]
        shared.extend(
            transform_bounds(item, alignment)
            for item, alignment in zip(collected[1:], alignments[1:], strict=True)
        )
        canvas = union_canvas(shared, context_id=context_id, aligned=True)
        return canvas, tuple(alignments), tuple(collected)

    canvas = unaligned_canvas(collected, context_id=context_id)
    identities = tuple(np.eye(3, dtype=np.float64) for _ in collected)
    return canvas, identities, tuple(collected)


def _canvas(
    homographies: Sequence[np.ndarray], width: int, height: int
) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    bounds = collect_bounds(homographies, width, height)
    if bounds is None:
        return None
    offset = _translation(-bounds.min_xy[0], -bounds.min_xy[1])
    adjusted = [offset @ matrix for matrix in bounds.homographies]
    return adjusted, (bounds.local_width, bounds.local_height)


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
    canvas: CanonicalCanvas | None = None,
    alignment: np.ndarray | None = None,
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
        canvas: When set, warp onto this shared canvas instead of the
            scene's local envelope. Required for predictive coding of
            scenes whose local panoramas differ in size. Offline: the
            canvas must already be the union of the context.
        alignment: Homography taking this scene's frame 0 into the canvas
            shared coordinate system. Ignored when ``canvas`` is None or
            unaligned.

    Returns the uint8 BGR plate and the per-frame homographies as 9-tuples.
    """
    stack = _as_frames(frames)
    n_frames, height, width, _ = stack.shape
    if register:
        homographies = estimate_homographies(stack)
    else:
        homographies = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]

    bounds = collect_bounds(homographies, width, height)
    if canvas is not None:
        if bounds is None:
            bounds = HomographyBounds(
                frame_width=width,
                frame_height=height,
                min_xy=(0.0, 0.0),
                max_xy=(float(width), float(height)),
                homographies=tuple(np.asarray(m, dtype=np.float64) for m in homographies),
            )
        maps = canvas_maps(bounds, canvas, alignment)
        canvas_w, canvas_h = int(canvas.width), int(canvas.height)
        pad_exterior = True
    elif bounds is None:
        homographies = [np.eye(3, dtype=np.float64) for _ in range(n_frames)]
        canvas_w, canvas_h = width, height
        maps = homographies
        pad_exterior = False
    else:
        offset = _translation(-bounds.min_xy[0], -bounds.min_xy[1])
        maps = [offset @ matrix for matrix in bounds.homographies]
        canvas_w, canvas_h = bounds.local_width, bounds.local_height
        pad_exterior = False

    exclusion = _normalize_masks(masks, n_frames, height, width)
    plate, packed = _composite(
        stack,
        maps,
        canvas_w,
        canvas_h,
        exclusion,
        pad_exterior=pad_exterior,
    )
    return plate, packed


def _composite(
    stack: np.ndarray,
    maps: Sequence[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    exclusion: np.ndarray | None,
    *,
    pad_exterior: bool,
) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    n_frames, height, width, _ = stack.shape
    masked_stack: list[np.ndarray] = []
    ever_valid = np.zeros((canvas_h, canvas_w), dtype=bool)
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
        ever_valid |= ~invalid
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
    if pad_exterior:
        filled[~ever_valid] = float(PAD_FILL)
    plate = np.asarray(np.clip(filled, 0.0, 255.0), dtype=np.uint8)
    packed = tuple(tuple(float(v) for v in np.asarray(matrix).reshape(-1)) for matrix in maps)
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
