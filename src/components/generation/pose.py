"""Shared letterbox / pose-rescale for every generator that puts a crop on a canvas.

The four ControlNet classes in the pre-rewrite tree each copied ~40 lines that
computed scale as ``max(canvas) / max(bbox)``. That overflows the canvas whenever
the bbox aspect and the canvas aspect disagree — offsets go negative and the
assignment silently clips, which moves joints. Every backend that letterboxes
now calls ``fit_to_canvas`` instead, which uses the independent min-scale so the
mapped box always sits inside the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a pinned project dependency
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Letterbox:
    """How a source rectangle is fitted into a generation canvas.

    Offsets are non-negative and ``scaled_*`` never exceeds the canvas. Callers
    paste the resized source at ``(offset_x, offset_y)``.
    """

    canvas_width: int
    canvas_height: int
    scaled_width: int
    scaled_height: int
    offset_x: int
    offset_y: int
    scale: float
    source_width: int
    source_height: int

    def map_xy(self, x: float, y: float, *, origin: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """Map a point from source/bbox space onto the canvas."""
        x0, y0 = origin
        return (
            (x - x0) * self.scale + self.offset_x,
            (y - y0) * self.scale + self.offset_y,
        )


def fit_to_canvas(
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> Letterbox:
    """Letterbox ``source`` into ``canvas``, preserving aspect and staying inside.

    The pre-rewrite copies used ``max(canvas) / max(source)``, which is the
    smallest scale that *covers* the longer side and therefore overflows the
    shorter canvas side. The fit that actually belongs here is the largest
    scale that *fits* both sides: ``min(canvas_w / src_w, canvas_h / src_h)``.
    """
    if min(source_width, source_height, canvas_width, canvas_height) < 1:
        raise ValueError(
            f"fit_to_canvas needs positive sizes, got source {source_width}x{source_height} "
            f"and canvas {canvas_width}x{canvas_height}."
        )
    scale = min(canvas_width / source_width, canvas_height / source_height)
    scaled_width = max(1, min(canvas_width, int(round(source_width * scale))))
    scaled_height = max(1, min(canvas_height, int(round(source_height * scale))))
    # Recompute scale from the integer size so keypoint mapping and the paste
    # region agree to the pixel. Using the pre-round scale here would put a
    # joint one pixel off the resized image.
    scale_x = scaled_width / source_width
    scale_y = scaled_height / source_height
    # Aspect is preserved by construction of `scale`; residual disagreement is
    # rounding. Map keypoints with the mean so x and y stay consistent.
    scale = (scale_x + scale_y) / 2.0
    offset_x = (canvas_width - scaled_width) // 2
    offset_y = (canvas_height - scaled_height) // 2
    return Letterbox(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        offset_x=offset_x,
        offset_y=offset_y,
        scale=scale,
        source_width=source_width,
        source_height=source_height,
    )


def letterbox_from_bbox(
    bbox: tuple[int, int, int, int] | None,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> Letterbox:
    """Fit the bbox (or the source image, if no bbox) into the canvas."""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        source_width = max(1, x2 - x1)
        source_height = max(1, y2 - y1)
    return fit_to_canvas(source_width, source_height, canvas_width, canvas_height)


def rescale_keypoints(
    keypoints: np.ndarray,
    letterbox: Letterbox,
    *,
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Map ``[..., 3]`` keypoints (x, y, conf) through ``letterbox``.

    Confidence is passed through. A trailing frame axis ``[T, N, 3]`` is mapped
    frame-wise; a lone ``[N, 3]`` is mapped in place. This is the block that
    used to be copy-pasted, including the ``ndim == 3: take [-1]`` line that
    dropped a temporal axis on the floor after mutating it.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    if points.ndim == 3 and points.shape[-1] == 3 and points.shape[0] != 0:
        # [T, N, 3] — map every frame, do not silently keep only the last.
        mapped = np.empty_like(points)
        for index in range(points.shape[0]):
            mapped[index] = rescale_keypoints(points[index], letterbox, bbox=bbox)
        return mapped
    if points.ndim != 2 or points.shape[-1] < 2:
        raise ValueError(
            f"rescale_keypoints expected (N, 2|3) or (T, N, 3), got {tuple(points.shape)}."
        )
    origin_x, origin_y = (
        (float(bbox[0]), float(bbox[1])) if bbox is not None else (0.0, 0.0)
    )
    out = points.copy()
    out[:, 0] = (out[:, 0] - origin_x) * letterbox.scale + letterbox.offset_x
    out[:, 1] = (out[:, 1] - origin_y) * letterbox.scale + letterbox.offset_y
    return out


def letterbox_image(image: np.ndarray, box: Letterbox) -> np.ndarray:
    """Resize ``image`` into ``box`` and paste it onto a zero canvas.

    2-D (mask/canny) and 3-D (HWC) images are both accepted. Interpolation is
    nearest for 2-D so a binary mask stays binary, linear otherwise.
    """
    if cv2 is None:
        raise RuntimeError("opencv is required to letterbox an image onto a canvas.")
    array = np.asarray(image)
    if array.ndim == 2:
        canvas = np.zeros((box.canvas_height, box.canvas_width), dtype=array.dtype)
        resized = cv2.resize(
            array, (box.scaled_width, box.scaled_height), interpolation=cv2.INTER_NEAREST
        )
        canvas[
            box.offset_y : box.offset_y + box.scaled_height,
            box.offset_x : box.offset_x + box.scaled_width,
        ] = resized
        return canvas
    if array.ndim != 3:
        raise ValueError(f"letterbox_image expected HW or HWC, got shape {tuple(array.shape)}.")
    channels = array.shape[2]
    canvas = np.zeros((box.canvas_height, box.canvas_width, channels), dtype=array.dtype)
    resized = cv2.resize(
        array, (box.scaled_width, box.scaled_height), interpolation=cv2.INTER_LINEAR
    )
    canvas[
        box.offset_y : box.offset_y + box.scaled_height,
        box.offset_x : box.offset_x + box.scaled_width,
    ] = resized
    return canvas
