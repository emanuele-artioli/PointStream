"""Object placement and compositing, independently testable.

A generated (or supplied) crop is resized to its bbox and blended with a mask.
When segmentation is off, the mask is the bbox rectangle — the catalogue's
``when_off`` for that row, not a missing feature.

Nothing here generates pixels. Dispatch produces the crop; this module only
puts it on the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline.reconstruction.clips import Clip, as_clip

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Placement:
    """One object on one frame, already in pixel space.

    Args:
        crop: Object appearance ``(H, W, 3)`` uint8. Ignored when compositing
            is skipped (generation off) — the residual carries it instead.
        bbox: ``(x1, y1, x2, y2)`` in frame coordinates, exclusive on the far
            edge.
        mask: Optional ``(H, W)`` or frame-sized mask. Used only when
            ``use_heuristic_mask`` is false.
        object_id: Stable identity, for logs and region names.
        frame_index: Which clip frame this lands on.
    """

    crop: np.ndarray
    bbox: Box
    mask: np.ndarray | None = None
    object_id: str = "object"
    frame_index: int = 0

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"placement bbox {self.bbox!r} is empty or inverted. A zero-area "
                "box is a resize to nothing, which would fail later as a shape error."
            )
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0; got {self.frame_index}.")


def heuristic_mask(bbox: Box, height: int, width: int) -> np.ndarray:
    """Filled rectangle covering ``bbox``, clipped to the frame.

    The fallback when segmentation is off. It includes background pixels
    around the subject — that is why a real mask, when available, is better,
    and why quality records which was used.
    """
    x1, y1, x2, y2 = _clip_bbox(bbox, height=height, width=width)
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def composite_frame(
    background: np.ndarray,
    placement: Placement,
    *,
    use_heuristic_mask: bool,
) -> np.ndarray:
    """Paste one object onto one ``(H, W, 3)`` background. Returns a new array."""
    frame = np.asarray(background, dtype=np.uint8).copy()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"background frame must be (H, W, 3); got {frame.shape}.")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(placement.bbox, height=height, width=width)
    region_h, region_w = y2 - y1, x2 - x1
    if region_h < 1 or region_w < 1:
        return frame

    crop = np.asarray(placement.crop, dtype=np.uint8)
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"object crop must be (H, W, 3); got {crop.shape}.")
    resized = cv2.resize(crop, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    if use_heuristic_mask or placement.mask is None:
        blend = np.ones((region_h, region_w), dtype=np.float32)
    else:
        blend = _mask_for_bbox(placement.mask, placement.bbox, height, width)[y1:y2, x1:x2]
        blend = blend.astype(np.float32)
        if blend.max() > 1.0:
            blend = blend / 255.0
        blend = np.clip(blend, 0.0, 1.0)

    alpha = blend[..., None]
    patch = frame[y1:y2, x1:x2].astype(np.float32)
    pasted = resized.astype(np.float32) * alpha + patch * (1.0 - alpha)
    frame[y1:y2, x1:x2] = np.clip(pasted, 0, 255).astype(np.uint8)
    return frame


def composite_clip(
    background: np.ndarray,
    placements: tuple[Placement, ...],
    *,
    use_heuristic_mask: bool,
) -> Clip:
    """Composite every placement onto the matching background frame."""
    frames = as_clip(background, path="background")
    out = frames.copy()
    for placement in placements:
        if placement.frame_index >= out.shape[0]:
            raise ValueError(
                f"placement {placement.object_id!r} names frame {placement.frame_index}, "
                f"but the clip only has {out.shape[0]} frames."
            )
        out[placement.frame_index] = composite_frame(
            out[placement.frame_index],
            placement,
            use_heuristic_mask=use_heuristic_mask,
        )
    return out


def _clip_bbox(bbox: Box, *, height: int, width: int) -> Box:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    clipped_x1 = max(0, min(width, x1))
    clipped_y1 = max(0, min(height, y1))
    clipped_x2 = max(clipped_x1, min(width, x2))
    clipped_y2 = max(clipped_y1, min(height, y2))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _mask_for_bbox(mask: np.ndarray, bbox: Box, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"object mask must be (H, W); got {array.shape}.")
    if array.shape == (height, width):
        return array
    x1, y1, x2, y2 = _clip_bbox(bbox, height=height, width=width)
    region_h, region_w = y2 - y1, x2 - x1
    resized = cv2.resize(
        array.astype(np.float32),
        (region_w, region_h),
        interpolation=cv2.INTER_LINEAR,
    )
    full = np.zeros((height, width), dtype=np.float32)
    full[y1:y2, x1:x2] = resized
    return full
