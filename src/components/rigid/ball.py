"""Ball as a difference blob or a segmentation blob. No skeleton."""

from __future__ import annotations

from collections.abc import Sequence
import math

import cv2
import numpy as np

from src.components.rigid.racket import reject_keypoints
from src.components.rigid.types import ObservedObject, RigidShape


def _centroid_and_radius(mask: np.ndarray) -> tuple[tuple[float, float], float] | None:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return None
    moments = cv2.moments(binary)
    area = float(moments["m00"])
    if area < 1.0:
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
        radius = max(1.0, math.sqrt(float(xs.size) / math.pi))
        return (cx, cy), radius
    cx = float(moments["m10"] / area)
    cy = float(moments["m01"] / area)
    radius = max(1.0, math.sqrt(area / math.pi))
    return (cx, cy), radius


def extract_ball_segmentation(obj: ObservedObject) -> RigidShape | None:
    """Centroid of a provided mask, or of the bbox treated as a rectangle."""
    reject_keypoints(obj)
    mask = obj.mask
    if mask is None and obj.bbox is not None:
        x1, y1, x2, y2 = obj.bbox
        x1i, y1i = int(max(0, x1)), int(max(0, y1))
        x2i, y2i = int(max(x1i + 1, x2)), int(max(y1i + 1, y2))
        synthetic = np.zeros((y2i, x2i), dtype=np.uint8)
        synthetic[y1i:y2i, x1i:x2i] = 255
        mask = synthetic
    if mask is None:
        return None
    found = _centroid_and_radius(np.asarray(mask))
    if found is None:
        return None
    (cx, cy), radius = found
    return RigidShape(
        object_id=obj.object_id,
        object_class="ball",
        kind="segmentation",
        frame_index=obj.frame_index,
        points=((cx, cy),),
        radius=radius,
    )


def _exclusion_union(
    objects: Sequence[ObservedObject],
    frame_index: int,
    height: int,
    width: int,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        if obj.frame_index != frame_index:
            continue
        if obj.object_class not in {"player", "person", "racket"}:
            continue
        if obj.mask is not None:
            mask = np.asarray(obj.mask, dtype=np.uint8)
            if mask.shape == (height, width):
                canvas = np.bitwise_or(canvas, (mask > 0).astype(np.uint8) * 255)
                continue
        if obj.bbox is None:
            continue
        x1, y1, x2, y2 = obj.bbox
        xa, ya = int(max(0, x1)), int(max(0, y1))
        xb, yb = int(min(width, x2)), int(min(height, y2))
        if xb > xa and yb > ya:
            canvas[ya:yb, xa:xb] = 255
    return canvas


def extract_ball_difference(
    frame: np.ndarray,
    background: np.ndarray | None,
    objects: Sequence[ObservedObject],
    *,
    frame_index: int,
    threshold: float = 18.0,
    min_area: int = 6,
    object_id: str = "ball_0",
) -> RigidShape | None:
    """Largest difference blob after subtracting a plate and actor masks."""
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a BGR frame [H, W, 3], got {tuple(image.shape)}.")
    height, width = int(image.shape[0]), int(image.shape[1])
    if background is None:
        return None
    plate = np.asarray(background, dtype=np.uint8)
    if plate.shape[:2] != (height, width):
        plate = np.asarray(
            cv2.resize(plate, (width, height), interpolation=cv2.INTER_LINEAR),
            dtype=np.uint8,
        )
    gray_f = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gray_f - gray_b)
    exclusion = _exclusion_union(objects, frame_index, height, width)
    diff[exclusion > 0] = 0
    binary = (diff >= threshold).astype(np.uint8) * 255
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    best_i = -1
    best_area = 0
    for index in range(1, n_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_area = area
            best_i = index
    if best_i < 0:
        return None
    cx, cy = float(centroids[best_i][0]), float(centroids[best_i][1])
    radius = max(1.0, math.sqrt(best_area / math.pi))
    return RigidShape(
        object_id=object_id,
        object_class="ball",
        kind="difference",
        frame_index=frame_index,
        points=((cx, cy),),
        radius=radius,
    )
