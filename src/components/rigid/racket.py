"""Racket as a convex hull, optionally anchored to a player's wrist.

The hull is the shape. The wrist is a borrowed point from a *player* pose,
not a skeleton on the racket. This module never emits keypoints for the
racket itself.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import cv2
import numpy as np

from src.components.rigid.types import ObservedObject, PlayerPose, RigidShape
from src.contracts.errors import ConfigValueError
from src.contracts.keypoints import CANONICAL_HUMAN, schema as resolve_schema

_WRISTS = ("left_wrist", "right_wrist")


def reject_keypoints(obj: ObservedObject) -> None:
    """A rigid class carrying keypoints is a silent quality loss if ignored.

    Raising here is what makes the mistake visible at the component, matching
    the contract's rejection of ``motion.per_class.racket = keypoints``.
    """
    if obj.keypoints is None:
        return
    array = np.asarray(obj.keypoints)
    if array.size == 0:
        return
    raise ConfigValueError(
        f"rigid.{obj.object_class}",
        f"class {obj.object_class!r} has no skeleton; keypoints are not a motion "
        f"representation for it. It carries a convex hull (racket) or a blob "
        f"(ball). Player wrists, when used to anchor a racket, arrive as "
        f"PlayerPose, not as a pose on the racket.",
    )


def _hull_from_object(obj: ObservedObject) -> np.ndarray | None:
    if obj.mask is not None:
        mask = np.asarray(obj.mask, dtype=np.uint8)
        if mask.ndim != 2 or mask.size == 0:
            return None
        ys, xs = np.nonzero(mask)
        if xs.size < 3:
            return None
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        hull = cv2.convexHull(pts)
        return hull.reshape(-1, 2)
    if obj.bbox is None:
        return None
    x1, y1, x2, y2 = obj.bbox
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _wrist_near(
    poses: Sequence[PlayerPose],
    frame_index: int,
    point: tuple[float, float],
) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_dist = math.inf
    px, py = point
    for pose in poses:
        if pose.frame_index != frame_index:
            continue
        joints = np.asarray(pose.keypoints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] < 2:
            continue
        try:
            schema = resolve_schema(pose.schema_name)
        except ValueError:
            schema = CANONICAL_HUMAN
        index_of = schema.index_of
        for name in _WRISTS:
            idx = index_of.get(name)
            if idx is None or idx >= joints.shape[0]:
                continue
            x, y = float(joints[idx, 0]), float(joints[idx, 1])
            conf = float(joints[idx, 2]) if joints.shape[1] > 2 else 1.0
            if conf <= 0.1:
                continue
            dist = math.hypot(x - px, y - py)
            if dist < best_dist:
                best_dist = dist
                best = (x, y)
    return best


def extract_racket(
    obj: ObservedObject,
    player_poses: Sequence[PlayerPose] = (),
) -> RigidShape | None:
    """Convex hull, shifted so the handle sits on the nearest player wrist."""
    reject_keypoints(obj)
    hull = _hull_from_object(obj)
    if hull is None or hull.shape[0] < 3:
        return None
    centroid = (float(hull[:, 0].mean()), float(hull[:, 1].mean()))
    wrist = _wrist_near(player_poses, obj.frame_index, centroid)
    points = hull.astype(np.float64)
    if wrist is not None:
        handle = points[np.argmin(np.hypot(points[:, 0] - wrist[0], points[:, 1] - wrist[1]))]
        offset_x = wrist[0] - float(handle[0])
        offset_y = wrist[1] - float(handle[1])
        points = points + np.array([offset_x, offset_y], dtype=np.float64)
    packed = tuple((float(x), float(y)) for x, y in points)
    return RigidShape(
        object_id=obj.object_id,
        object_class="racket",
        kind="hull",
        frame_index=obj.frame_index,
        points=packed,
        wrist_anchor=wrist,
    )
