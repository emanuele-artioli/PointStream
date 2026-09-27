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
) -> tuple[tuple[float, float], str, str] | None:
    best: tuple[tuple[float, float], str, str] | None = None
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
                best = ((x, y), pose.object_id, name)
    return best


def extract_racket(
    obj: ObservedObject,
    player_poses: Sequence[PlayerPose] = (),
) -> RigidShape | None:
    """Return the observed hull unchanged, with an optional nearby wrist link.

    The wrist is association metadata. It never translates the segmentation
    mask or hull, since doing so would turn an observed shape into a fabricated
    one.
    """
    reject_keypoints(obj)
    hull = _hull_from_object(obj)
    if hull is None or hull.shape[0] < 3:
        return None
    centroid = (float(hull[:, 0].mean()), float(hull[:, 1].mean()))
    association = _associated_wrist(obj, player_poses, centroid)
    wrist = association[0] if association is not None else None
    points = hull.astype(np.float64)
    packed = tuple((float(x), float(y)) for x, y in points)
    return RigidShape(
        object_id=obj.object_id,
        object_class="racket",
        kind="hull",
        frame_index=obj.frame_index,
        points=packed,
        wrist_anchor=wrist,
        associated_player_id=association[1] if association is not None else None,
        associated_wrist=association[2] if association is not None else None,
    )


def extract_racket_cross(
    obj: ObservedObject,
    player_poses: Sequence[PlayerPose] = (),
    *,
    width_position: float = 0.75,
    minimum_width: float = 1.0,
) -> RigidShape | None:
    """Create a wrist/tip/transverse-width cross from the continuous hull.

    Point order is ``(wrist, tip, width_negative, width_positive)``. The
    transverse endpoints are ordered along the perpendicular to the directed
    wrist-to-tip axis, which keeps their meaning stable as contour sampling
    changes. If the wrist or the geometry is unavailable, return a separately
    typed hull fallback so cross-conditioned training can exclude it.
    """
    if not 0.0 < width_position < 1.0:
        raise ValueError("width_position must be between 0 and 1")
    reject_keypoints(obj)
    hull = _hull_from_object(obj)
    if hull is None or hull.shape[0] < 3:
        return None
    center = (float(hull[:, 0].mean()), float(hull[:, 1].mean()))
    association = _associated_wrist(obj, player_poses, center)
    if association is None:
        return _hull_fallback(obj, hull, None, "missing_visible_wrist")
    wrist, player_id, wrist_name = association
    points = hull.astype(np.float64)
    tip_index = int(np.argmax(np.sum((points - np.asarray(wrist)) ** 2, axis=1)))
    tip = points[tip_index]
    axis = tip - np.asarray(wrist, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 1e-6:
        return _hull_fallback(obj, hull, association, "degenerate_wrist_to_tip_axis")
    axis /= length
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    cross_center = np.asarray(wrist, dtype=np.float64) + width_position * (tip - wrist)
    intersections = _line_polygon_intersections(points, cross_center, normal)
    if len(intersections) < 2:
        return _hull_fallback(obj, hull, association, "no_transverse_polygon_intersection")
    ordered = sorted(intersections, key=lambda p: float(np.dot(p - cross_center, normal)))
    low, high = ordered[0], ordered[-1]
    width = float(np.linalg.norm(high - low))
    if not np.isfinite(width) or width < minimum_width:
        return _hull_fallback(obj, hull, association, "degenerate_transverse_width")
    packed = tuple(
        (float(point[0]), float(point[1]))
        for point in (np.asarray(wrist), tip, low, high)
    )
    return RigidShape(
        object_id=obj.object_id,
        object_class="racket",
        kind="racket_cross_v1",
        frame_index=obj.frame_index,
        points=packed,
        wrist_anchor=wrist,
        associated_player_id=player_id,
        associated_wrist=wrist_name,
        endpoint_order="directed_axis_perpendicular_negative_then_positive",
    )


def _associated_wrist(
    obj: ObservedObject,
    poses: Sequence[PlayerPose],
    point: tuple[float, float],
) -> tuple[tuple[float, float], str, str] | None:
    candidates = [
        pose
        for pose in poses
        if pose.frame_index == obj.frame_index
        and (obj.associated_player_id is None or pose.object_id == obj.associated_player_id)
    ]
    selected = _wrist_near(candidates, obj.frame_index, point)
    if selected is None:
        return None
    wrist, player_id, wrist_name = selected
    if obj.associated_wrist is not None and wrist_name != obj.associated_wrist:
        return None
    # A distant wrist is not evidence of association.  The floor covers small
    # objects while the diagonal term adapts the limit to the current mask.
    hull = _hull_from_object(obj)
    if hull is not None and hull.size:
        span = np.ptp(hull, axis=0)
        max_distance = max(32.0, 0.25 * float(np.linalg.norm(span)))
        distance = min(float(np.linalg.norm(np.asarray(wrist) - p)) for p in hull)
        if distance > max_distance:
            return None
    return selected


def _hull_fallback(
    obj: ObservedObject,
    hull: np.ndarray,
    association: tuple[tuple[float, float], str, str] | None,
    reason: str,
) -> RigidShape:
    return RigidShape(
        object_id=obj.object_id,
        object_class="racket",
        kind="hull_fallback_v1",
        frame_index=obj.frame_index,
        points=tuple((float(x), float(y)) for x, y in hull),
        wrist_anchor=association[0] if association else None,
        associated_player_id=association[1] if association else None,
        associated_wrist=association[2] if association else None,
        fallback_reason=reason,
    )


def _line_polygon_intersections(
    polygon: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> list[np.ndarray]:
    """Intersect an infinite line with polygon edges, independent of vertex count."""
    result: list[np.ndarray] = []
    normal = np.array([-line_direction[1], line_direction[0]], dtype=np.float64)
    signed = np.dot(polygon - line_point, normal)
    n = len(polygon)
    for index in range(n):
        a = polygon[index]
        b = polygon[(index + 1) % n]
        da, db = float(signed[index]), float(signed[(index + 1) % n])
        if abs(da) <= epsilon:
            result.append(a.copy())
        if da * db < -(epsilon * epsilon):
            fraction = da / (da - db)
            result.append(a + fraction * (b - a))
        if abs(da) <= epsilon and abs(db) <= epsilon:
            result.append(b.copy())
    unique: list[np.ndarray] = []
    for point in result:
        if not any(float(np.linalg.norm(point - other)) <= 1e-6 for other in unique):
            unique.append(point)
    return unique
