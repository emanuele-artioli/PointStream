"""Shared object appearance, mask, and pose conditioning renderers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.components.perception.coordinates import (
    CropTransform,
    render_object_view,
)
from src.components.pose.wire import Pose


@dataclass(frozen=True)
class ConditioningView:
    appearance: np.ndarray
    mask: np.ndarray
    pose: np.ndarray | None
    appearance_transform: CropTransform
    pose_transform: CropTransform | None


def render_conditioning_view(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    target_size: tuple[int, int] = (512, 512),
    pose: Pose | None = None,
    pose_transform: CropTransform | None = None,
) -> ConditioningView:
    """Render aligned black-background appearance, mask, and pose condition.

    ``pose_transform`` may have a different source size from the appearance
    transform when a pose was estimated on a differently sized source image.
    Both transforms must target the same model canvas; each remains separately
    recorded so coordinates can be restored without guessing a scale.
    """
    appearance, condition_mask, appearance_transform = render_object_view(
        frame,
        mask,
        bbox,
        target_size=target_size,
    )
    if pose is None:
        if pose_transform is not None:
            raise ValueError("pose_transform cannot be supplied without a pose")
        return ConditioningView(
            appearance,
            condition_mask,
            None,
            appearance_transform,
            None,
        )
    active_pose_transform = pose_transform or appearance_transform
    if (active_pose_transform.target_width, active_pose_transform.target_height) != (
        appearance_transform.target_width,
        appearance_transform.target_height,
    ):
        raise ValueError("appearance and pose transforms must target the same canvas size")
    pose_image = render_pose_condition(pose, active_pose_transform)
    return ConditioningView(
        appearance,
        condition_mask,
        pose_image,
        appearance_transform,
        active_pose_transform,
    )


def render_pose_condition(pose: Pose, transform: CropTransform) -> np.ndarray:
    """Draw a deterministic RGB skeleton image using the shared crop transform."""
    canvas = np.zeros((transform.target_height, transform.target_width, 3), dtype=np.uint8)
    points = transform.source_to_canvas(pose.values[:, :2])
    visible = np.asarray(pose.visibility) > 0
    indexes = pose.schema.index_of
    for start_name, end_name in pose.schema.edges:
        start, end = indexes[start_name], indexes[end_name]
        if not (visible[start] and visible[end]):
            continue
        a = tuple(np.rint(points[start]).astype(int))
        b = tuple(np.rint(points[end]).astype(int))
        cv2.line(canvas, a, b, (255, 96, 64), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if visible[index]:
            xy = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, xy, 2, (64, 224, 255), -1, cv2.LINE_AA)
    return canvas


__all__ = ["ConditioningView", "render_conditioning_view", "render_pose_condition"]
