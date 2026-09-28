"""Model-specific condition renderers over shared PointStream observations.

These adapters define training inputs; they do not claim that existing model
checkpoints were trained for racket or joint conditioning. New racket-aware
SPADE/ControlNet checkpoints must be trained and validated against these maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from src.components.perception.coordinates import CropTransform
from src.components.perception.conditioning import render_pose_condition
from src.components.pose.wire import Pose, to_wire
from src.contracts.keypoints import OPENPOSE_18


@dataclass(frozen=True)
class ModelAdapterViews:
    """Canvas-aligned model inputs and their named geometric channels."""

    animate_anyone_pose_rgb: np.ndarray | None
    spade_condition_rgb: np.ndarray
    controlnet_condition_rgb: np.ndarray
    controlnet_channels: dict[str, np.ndarray]
    schema: str
    eligible_for_cross_training: bool
    exclusion_reason: str | None = None


def render_model_adapter_views(
    *,
    view: str,
    transform: CropTransform,
    player_mask: np.ndarray,
    racket_mask: np.ndarray,
    pose: Pose | None = None,
    racket_geometry: Any = None,
) -> ModelAdapterViews:
    """Render AnimateAnyone, SPADE, and ControlNet inputs on one crop canvas.

    ControlNet channels are grayscale and independently named: player mask,
    racket mask, wrist-to-tip axis, and transverse racket width. The companion
    RGB image packs those channels as R, G, B respectively, with the axis and
    width unioned in blue. SPADE gets an RGB display map: OpenPose-18 for a
    player, or a white racket silhouette with cyan axis and magenta width.
    A joint SPADE view overlays the OpenPose-18 skeleton and racket geometry.
    """
    if view not in {"player", "racket", "joint"}:
        raise ValueError("view must be player, racket, or joint")
    expected = (transform.target_height, transform.target_width)
    player = _canvas_mask(player_mask, expected, "player_mask")
    racket = _canvas_mask(racket_mask, expected, "racket_mask")

    animate_pose: np.ndarray | None = None
    openpose_pose: np.ndarray | None = None
    if pose is not None:
        projected = to_wire(pose, OPENPOSE_18)
        openpose_pose = render_pose_condition(projected, transform)
        if view == "player":
            animate_pose = openpose_pose.copy()

    axis = np.zeros(expected, dtype=np.uint8)
    width = np.zeros(expected, dtype=np.uint8)
    shape_kind = getattr(racket_geometry, "kind", None)
    shape_points = getattr(racket_geometry, "points", ())
    has_cross = shape_kind == "racket_cross_v1" and len(shape_points) == 4
    if has_cross:
        points = np.rint(transform.source_to_canvas(np.asarray(shape_points))).astype(int)
        def _point(row: np.ndarray) -> tuple[int, int]:
            return (int(row[0]), int(row[1]))

        cast(Any, cv2).line(axis, _point(points[0]), _point(points[1]), 255, 2, cv2.LINE_AA)
        cast(Any, cv2).line(width, _point(points[2]), _point(points[3]), 255, 2, cv2.LINE_AA)

    # Channels remain separate for new model heads. Existing 3-channel
    # checkpoints cannot consume them without a compatible adapter/training.
    control_rgb = np.stack(
        [player * 255, racket * 255, np.maximum(axis, width)], axis=-1
    ).astype(np.uint8)
    spade = np.zeros((*expected, 3), dtype=np.uint8)
    if view == "player":
        if openpose_pose is not None:
            spade = openpose_pose.copy()
    elif view == "racket":
        spade[racket != 0] = (235, 235, 235)
        _draw_geometry(spade, points if has_cross else None)
    else:
        if openpose_pose is not None:
            spade = openpose_pose.copy()
        spade[racket != 0] = np.maximum(spade[racket != 0], (160, 80, 20))
        _draw_geometry(spade, points if has_cross else None)

    eligible = view == "player" or has_cross
    reason = None if eligible else "missing_or_fallback_racket_cross"
    return ModelAdapterViews(
        animate_anyone_pose_rgb=animate_pose,
        spade_condition_rgb=spade,
        controlnet_condition_rgb=control_rgb,
        controlnet_channels={
            "player_mask": (player * 255).astype(np.uint8),
            "racket_mask": (racket * 255).astype(np.uint8),
            "racket_axis": axis,
            "racket_width": width,
        },
        schema="pointstream.model-adapter-views.v1",
        eligible_for_cross_training=eligible,
        exclusion_reason=reason,
    )


def _canvas_mask(mask: np.ndarray, expected: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != expected:
        raise ValueError(f"{name} shape {array.shape} does not match canvas {expected}")
    return (array != 0).astype(np.uint8)


def _draw_geometry(image: np.ndarray, points: np.ndarray | None) -> None:
    if points is None:
        return
    cv2.line(image, tuple(points[0]), tuple(points[1]), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, tuple(points[2]), tuple(points[3]), (255, 0, 255), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)


__all__ = ["ModelAdapterViews", "render_model_adapter_views"]
