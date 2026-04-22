from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def dw18_to_pose_results(people_dw: np.ndarray, confidence_threshold: float) -> list[Any]:
    from dwpose.types import BodyResult, Keypoint, PoseResult

    poses: list[Any] = []
    for person in people_dw:
        body_kpts: list[Any] = [None] * 18
        for op_idx, kpt in enumerate(person):
            x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
            if conf < confidence_threshold:
                continue
            # Use positional constructor to match DWPose Keypoint API expected by draw_poses.
            body_kpts[op_idx] = Keypoint(x, y, conf, op_idx)

        poses.append(
            PoseResult(
                body=BodyResult(keypoints=body_kpts),
                left_hand=None,
                right_hand=None,
                face=None,
            )
        )

    return poses


def _normalize_people_to_unit_space(people_dw: np.ndarray, width: int, height: int) -> np.ndarray:
    people = np.asarray(people_dw, dtype=np.float32)
    if people.ndim == 2:
        people = people[np.newaxis, ...]
    if people.ndim != 3 or people.shape[1:] != (18, 3):
        raise ValueError(f"Expected people_dw shape [N,18,3] or [18,3], got {tuple(people_dw.shape)}")

    normalized = people.copy()
    if int(normalized.shape[0]) == 0:
        return normalized
    if int(width) <= 1 or int(height) <= 1:
        return normalized

    x_vals = normalized[:, :, 0]
    y_vals = normalized[:, :, 1]
    if float(np.nanmax(np.abs(x_vals))) > 1.5 or float(np.nanmax(np.abs(y_vals))) > 1.5:
        normalized[:, :, 0] = normalized[:, :, 0] / float(width - 1)
        normalized[:, :, 1] = normalized[:, :, 1] / float(height - 1)

    normalized[:, :, 0] = np.clip(normalized[:, :, 0], 0.0, 1.0)
    normalized[:, :, 1] = np.clip(normalized[:, :, 1], 0.0, 1.0)
    return normalized


def _draw_dwpose_fallback(people_dw: np.ndarray, height: int, width: int, confidence_threshold: float) -> np.ndarray:
    # Keep a signed integer buffer while drawing to satisfy OpenCV type stubs.
    canvas = np.zeros((height, width, 3), dtype=np.int32)
    people = _normalize_people_to_unit_space(people_dw=people_dw, width=width, height=height)

    limb_seq = [
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (1, 11),
        (11, 12),
        (12, 13),
        (1, 0),
        (0, 14),
        (14, 16),
        (0, 15),
        (15, 17),
    ]
    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
        (255, 0, 85),
    ]
    stick_width = 4

    for person in people:
        valid = person[:, 2] >= confidence_threshold
        for limb_idx, (a, b) in enumerate(limb_seq):
            if not (valid[a] and valid[b]):
                continue

            x0 = float(person[a, 0]) * float(width)
            y0 = float(person[a, 1]) * float(height)
            x1 = float(person[b, 0]) * float(width)
            y1 = float(person[b, 1]) * float(height)

            center_x = int(round(0.5 * (x0 + x1)))
            center_y = int(round(0.5 * (y0 + y1)))
            length = float(np.hypot(x0 - x1, y0 - y1))
            angle = int(round(math.degrees(math.atan2(y0 - y1, x0 - x1))))
            polygon = cv2.ellipse2Poly(
                (center_x, center_y),
                (max(1, int(round(length * 0.5))), stick_width),
                angle,
                0,
                360,
                1,
            )
            polygon_np = np.asarray(polygon, dtype=np.int32)
            color = tuple(float(channel) for channel in colors[limb_idx])
            cv2.fillConvexPoly(canvas, polygon_np, color)

    canvas = np.asarray(canvas.astype(np.float32) * 0.6, dtype=np.uint8)

    for person in people:
        valid = person[:, 2] >= confidence_threshold
        for idx in np.where(valid)[0]:
            x = int(np.clip(round(float(person[idx, 0]) * float(width)), 0, width - 1))
            y = int(np.clip(round(float(person[idx, 1]) * float(height)), 0, height - 1))
            cv2.circle(canvas, (x, y), 4, colors[int(idx)], thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def draw_dwpose_canvas(
    height: int,
    width: int,
    people_dw: np.ndarray,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    normalized_people = _normalize_people_to_unit_space(
        people_dw=people_dw,
        width=int(width),
        height=int(height),
    )

    try:
        from dwpose import draw_poses
    except Exception:
        return _draw_dwpose_fallback(
            people_dw=normalized_people,
            height=int(height),
            width=int(width),
            confidence_threshold=float(confidence_threshold),
        )

    pose_results = dw18_to_pose_results(people_dw=normalized_people, confidence_threshold=confidence_threshold)
    try:
        return draw_poses(pose_results, height, width, draw_body=True, draw_hand=False, draw_face=False)
    except Exception:
        return _draw_dwpose_fallback(
            people_dw=normalized_people,
            height=int(height),
            width=int(width),
            confidence_threshold=float(confidence_threshold),
        )