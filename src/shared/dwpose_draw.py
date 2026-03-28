from __future__ import annotations

from typing import Any

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


def draw_dwpose_canvas(
    height: int,
    width: int,
    people_dw: np.ndarray,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    from dwpose import draw_poses

    pose_results = dw18_to_pose_results(people_dw=people_dw, confidence_threshold=confidence_threshold)
    return draw_poses(pose_results, height, width, draw_body=True, draw_hand=False, draw_face=False)