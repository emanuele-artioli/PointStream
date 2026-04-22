"""
DWPose module for PointStream.

Provides whole-body pose estimation (body, hands, face) using ONNX models,
and skeleton drawing utilities compatible with AnimateAnyone's expected format.

Adapted from Moore-AnimateAnyone's DWPose implementation (IDEA-Research/DWPose).
"""

import numpy as np

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    """
    Draw a DWPose skeleton on a black canvas.

    Args:
        pose: dict with keys 'bodies', 'hands', 'faces'.
              bodies has 'candidate' (N*18, 2) normalized and 'subset' (N, 18) index-or-neg1.
              hands is (2, 21, 2) normalized.
              faces is (1, 68, 2) normalized.
        H: output canvas height
        W: output canvas width

    Returns:
        canvas: (H, W, 3) uint8 BGR image.
    """
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    return canvas


def extract_best_person(keypoints, scores, score_threshold=0.3):
    """
    Select the person with the highest mean body-keypoint score.

    Args:
        keypoints: (N, 134, 2) pixel coordinates from Wholebody.
        scores: (N, 134) confidence scores.
        score_threshold: visibility threshold.

    Returns:
        (best_keypoints, best_scores) – arrays of shape (134, 2) and (134,),
        or (None, None) when no person was detected.
    """
    if len(keypoints) == 0:
        return None, None

    body_scores = scores[:, :18]
    best_idx = int(np.mean(body_scores, axis=-1).argmax())
    return keypoints[best_idx], scores[best_idx]


def keypoints_to_pose_dict(keypoints, scores, W, H, score_threshold=0.3):
    """
    Convert raw DWPose keypoints/scores into the pose dict consumed by ``draw_pose``.

    The conversion mirrors the post-processing in AnimateAnyone's DWposeDetector.__call__:
    normalise to [0, 1], apply visibility masking, split into body / foot / face / hands,
    and build the index-based subset array expected by ``draw_bodypose``.

    Args:
        keypoints: (134, 2) pixel coordinates (one person).
        scores: (134,) confidence scores.
        W: detection image width (for normalisation).
        H: detection image height (for normalisation).
        score_threshold: below this, keypoints are marked invisible.

    Returns:
        pose dict: {bodies: {candidate, subset}, hands, faces}.
    """
    kpts = keypoints.copy().astype(float)
    scrs = scores.copy()

    # Normalise to [0, 1]
    kpts[:, 0] /= float(W)
    kpts[:, 1] /= float(H)

    # Mark invisible keypoints
    invisible = scrs < score_threshold
    kpts[invisible] = -1

    # Body (0-17), foot (18-23), face (24-91), left hand (92-112), right hand (113-133)
    body = kpts[:18].copy()
    faces = kpts[24:92].reshape(1, -1, 2)
    hands = np.stack([kpts[92:113], kpts[113:134]])  # (2, 21, 2)

    # Build index-based subset for draw_bodypose
    body_scores = scrs[:18]
    subset = np.zeros((1, 18))
    for j in range(18):
        subset[0][j] = j if body_scores[j] > score_threshold else -1

    return {
        "bodies": {"candidate": body, "subset": subset},
        "hands": hands,
        "faces": faces,
    }
