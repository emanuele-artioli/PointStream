"""Robotics & teleoperation task utility evaluator: hand detection rate and MPJPE joint error."""

from __future__ import annotations

import sqlite3  # noqa: F401
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from demo.pipeline.hand_keypoints import FrameHandPose, extract_video_hand_poses


def evaluate_teleop_utility(
    ref_poses: list[FrameHandPose],
    rec_video_path: Path,
    max_frames: int | None = None,
) -> dict[str, float]:
    """Measures hand detection rate, joint tracking error (MPJPE), and oracle capture ratio.

    Uses spatial proximity and handedness to match reference and reconstructed hands,
    preventing egocentric mirror flips from being penalized as 0% detections while
    tracking handedness accuracy as an independent metric.
    """
    rec_poses = extract_video_hand_poses(rec_video_path, max_frames=max_frames)

    n_frames = min(len(ref_poses), len(rec_poses))
    if n_frames == 0:
        return {
            "detection_rate": 0.0,
            "oracle_capture_ratio": 0.0,
            "mpjpe_pixels": 0.0,
            "mean_confidence": 0.0,
            "handedness_agreement": 0.0,
        }

    ref_hands_total = 0
    detected_hands_count = 0
    handedness_agreed_count = 0
    joint_errors: list[float] = []
    confidences: list[float] = []

    for i in range(n_frames):
        r_frame = ref_poses[i]
        d_frame = rec_poses[i]

        unmatched_d_hands = list(d_frame.hands)

        for r_hand in r_frame.hands:
            ref_hands_total += 1
            r_pts = np.array(r_hand.landmarks_pixel)  # [21, 2]
            r_centroid = np.mean(r_pts, axis=0)

            best_d_hand = None
            best_dist = float("inf")
            best_idx = -1

            # Match priority: same handedness within proximity threshold, then closest hand
            for d_idx, d_hand in enumerate(unmatched_d_hands):
                d_pts = np.array(d_hand.landmarks_pixel)
                d_centroid = np.mean(d_pts, axis=0)
                dist = float(np.linalg.norm(r_centroid - d_centroid))

                # Preference bonus for matching handedness
                effective_dist = dist if d_hand.handedness == r_hand.handedness else dist * 1.5

                # Threshold: hand centroid must be within 350 pixels on 1080p frame
                if dist < 350.0 and effective_dist < best_dist:
                    best_dist = effective_dist
                    best_d_hand = d_hand
                    best_idx = d_idx

            if best_d_hand is not None:
                detected_hands_count += 1
                confidences.append(best_d_hand.confidence)
                if best_d_hand.handedness == r_hand.handedness:
                    handedness_agreed_count += 1

                # Compute MPJPE across the 21 landmarks
                d_pts = np.array(best_d_hand.landmarks_pixel)
                per_joint_dist = np.linalg.norm(r_pts - d_pts, axis=1)  # [21]
                joint_errors.append(float(np.mean(per_joint_dist)))

                # Remove from available pool for this frame
                unmatched_d_hands.pop(best_idx)

    detection_rate = float(detected_hands_count / ref_hands_total) if ref_hands_total > 0 else 1.0
    mean_mpjpe = float(np.mean(joint_errors)) if joint_errors else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    handedness_agreement = float(handedness_agreed_count / detected_hands_count) if detected_hands_count > 0 else 1.0

    return {
        "detection_rate": detection_rate,
        "oracle_capture_ratio": detection_rate,  # fraction of ground-truth reference hands successfully recovered
        "mpjpe_pixels": mean_mpjpe,
        "mean_confidence": mean_conf,
        "handedness_agreement": handedness_agreement,
    }

