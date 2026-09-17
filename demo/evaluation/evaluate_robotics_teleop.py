"""Robotics & teleoperation task utility evaluator: hand detection rate and MPJPE joint error."""

from __future__ import annotations

import sqlite3  # noqa: F401
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand

MATCH_CENTROID_PX = 350.0
PCK_PX = 50.0


def score_pose_tracks(
    gt_poses: list[FrameHandPose],
    pred_poses: list[FrameHandPose],
    match_px: float = MATCH_CENTROID_PX,
    pck_px: float = PCK_PX,
) -> dict[str, float]:
    """Compare predicted 21-joint tracks to a ground-truth track.

    Detection is GT-hand recall (centroid match). MPJPE is *conditional* on a
    match (survivorship). ``pck50_all_gt`` treats an unmatched GT hand as 21
    missed joints so it cannot improve by dropping hard frames.
    """
    n_frames = min(len(gt_poses), len(pred_poses))
    empty = {
        "detection_rate": 0.0,
        "oracle_capture_ratio": 0.0,
        "mpjpe_pixels": 0.0,
        "mean_confidence": 0.0,
        "handedness_agreement": 0.0,
        "pck50_matched": 0.0,
        "pck50_all_gt": 0.0,
        "gt_hands": 0.0,
        "pred_hands": 0.0,
        "matched_hands": 0.0,
    }
    if n_frames == 0:
        return empty

    gt_hands_total = 0
    pred_hands_total = 0
    detected_hands_count = 0
    handedness_agreed_count = 0
    joint_errors: list[float] = []
    confidences: list[float] = []
    pck_hits_matched = 0
    pck_total_matched = 0
    pck_hits_all = 0
    pck_total_all = 0

    for i in range(n_frames):
        unmatched_pred = list(pred_poses[i].hands)
        pred_hands_total += len(pred_poses[i].hands)

        for r_hand in gt_poses[i].hands:
            gt_hands_total += 1
            r_pts = np.array(r_hand.landmarks_pixel, dtype=np.float64)
            r_centroid = np.mean(r_pts, axis=0)
            best_d_hand, best_dist, best_idx = _nearest_hand(r_hand, r_centroid, unmatched_pred, match_px)

            pck_total_all += 21
            if best_d_hand is None:
                continue

            detected_hands_count += 1
            unmatched_pred.pop(best_idx)
            confidences.append(best_d_hand.confidence)
            if best_d_hand.handedness.lower() == r_hand.handedness.lower():
                handedness_agreed_count += 1

            d_pts = np.array(best_d_hand.landmarks_pixel, dtype=np.float64)
            n_joints = min(len(r_pts), len(d_pts))
            per_joint = np.linalg.norm(r_pts[:n_joints] - d_pts[:n_joints], axis=1)
            joint_errors.append(float(np.mean(per_joint)))
            hits = int(np.sum(per_joint <= pck_px))
            pck_hits_matched += hits
            pck_total_matched += n_joints
            pck_hits_all += hits

    detection_rate = float(detected_hands_count / gt_hands_total) if gt_hands_total > 0 else 1.0
    return {
        "detection_rate": detection_rate,
        "oracle_capture_ratio": detection_rate,
        "mpjpe_pixels": float(np.mean(joint_errors)) if joint_errors else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "handedness_agreement": (
            float(handedness_agreed_count / detected_hands_count) if detected_hands_count > 0 else 1.0
        ),
        "pck50_matched": float(pck_hits_matched / pck_total_matched) if pck_total_matched else 0.0,
        "pck50_all_gt": float(pck_hits_all / pck_total_all) if pck_total_all else 0.0,
        "gt_hands": float(gt_hands_total),
        "pred_hands": float(pred_hands_total),
        "matched_hands": float(detected_hands_count),
    }


def _nearest_hand(
    r_hand: SingleHand,
    r_centroid: np.ndarray,
    unmatched: list[SingleHand],
    match_px: float,
) -> tuple[SingleHand | None, float, int]:
    best_d_hand = None
    best_dist = float("inf")
    best_idx = -1
    r_side = r_hand.handedness.lower()
    for d_idx, d_hand in enumerate(unmatched):
        d_pts = np.array(d_hand.landmarks_pixel, dtype=np.float64)
        d_centroid = np.mean(d_pts, axis=0)
        dist = float(np.linalg.norm(r_centroid - d_centroid))
        same_side = d_hand.handedness.lower() in {r_side, "unknown"} or r_side == "unknown"
        effective_dist = dist if same_side else dist * 1.5
        if dist < match_px and effective_dist < best_dist:
            best_dist = effective_dist
            best_d_hand = d_hand
            best_idx = d_idx
    return best_d_hand, best_dist, best_idx


def evaluate_teleop_utility(
    ref_poses: list[FrameHandPose],
    rec_video_path: Path,
    max_frames: int | None = None,
) -> dict[str, float]:
    """Legacy display-path score: MediaPipe Hands on the reconstruction vs ``ref_poses``."""
    from demo.pipeline.hand_keypoints import extract_video_hand_poses

    rec_poses = extract_video_hand_poses(rec_video_path, max_frames=max_frames)
    return score_pose_tracks(ref_poses, rec_poses)
