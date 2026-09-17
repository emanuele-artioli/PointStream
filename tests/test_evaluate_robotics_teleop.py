"""Unit tests for GT-relative teleop scoring (no MediaPipe / no video)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


def _hand(x: float, y: float, jitter: float = 0.0) -> SingleHand:
    rng = np.random.default_rng(0)
    pts = [[x + 10 * i + float(rng.normal(0, jitter)), y + float(rng.normal(0, jitter))] for i in range(21)]
    return SingleHand(
        handedness="Right",
        confidence=0.9,
        bbox=[int(x), int(y), int(x) + 210, int(y) + 40],
        landmarks_norm=[[p[0] / 1920, p[1] / 1080, 0.0] for p in pts],
        landmarks_pixel=pts,
    )


def test_perfect_match_is_full_recall_and_zero_error() -> None:
    gt = [FrameHandPose(0, [_hand(400, 400)])]
    pred = [FrameHandPose(0, [_hand(400, 400)])]
    s = score_pose_tracks(gt, pred)
    assert s["detection_rate"] == 1.0
    assert s["mpjpe_pixels"] < 1e-6
    assert s["pck50_all_gt"] == 1.0


def test_unmatched_gt_hand_zeros_all_gt_pck_but_not_matched_mpjpe() -> None:
    gt = [FrameHandPose(0, [_hand(400, 400), _hand(900, 400)])]
    pred = [FrameHandPose(0, [_hand(400, 400)])]
    s = score_pose_tracks(gt, pred)
    assert s["detection_rate"] == 0.5
    assert s["matched_hands"] == 1
    assert s["pck50_matched"] == 1.0
    assert abs(s["pck50_all_gt"] - 0.5) < 1e-6
    assert s["mpjpe_pixels"] < 1e-6


def test_far_prediction_is_a_miss() -> None:
    gt = [FrameHandPose(0, [_hand(100, 100)])]
    pred = [FrameHandPose(0, [_hand(1600, 900)])]
    s = score_pose_tracks(gt, pred)
    assert s["detection_rate"] == 0.0
    assert s["pck50_all_gt"] == 0.0
    assert s["mpjpe_pixels"] == 0.0
