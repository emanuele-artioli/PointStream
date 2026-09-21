"""Tests for Object Keypoint Similarity (OKS) and pose drift metric."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.metrics import REGISTRY
from src.components.metrics.pose import (
    PoseMetric,
    compute_oks,
)


def _coco17_test_points() -> np.ndarray:
    """Standard synthetic 17-keypoint human skeleton."""
    return np.array(
        [
            [100.0, 50.0],  # 0: nose
            [95.0, 45.0],  # 1: left eye
            [105.0, 45.0],  # 2: right eye
            [90.0, 50.0],  # 3: left ear
            [110.0, 50.0],  # 4: right ear
            [80.0, 80.0],  # 5: left shoulder
            [120.0, 80.0],  # 6: right shoulder
            [70.0, 110.0],  # 7: left elbow
            [130.0, 110.0],  # 8: right elbow
            [60.0, 140.0],  # 9: left wrist
            [140.0, 140.0],  # 10: right wrist
            [85.0, 150.0],  # 11: left hip
            [115.0, 150.0],  # 12: right hip
            [85.0, 200.0],  # 13: left knee
            [115.0, 200.0],  # 14: right knee
            [85.0, 250.0],  # 15: left ankle
            [115.0, 250.0],  # 16: right ankle
        ],
        dtype=np.float64,
    )


def test_compute_oks_identical_points_gives_one() -> None:
    pts = _coco17_test_points()
    bbox = (50.0, 40.0, 150.0, 260.0)  # scale ~ 148 px
    oks, drift, dropouts, num_vis, conf_diff = compute_oks(pts, pts, bbox_gt=bbox)

    assert pytest.approx(oks, abs=1e-6) == 1.0
    assert pytest.approx(drift, abs=1e-6) == 0.0
    assert dropouts == 0
    assert num_vis == 17
    assert pytest.approx(conf_diff, abs=1e-6) == 0.0


def test_oks_and_drift_are_monotonic_with_displacement() -> None:
    pts_gt = _coco17_test_points()
    bbox = (50.0, 40.0, 150.0, 260.0)

    displacements = [1.0, 3.0, 8.0, 20.0, 50.0]
    oks_scores = []
    drifts = []

    for d in displacements:
        shifted = pts_gt + d
        oks, drift, dropouts, _, _ = compute_oks(shifted, pts_gt, bbox_gt=bbox)
        oks_scores.append(oks)
        drifts.append(drift)
        assert dropouts == 0

    # OKS strictly decreases as displacement grows
    for i in range(len(oks_scores) - 1):
        assert oks_scores[i] > oks_scores[i + 1]

    # Drift strictly increases as displacement grows
    for i in range(len(drifts) - 1):
        assert drifts[i] < drifts[i + 1]

    # Small displacement (1px) preserves very high OKS
    assert oks_scores[0] > 0.97
    # Large displacement (50px) drops OKS severely
    assert oks_scores[-1] < 0.10


def test_keypoint_dropout_penalizes_oks() -> None:
    pts_gt = _coco17_test_points()
    pts_pred = pts_gt.copy()
    bbox = (50.0, 40.0, 150.0, 260.0)

    # 17 keypoints with confidences
    conf_gt = np.ones(17)
    conf_pred = np.ones(17)

    # Drop 4 keypoints (e.g. lost arm or leg in generated frame)
    conf_pred[7:11] = 0.0  # elbows and wrists lost

    oks, drift, dropouts, num_vis, conf_diff = compute_oks(
        pts_pred, pts_gt, bbox_gt=bbox, conf_pred=conf_pred, conf_gt=conf_gt
    )

    assert dropouts == 4
    assert num_vis == 17
    assert conf_diff < 0.0
    # OKS is penalized proportionally for the lost limbs
    assert oks < (13 / 17) + 0.01


def test_pose_metric_with_mock_estimator() -> None:
    pts_gt = _coco17_test_points()
    confs = np.ones(17)
    bbox = (50.0, 40.0, 150.0, 260.0)

    def mock_estimator(frame: np.ndarray):
        # Return GT pose if mean pixel > 100, slightly displaced if mean pixel < 100
        if frame.mean() > 100:
            return pts_gt, confs, bbox
        return pts_gt + 2.0, confs, bbox

    metric = PoseMetric(estimator=mock_estimator)

    ref_frame = np.full((300, 300, 3), 150, dtype=np.uint8)
    pred_frame = np.full((300, 300, 3), 50, dtype=np.uint8)

    score = metric.score(ref_frame, pred_frame)
    assert 0.90 < score < 1.0

    detailed = metric.score_detailed(ref_frame, pred_frame)
    assert len(detailed) == 1
    assert detailed[0].ref_detected
    assert detailed[0].pred_detected
    assert detailed[0].dropout_count == 0
    assert detailed[0].mean_drift_norm > 0.0


def test_pose_metric_registered_in_registry() -> None:
    assert "pose_oks" in REGISTRY
    spec = REGISTRY.spec("pose_oks")
    assert spec.name == "pose_oks"
    assert "pose" in spec.capabilities
