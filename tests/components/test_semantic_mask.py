"""Tests for silhouette mask IoU and boundary integrity metric."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.components.metrics import REGISTRY
from src.components.metrics.semantic_mask import (
    SamIouMetric,
    compute_mask_iou,
)


def _circle_mask(size: int = 100, radius: int = 25) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    cv2.circle(mask.view(np.uint8), (size // 2, size // 2), radius, (1,), -1)
    return mask


def test_compute_mask_iou_identical_masks_gives_one() -> None:
    mask = _circle_mask()
    iou, prec, rec, leak, a_g, a_p = compute_mask_iou(mask, mask)

    assert pytest.approx(iou, abs=1e-6) == 1.0
    assert pytest.approx(prec, abs=1e-6) == 1.0
    assert pytest.approx(rec, abs=1e-6) == 1.0
    assert leak == 0
    assert a_g == a_p > 0


def test_compute_mask_iou_detects_leakage() -> None:
    mask_gt = _circle_mask(radius=25)
    # Dilated mask (prediction leaks into background)
    mask_pred = _circle_mask(radius=30)

    iou, prec, rec, leak, a_g, a_p = compute_mask_iou(mask_pred, mask_gt)

    assert iou < 1.0
    assert prec < 1.0  # Precision drops due to false positive bleed
    assert pytest.approx(rec, abs=1e-6) == 1.0  # Recall is complete
    assert leak > 0
    assert leak == (a_p - a_g)


def test_compute_mask_iou_detects_missing_parts() -> None:
    mask_gt = _circle_mask(radius=30)
    # Eroded mask (prediction cuts off parts of actor)
    mask_pred = _circle_mask(radius=25)

    iou, prec, rec, leak, a_g, a_p = compute_mask_iou(mask_pred, mask_gt)

    assert iou < 1.0
    assert pytest.approx(prec, abs=1e-6) == 1.0  # No background leakage
    assert rec < 1.0  # Recall drops because actor parts are lost
    assert leak == 0


def test_compute_mask_iou_disjoint_gives_zero() -> None:
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[5:15, 5:15] = True
    m2[25:35, 25:35] = True

    iou, prec, rec, leak, _, _ = compute_mask_iou(m2, m1)
    assert iou == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_sam_iou_metric_with_mock_segmenter() -> None:
    mask_gt = _circle_mask(radius=25)

    def mock_segmenter(frame: np.ndarray) -> np.ndarray:
        if frame.mean() > 100:
            return mask_gt
        # Return slightly dilated mask
        return _circle_mask(radius=28)

    metric = SamIouMetric(segmenter=mock_segmenter)

    ref_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    pred_frame = np.full((100, 100, 3), 50, dtype=np.uint8)

    score = metric.score(ref_frame, pred_frame)
    assert 0.75 < score < 1.0

    detailed = metric.score_detailed(ref_frame, pred_frame)
    assert len(detailed) == 1
    assert detailed[0].leakage_area_px > 0
    assert detailed[0].recall == 1.0


def test_sam_iou_metric_registered_in_registry() -> None:
    assert "sam_iou" in REGISTRY
    spec = REGISTRY.spec("sam_iou")
    assert spec.name == "sam_iou"
    assert "mask" in spec.capabilities

