"""Unit tests for E04B audit and derived reporting functions."""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
import numpy as np

from scripts.e04b_audit_and_derived_report import compute_ghost_and_boundary_metrics


def test_compute_ghost_and_boundary_metrics_identical_inputs() -> None:
    """When ref and pred are identical, MAD should be zero and SSIM should be 1.0."""
    t, h, w = 3, 64, 64
    ref_rgb = np.full((t, h, w, 3), 128, dtype=np.uint8)
    pred_rgb = np.full((t, h, w, 3), 128, dtype=np.uint8)

    # masks: frame 0 has actor mask, frames 1..t-1 do not (creating ghost region m0 & ~mt)
    masks = np.zeros((t, h, w), dtype=bool)
    masks[0, 10:30, 10:30] = True

    bnd_masks = np.zeros((t, h, w), dtype=bool)
    bnd_masks[:, 5:10, 5:10] = True

    metrics = compute_ghost_and_boundary_metrics(ref_rgb, pred_rgb, masks, bnd_masks)

    assert metrics["ghosting_luma_mad"] == 0.0
    assert metrics["boundary_luma_mad"] == 0.0
    assert metrics["boundary_ssim"] == 1.0


def test_compute_ghost_and_boundary_metrics_with_distortion() -> None:
    """Distortion on ghost and boundary regions should reflect in positive MAD."""
    t, h, w = 3, 64, 64
    ref_rgb = np.full((t, h, w, 3), 100, dtype=np.uint8)
    pred_rgb = np.full((t, h, w, 3), 110, dtype=np.uint8)  # +10 luma delta

    masks = np.zeros((t, h, w), dtype=bool)
    masks[0, 10:30, 10:30] = True

    bnd_masks = np.zeros((t, h, w), dtype=bool)
    bnd_masks[:, 5:10, 5:10] = True

    metrics = compute_ghost_and_boundary_metrics(ref_rgb, pred_rgb, masks, bnd_masks)

    # Luma of (110, 110, 110) - (100, 100, 100) is 10.0
    assert abs(metrics["ghosting_luma_mad"] - 10.0) < 0.2
    assert abs(metrics["boundary_luma_mad"] - 10.0) < 0.2
    assert metrics["boundary_psnr_y_dB"] > 0.0


def test_compute_ghost_and_boundary_metrics_empty_masks() -> None:
    """When masks are empty, defaults to 0.0 without errors."""
    t, h, w = 2, 32, 32
    ref_rgb = np.full((t, h, w, 3), 100, dtype=np.uint8)
    pred_rgb = np.full((t, h, w, 3), 100, dtype=np.uint8)
    empty_masks = np.zeros((t, h, w), dtype=bool)

    metrics = compute_ghost_and_boundary_metrics(ref_rgb, pred_rgb, empty_masks, empty_masks)

    assert metrics["ghosting_luma_mad"] == 0.0
    assert metrics["boundary_luma_mad"] == 0.0
