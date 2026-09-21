"""Tests for visual inspection and comparison montage generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.components.metrics.visual_inspection import (
    compute_error_heatmap,
    create_comparison_strip,
    draw_mask_contour,
    draw_skeleton,
    generate_carousel_markdown,
    save_montage_image,
)


def test_draw_skeleton_adds_limbs_and_joints() -> None:
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    kpts = np.array([[50.0, 50.0], [50.0, 70.0], [70.0, 70.0]], dtype=np.float32)

    drawn = draw_skeleton(canvas, kpts, color=(0, 255, 0))

    assert drawn.shape == canvas.shape
    # Non-zero pixels must exist where skeleton was drawn
    assert np.any(drawn > 0)


def test_draw_mask_contour_adds_boundary() -> None:
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True

    drawn = draw_mask_contour(canvas, mask, color=(0, 255, 255))

    assert drawn.shape == canvas.shape
    assert np.any(drawn > 0)


def test_compute_error_heatmap_highlights_differences() -> None:
    ref = np.full((50, 50, 3), 100, dtype=np.uint8)
    pred = ref.copy()
    pred[20:30, 20:30] = 150  # error region

    heatmap = compute_error_heatmap(ref, pred, amplify=5.0)

    assert heatmap.shape == ref.shape
    # Error region should have distinct color from non-error background
    assert not np.array_equal(heatmap[25, 25], heatmap[0, 0])


def test_create_comparison_strip_layout_and_footer() -> None:
    ref = np.full((64, 64, 3), 120, dtype=np.uint8)
    pred = np.full((64, 64, 3), 130, dtype=np.uint8)
    cond = np.zeros((64, 64, 3), dtype=np.uint8)

    summary = "PSNR: 32.4 dB | OKS: 0.94 | Mask IoU: 0.88"
    strip = create_comparison_strip(
        reference=ref,
        predicted=pred,
        conditioning=cond,
        metrics_summary=summary,
    )

    # 4 panels horizontally (64 * 4 = 256 width)
    # Height = panel_h (64) + banner_h (30) + footer_h (28) = 122
    assert strip.shape[1] == 256
    assert strip.shape[0] == 122
    assert strip.ndim == 3


def test_save_montage_image(tmp_path: Path) -> None:
    strip = np.zeros((64, 128, 3), dtype=np.uint8)
    out_file = tmp_path / "test_montage.png"

    saved = save_montage_image(strip, out_file)
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_generate_carousel_markdown() -> None:
    paths = [Path("/tmp/slide1.png"), Path("/tmp/slide2.png")]
    titles = ["Frame 001", "Frame 002"]

    md = generate_carousel_markdown(paths, titles)
    assert "````carousel" in md
    assert "<!-- slide -->" in md
    assert "![Frame 001]" in md
    assert "![Frame 002]" in md

