from __future__ import annotations

import numpy as np

from demo.pipeline.maps.canny_quality import (
    boundary_band,
    extract_tuned_frame,
    score_edges,
)


def test_box_canny_recalls_its_own_boundary() -> None:
    frame = np.zeros((80, 120), dtype=np.uint8)
    frame[20:60, 30:90] = 255
    edges = extract_tuned_frame(frame, lo=50, hi=150, target_height=80)
    band = boundary_band(frame, radius=2)
    scored = score_edges(edges, band, None)
    assert scored["recall"] > 0.9
    assert scored["waste"] < 0.5
    assert scored["edge_px"] > 0


def test_blur_and_higher_thresholds_drop_edges() -> None:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 40, size=(72, 96), dtype=np.uint8)
    frame[15:55, 20:70] = 220
    dense = extract_tuned_frame(frame, lo=50, hi=150, target_height=72, blur_sigma=0.0)
    sparse = extract_tuned_frame(frame, lo=150, hi=250, target_height=72, blur_sigma=1.6)
    assert int(sparse.sum()) < int(dense.sum())


def test_task_gate_drops_edges_off_the_contour() -> None:
    from demo.pipeline.maps.canny_quality import gate_to_task_boundary

    edge = np.zeros((40, 60), dtype=np.uint8)
    edge[10, 15] = 1
    edge[30, 50] = 1
    filled = np.zeros((40, 60), dtype=np.uint8)
    filled[8:22, 12:28] = 1
    strokes = np.zeros((40, 60), dtype=np.uint8)
    gated = gate_to_task_boundary(edge, filled, strokes, radius=2)
    assert gated[10, 15] == 1
    assert gated[30, 50] == 0


def test_resize_first_is_smaller_grid_than_source() -> None:
    frame = np.zeros((90, 160), dtype=np.uint8)
    frame[20:70, 30:120] = 255
    edges = extract_tuned_frame(frame, lo=50, hi=150, target_height=45)
    assert edges.shape[0] == 45
    assert edges.shape[1] == 80
