from __future__ import annotations

import numpy as np
import pytest

from src.decoder import animate_anyone_runtime as runtime


def test_normalize_pose_to_canvas_brings_large_coordinates_in_bounds() -> None:
    pose = np.zeros((18, 3), dtype=np.float32)
    pose[:, 0] = np.linspace(900.0, 1500.0, 18)
    pose[:, 1] = np.linspace(1200.0, 2200.0, 18)
    pose[:, 2] = 0.95

    normalized = runtime._normalize_pose_to_canvas(pose, width=512, height=784)

    assert normalized.shape == (18, 3)
    assert float(np.min(normalized[:, 0])) >= 0.0
    assert float(np.max(normalized[:, 0])) <= 511.0
    assert float(np.min(normalized[:, 1])) >= 0.0
    assert float(np.max(normalized[:, 1])) <= 783.0
    assert float(np.max(normalized[:, 0]) - np.min(normalized[:, 0])) > 50.0
    assert float(np.max(normalized[:, 1]) - np.min(normalized[:, 1])) > 120.0


def test_prepare_pose_sequence_produces_visible_condition_from_global_coords() -> None:
    pil = pytest.importorskip("PIL.Image")
    _ = pil

    pose_seq = np.zeros((2, 18, 3), dtype=np.float32)
    for idx in range(2):
        pose_seq[idx, :, 0] = np.linspace(800.0 + idx * 30.0, 1300.0 + idx * 30.0, 18)
        pose_seq[idx, :, 1] = np.linspace(1000.0 + idx * 20.0, 2000.0 + idx * 20.0, 18)
        pose_seq[idx, :, 2] = 0.9

    images = runtime._prepare_pose_sequence(pose_seq, width=512, height=784)
    assert len(images) == 2

    frame0 = np.asarray(images[0], dtype=np.uint8)
    frame1 = np.asarray(images[1], dtype=np.uint8)
    assert frame0.shape == (784, 512, 3)
    assert frame1.shape == (784, 512, 3)
    assert int(np.max(frame0)) > 0
    assert int(np.max(frame1)) > 0


def test_letterbox_resize_preserves_content_visibility() -> None:
    src = np.zeros((120, 60, 3), dtype=np.uint8)
    src[:, :, 1] = 200

    out = runtime._letterbox_resize_rgb(src, target_w=512, target_h=784)

    assert out.shape == (784, 512, 3)
    assert int(np.max(out[:, :, 1])) == 200
    assert int(np.count_nonzero(out[:, :, 1])) > 0
