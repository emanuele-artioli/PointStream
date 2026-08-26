"""Skeleton drawing for generators that put a pose on a canvas.

Ported from tests/test_coverage_utilities.py (BP22). Not tested here: the
third-party ``dwpose.draw_poses`` internals; the fallback path is what we own.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.components.generation import dwpose_draw


def test_dw18_to_pose_results_and_canvas_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    types_mod = types.ModuleType("dwpose.types")

    class _Keypoint:
        def __init__(self, x, y, conf, idx):
            self.x = x
            self.y = y
            self.conf = conf
            self.idx = idx

    class _BodyResult:
        def __init__(self, keypoints):
            self.keypoints = keypoints

    class _PoseResult:
        def __init__(self, body, left_hand=None, right_hand=None, face=None):
            self.body = body
            self.left_hand = left_hand
            self.right_hand = right_hand
            self.face = face

    setattr(types_mod, "Keypoint", _Keypoint)
    setattr(types_mod, "BodyResult", _BodyResult)
    setattr(types_mod, "PoseResult", _PoseResult)
    monkeypatch.setitem(sys.modules, "dwpose.types", types_mod)

    pose = np.zeros((1, 18, 3), dtype=np.float32)
    pose[0, :, 0] = np.linspace(0.1, 0.8, 18)
    pose[0, :, 1] = np.linspace(0.2, 0.9, 18)
    pose[0, :, 2] = 0.9
    pose[0, 0, 2] = 0.05

    results = dwpose_draw.dw18_to_pose_results(pose, confidence_threshold=0.2)
    assert len(results) == 1
    assert results[0].body.keypoints[0] is None
    assert results[0].body.keypoints[1] is not None

    fake_dwpose = types.ModuleType("dwpose")

    def _draw_poses(pose_results, height, width, draw_body=True, draw_hand=False, draw_face=False):
        _ = (pose_results, draw_body, draw_hand, draw_face)
        return np.full((height, width, 3), 7, dtype=np.uint8)

    setattr(fake_dwpose, "draw_poses", _draw_poses)
    monkeypatch.setitem(sys.modules, "dwpose", fake_dwpose)

    canvas = dwpose_draw.draw_dwpose_canvas(height=32, width=24, people_dw=pose, confidence_threshold=0.2)
    assert canvas.shape == (32, 24, 3)
    assert int(canvas.max()) == 7


def test_dwpose_canvas_falls_back_when_renderer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dwpose = types.ModuleType("dwpose")
    fake_types = types.ModuleType("dwpose.types")

    class _Keypoint:
        def __init__(self, x, y, conf, idx):
            self.x = x
            self.y = y
            self.conf = conf
            self.idx = idx

    class _BodyResult:
        def __init__(self, keypoints):
            self.keypoints = keypoints

    class _PoseResult:
        def __init__(self, body, left_hand=None, right_hand=None, face=None):
            self.body = body
            self.left_hand = left_hand
            self.right_hand = right_hand
            self.face = face

    setattr(fake_types, "Keypoint", _Keypoint)
    setattr(fake_types, "BodyResult", _BodyResult)
    setattr(fake_types, "PoseResult", _PoseResult)
    monkeypatch.setitem(sys.modules, "dwpose.types", fake_types)

    def _draw_poses(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("boom")

    setattr(fake_dwpose, "draw_poses", _draw_poses)
    monkeypatch.setitem(sys.modules, "dwpose", fake_dwpose)

    pose = np.zeros((1, 18, 3), dtype=np.float32)
    pose[0, :, 0] = np.linspace(0.1, 0.8, 18)
    pose[0, :, 1] = np.linspace(0.2, 0.9, 18)
    pose[0, :, 2] = 0.9

    canvas = dwpose_draw.draw_dwpose_canvas(height=32, width=24, people_dw=pose, confidence_threshold=0.2)
    assert canvas.shape == (32, 24, 3)
    assert int(canvas.max()) > 0
