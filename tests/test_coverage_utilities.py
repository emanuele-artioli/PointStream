from __future__ import annotations

from pathlib import Path
import sys
import types
import numpy as np
from src.shared.config import PointstreamConfig
import pytest
import torch

from src.shared import dwpose_draw
from src.shared import track_id
from src.shared import torch_dtype as td
from src.transport.panorama_encoder import (
    JpegPanoramaEncoder,
    PngPanoramaEncoder,
    build_panorama_encoder,
)


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


def test_track_id_and_dtype_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert track_id.scene_track_id_to_int("person_17") == 17
    assert track_id.scene_track_id_to_int("person_alpha") == track_id.scene_track_id_to_int("person_alpha")

    assert td.parse_gpu_dtype("fp16") == torch.float16


def test_panorama_encoder_build_and_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    jpeg_encoder = build_panorama_encoder("jpeg")
    assert isinstance(jpeg_encoder, JpegPanoramaEncoder)
    jpeg_path = jpeg_encoder.encode(image, tmp_path / "pano_jpeg")
    assert jpeg_path.suffix == ".jpg"
    assert jpeg_path.exists()

    png_encoder = build_panorama_encoder("png")
    assert isinstance(png_encoder, PngPanoramaEncoder)
    png_path = png_encoder.encode(image, tmp_path / "pano_png")
    assert png_path.suffix == ".png"
    assert png_path.exists()

    with pytest.raises(ValueError, match=r"expected \[H, W, 3\]"):
        jpeg_encoder.encode(np.zeros((12, 16), dtype=np.uint8), tmp_path / "bad")

    try:
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            build_panorama_encoder("jpeg", config=PointstreamConfig(panorama_jpeg_quality="bad"))  # type: ignore[arg-type]
    except ImportError:
        pass

    with pytest.raises(ValueError, match="Unsupported panorama encoder"):
        build_panorama_encoder("webp")
