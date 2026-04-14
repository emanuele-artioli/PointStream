from __future__ import annotations

import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest
import torch

import src.decoder.genai_compositor as gc


def _install_fake_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fromarray(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.uint8)

    image_module = types.ModuleType("PIL.Image")
    setattr(image_module, "fromarray", _fromarray)

    pil_module = types.ModuleType("PIL")
    setattr(pil_module, "Image", image_module)

    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)


def _make_pose_sequence(valid_confidence: float = 0.95) -> torch.Tensor:
    pose = torch.zeros((1, 18, 3), dtype=torch.float32)
    pose[0, :, 0] = torch.linspace(70.0, 120.0, 18)
    pose[0, :, 1] = torch.linspace(30.0, 150.0, 18)
    pose[0, :, 2] = valid_confidence
    return pose


def test_mock_compositor_process_changes_frame() -> None:
    compositor = gc.MockCompositor()

    reference = torch.full((3, 96, 64), 200, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )

    assert tuple(out.shape) == (3, 180, 320)
    assert out.dtype == torch.uint8
    assert int(torch.count_nonzero(out)) > 0


def test_mock_compositor_handles_no_visible_pose() -> None:
    compositor = gc.MockCompositor()

    reference = torch.full((3, 80, 48), 120, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.0)
    background = torch.zeros((3, 120, 200), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )
    assert int(torch.count_nonzero(out)) > 0


class _DummyStrategy(gc.BaseGenAIStrategy):
    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        _ = dense_dwpose_tensor
        _ = seed
        _ = device
        h = int(reference_crop_tensor.shape[1])
        w = int(reference_crop_tensor.shape[2])
        actor = np.zeros((h, w, 3), dtype=np.uint8)
        actor[:, :, 1] = 220
        actor[:, :, 2] = 80
        return torch.from_numpy(actor).permute(2, 0, 1).contiguous().to(torch.uint8)


def test_diffusers_compositor_process_with_stub_strategy(monkeypatch) -> None:
    monkeypatch.setattr(gc.DiffusersCompositor, "_build_strategy", lambda self, backend: _DummyStrategy())

    compositor = gc.DiffusersCompositor(backend="controlnet", seed=1234)
    reference = torch.full((3, 64, 48), 180, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    background = torch.zeros((3, 180, 320), dtype=torch.uint8)

    out = compositor.process(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        warped_background_frame=background,
    )

    assert tuple(out.shape) == (3, 180, 320)
    assert int(torch.count_nonzero(out)) > 0


def test_diffusers_compositor_invalid_backend_raises() -> None:
    with pytest.raises(ValueError):
        gc.DiffusersCompositor(backend="unknown-backend")


def test_baseline_controlnet_strategy_generate_with_fake_pipe(monkeypatch) -> None:
    _install_fake_pillow(monkeypatch)
    pil_image = sys.modules["PIL.Image"]

    strategy = gc.BaselineControlNetStrategy()

    class _FakePipe:
        def __call__(self, **kwargs):
            _ = kwargs
            fake_rgb = np.full((64, 48, 3), 140, dtype=np.uint8)
            return SimpleNamespace(images=[pil_image.fromarray(fake_rgb)])

    monkeypatch.setattr(strategy, "_ensure_pipeline", lambda device: _FakePipe())

    reference = torch.full((3, 64, 48), 90, dtype=torch.uint8)
    pose = _make_pose_sequence(valid_confidence=0.9)
    out = strategy.generate(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=pose,
        seed=2026,
        device=torch.device("cpu"),
    )

    assert tuple(out.shape) == (3, 64, 48)
    assert out.dtype == torch.uint8


def test_animate_anyone_strategy_missing_repo_raises(monkeypatch) -> None:
    monkeypatch.delenv("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR", raising=False)
    strategy = gc.AnimateAnyoneStrategy(repo_dir=None)

    with pytest.raises(FileNotFoundError):
        strategy.generate(
            reference_crop_tensor=torch.zeros((3, 32, 32), dtype=torch.uint8),
            dense_dwpose_tensor=_make_pose_sequence(valid_confidence=0.9),
            seed=1,
            device=torch.device("cpu"),
        )


def test_animate_anyone_strategy_generate_with_stub_runtime(monkeypatch) -> None:
    strategy = gc.AnimateAnyoneStrategy(repo_dir="/tmp/does-not-matter")

    def _runtime_fn(reference_image_bgr, dense_pose_sequence, seed, device):
        _ = dense_pose_sequence
        _ = seed
        _ = device
        return np.asarray(reference_image_bgr, dtype=np.uint8)

    monkeypatch.setattr(strategy, "_ensure_runtime", lambda: _runtime_fn)

    reference = torch.full((3, 40, 30), 77, dtype=torch.uint8)
    out = strategy.generate(
        reference_crop_tensor=reference,
        dense_dwpose_tensor=_make_pose_sequence(valid_confidence=0.9),
        seed=44,
        device=torch.device("cpu"),
    )

    assert tuple(out.shape) == (3, 40, 30)


def test_numpy_and_pose_helpers() -> None:
    float_img = torch.rand((3, 40, 30), dtype=torch.float32) * 255.0
    np_img = gc._to_numpy_bgr(float_img)
    assert np_img.dtype == np.uint8
    assert np_img.shape == (40, 30, 3)

    pose = _make_pose_sequence(valid_confidence=0.8)[0]
    condition = gc._render_pose_condition(pose, output_height=64, output_width=64)
    assert condition.shape == (64, 64, 3)
    assert int(np.max(condition)) > 0

    with pytest.raises(ValueError):
        gc._render_pose_condition(torch.zeros((17, 3), dtype=torch.float32), output_height=64, output_width=64)

    with pytest.raises(ValueError):
        gc._to_numpy_bgr(torch.zeros((2, 40, 30), dtype=torch.uint8))
