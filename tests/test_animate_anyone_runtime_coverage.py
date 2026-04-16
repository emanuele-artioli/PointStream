from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import types
from typing import Any

import numpy as np
import pytest
import torch

from src.decoder import animate_anyone_runtime as runtime


@pytest.fixture(autouse=True)
def _reset_runtime_pipeline_cache() -> None:
    runtime._PIPELINE = None
    runtime._PIPELINE_DEVICE = None
    runtime._PIPELINE_DTYPE = None
    runtime._PIPELINE_REPO_ROOT = None
    runtime._PIPELINE_MODEL_ROOT = None


def _install_fake_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fromarray(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.uint8)

    image_module: Any = types.ModuleType("PIL.Image")
    setattr(image_module, "fromarray", _fromarray)

    pil_module: Any = types.ModuleType("PIL")
    setattr(pil_module, "Image", image_module)

    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)


def _install_fake_animate_anyone_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ToDevice:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _ = (args, kwargs)
            return cls()

        @classmethod
        def from_pretrained_2d(cls, *args, **kwargs):
            _ = (args, kwargs)
            return cls()

        def to(self, device=None, dtype=None):
            self.device = device
            self.dtype = dtype
            return self

        def load_state_dict(self, state, strict=False):
            _ = strict
            self.state = state

    class _AutoencoderKL(_ToDevice):
        pass

    class _UNet2DConditionModel(_ToDevice):
        pass

    class _UNet3DConditionModel(_ToDevice):
        pass

    class _CLIPVisionModelWithProjection(_ToDevice):
        pass

    class _PoseGuider(_ToDevice):
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    class _DDIMScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Pose2VideoPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device=None, dtype=None):
            self.device = device
            self.dtype = dtype
            return self

    class _InferConfig:
        def __init__(self) -> None:
            self.unet_additional_kwargs = {"foo": 1}
            self.noise_scheduler_kwargs = {"bar": 2}

    class _OmegaConf:
        @staticmethod
        def load(path: str):
            _ = path
            return _InferConfig()

        @staticmethod
        def to_container(value):
            return value

    diffusers: Any = types.ModuleType("diffusers")
    setattr(diffusers, "AutoencoderKL", _AutoencoderKL)
    setattr(diffusers, "DDIMScheduler", _DDIMScheduler)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    transformers: Any = types.ModuleType("transformers")
    setattr(transformers, "CLIPVisionModelWithProjection", _CLIPVisionModelWithProjection)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    omegaconf: Any = types.ModuleType("omegaconf")
    setattr(omegaconf, "OmegaConf", _OmegaConf)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    pose_guider_mod: Any = types.ModuleType("src.models.pose_guider")
    setattr(pose_guider_mod, "PoseGuider", _PoseGuider)
    monkeypatch.setitem(sys.modules, "src.models.pose_guider", pose_guider_mod)

    unet2d_mod: Any = types.ModuleType("src.models.unet_2d_condition")
    setattr(unet2d_mod, "UNet2DConditionModel", _UNet2DConditionModel)
    monkeypatch.setitem(sys.modules, "src.models.unet_2d_condition", unet2d_mod)

    unet3d_mod: Any = types.ModuleType("src.models.unet_3d")
    setattr(unet3d_mod, "UNet3DConditionModel", _UNet3DConditionModel)
    monkeypatch.setitem(sys.modules, "src.models.unet_3d", unet3d_mod)

    pipeline_mod: Any = types.ModuleType("src.pipelines.pipeline_pose2vid_long")
    setattr(pipeline_mod, "Pose2VideoPipeline", _Pose2VideoPipeline)
    monkeypatch.setitem(sys.modules, "src.pipelines.pipeline_pose2vid_long", pipeline_mod)


def _create_required_model_entries(model_root: Path) -> None:
    required = [
        "stable-diffusion-v1-5",
        "sd-vae-ft-mse",
        "image_encoder",
        "denoising_unet.pth",
        "reference_unet.pth",
        "pose_guider.pth",
        "motion_module.pth",
    ]
    model_root.mkdir(parents=True, exist_ok=True)
    for entry in required:
        path = model_root / entry
        if entry.endswith(".pth"):
            path.write_bytes(b"x")
        else:
            path.mkdir(parents=True, exist_ok=True)


def test_resolve_repo_root_missing_and_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="not set"):
        runtime._resolve_repo_root(None)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime._resolve_repo_root(str(tmp_path / "missing_repo"))


def test_resolve_model_root_explicit_relative_and_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)

    relative_model = repo_root / "custom_models"
    _create_required_model_entries(relative_model)

    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR", "custom_models")
    resolved_relative = runtime._resolve_model_root(repo_root=repo_root, runtime=runtime._RuntimeConfig())
    assert resolved_relative == relative_model.resolve()

    absolute_model = tmp_path / "absolute_models"
    _create_required_model_entries(absolute_model)
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR", str(absolute_model))
    resolved_absolute = runtime._resolve_model_root(repo_root=repo_root, runtime=runtime._RuntimeConfig())
    assert resolved_absolute == absolute_model.resolve()


def test_render_and_pose_preparation_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pillow(monkeypatch)

    zeros = runtime._render_pose_image_from_dwpose(np.zeros((17, 3), dtype=np.float32), width=32, height=24)
    assert zeros.shape == (24, 32, 3)
    assert int(np.max(zeros)) == 0

    invalid_shape = np.zeros((10, 2), dtype=np.float32)
    assert runtime._normalize_pose_to_canvas(invalid_shape, width=32, height=24).shape == (10, 2)

    low_conf_pose = np.zeros((18, 3), dtype=np.float32)
    low_conf_pose[:, 2] = 0.1
    low_conf_out = runtime._normalize_pose_to_canvas(low_conf_pose, width=32, height=24)
    assert np.array_equal(low_conf_out, low_conf_pose)

    two_d_pose = np.zeros((18, 3), dtype=np.float32)
    two_d_pose[:, 2] = 0.9
    images = runtime._prepare_pose_sequence(two_d_pose, width=32, height=24)
    assert len(images) == 1

    with pytest.raises(ValueError, match="Expected dense pose sequence"):
        runtime._prepare_pose_sequence(np.zeros((1, 2, 3, 4), dtype=np.float32), width=32, height=24)


def test_letterbox_resize_handles_empty_input() -> None:
    out = runtime._letterbox_resize_rgb(np.zeros((0, 0, 3), dtype=np.uint8), target_w=32, target_h=24)
    assert out.shape == (24, 32, 3)
    assert int(np.max(out)) == 0


def test_load_pipeline_success_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_animate_anyone_modules(monkeypatch)

    monkeypatch.setattr(runtime, "resolve_torch_dtype_for_device", lambda *args, **kwargs: torch.float32)
    monkeypatch.setattr(runtime.torch, "load", lambda path, map_location=None: {})

    repo_root = tmp_path / "repo"
    model_root = tmp_path / "model"
    repo_root.mkdir(parents=True)
    model_root.mkdir(parents=True)

    first = runtime._load_pipeline(repo_root=repo_root, model_root=model_root, device="cpu")
    second = runtime._load_pipeline(repo_root=repo_root, model_root=model_root, device="cpu")

    assert first is second
    assert runtime._PIPELINE_DEVICE == "cpu"
    assert runtime._PIPELINE_DTYPE == torch.float32
    assert runtime._PIPELINE_REPO_ROOT == str(repo_root)
    assert runtime._PIPELINE_MODEL_ROOT == str(model_root)


def test_load_pipeline_raises_when_scheduler_config_is_not_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_animate_anyone_modules(monkeypatch)

    class _BadOmegaConf:
        @staticmethod
        def load(path: str):
            _ = path
            return SimpleNamespace(unet_additional_kwargs={"foo": 1}, noise_scheduler_kwargs=[1, 2, 3])

        @staticmethod
        def to_container(value):
            return value

    omegaconf: Any = types.ModuleType("omegaconf")
    setattr(omegaconf, "OmegaConf", _BadOmegaConf)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    monkeypatch.setattr(runtime, "resolve_torch_dtype_for_device", lambda *args, **kwargs: torch.float32)
    monkeypatch.setattr(runtime.torch, "load", lambda path, map_location=None: {})

    with pytest.raises(ValueError, match="must be a mapping"):
        runtime._load_pipeline(repo_root=tmp_path / "repo", model_root=tmp_path / "model", device="cpu")


def test_generate_frame_uses_cuda_when_available_and_usable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_pillow(monkeypatch)

    repo_root = tmp_path / "repo"
    model_root = tmp_path / "models"
    repo_root.mkdir(parents=True)
    model_root.mkdir(parents=True)

    monkeypatch.setattr(runtime, "_runtime_config", lambda: runtime._RuntimeConfig(width=16, height=16, inference_steps=2, guidance_scale=1.0, model_variant="finetuned_tennis"))
    monkeypatch.setattr(runtime, "_resolve_repo_root", lambda repo_dir: repo_root)
    monkeypatch.setattr(runtime, "_resolve_model_root", lambda repo_root, runtime: model_root)
    monkeypatch.setattr(runtime, "_prepare_pose_sequence", lambda dense_pose_sequence, width, height: [])

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "is_cuda_device_usable", lambda device: True)

    class _FakeGenerator:
        def __init__(self, device=None) -> None:
            self.device = device

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    monkeypatch.setattr(runtime.torch, "Generator", _FakeGenerator)

    captured: dict[str, str] = {}

    class _StubPipe:
        def __call__(self, *args, **kwargs):
            _ = (args, kwargs)
            videos = torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32)
            return SimpleNamespace(videos=videos)

    def _load_pipeline(repo_root: Path, model_root: Path, device: str):
        _ = (repo_root, model_root)
        captured["device"] = device
        return _StubPipe()

    monkeypatch.setattr(runtime, "_load_pipeline", _load_pipeline)

    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    pose_seq = np.zeros((1, 18, 3), dtype=np.float32)

    out = runtime.generate_frame(reference, pose_seq, seed=7, device="cuda")
    assert captured["device"] == "cuda"
    assert out.shape == (16, 16, 3)
