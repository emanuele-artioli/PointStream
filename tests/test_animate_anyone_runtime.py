from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


def test_runtime_config_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_WIDTH", "640")
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_HEIGHT", "360")
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_STEPS", "22")
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_CFG", "4.2")
    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_MODEL_VARIANT", "original")

    cfg = runtime._runtime_config()
    assert cfg.width == 640
    assert cfg.height == 360
    assert cfg.inference_steps == 22
    assert abs(cfg.guidance_scale - 4.2) < 1e-6
    assert cfg.model_variant == "original"


def test_resolve_repo_root_and_model_root(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "moore_aa"
    repo_root.mkdir(parents=True)

    monkeypatch.setenv("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR", str(repo_root))
    resolved_repo = runtime._resolve_repo_root(None)
    assert resolved_repo == repo_root.resolve()

    model_root = repo_root / "Models" / "finetuned_tennis"
    model_root.mkdir(parents=True)
    required = [
        "stable-diffusion-v1-5",
        "sd-vae-ft-mse",
        "image_encoder",
        "denoising_unet.pth",
        "reference_unet.pth",
        "pose_guider.pth",
        "motion_module.pth",
    ]
    for name in required:
        target = model_root / name
        if name.endswith(".pth"):
            target.write_bytes(b"x")
        else:
            target.mkdir(parents=True, exist_ok=True)

    cfg = runtime._RuntimeConfig(model_variant="tennis")
    resolved_model = runtime._resolve_model_root(repo_root=resolved_repo, runtime=cfg)
    assert resolved_model == model_root.resolve()


def test_resolve_model_root_missing_entries_raises(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    model_root = repo_root / "Models" / "finetuned_tennis"
    model_root.mkdir(parents=True)

    cfg = runtime._RuntimeConfig(model_variant="finetuned_tennis")
    with pytest.raises(FileNotFoundError):
        runtime._resolve_model_root(repo_root=repo_root, runtime=cfg)


def test_generate_frame_smoke_with_stub_pipeline(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    model_root = tmp_path / "models"
    model_root.mkdir(parents=True)

    monkeypatch.setattr(runtime, "_runtime_config", lambda: runtime._RuntimeConfig(width=64, height=96, inference_steps=5, guidance_scale=2.0, model_variant="finetuned_tennis"))
    monkeypatch.setattr(runtime, "_resolve_repo_root", lambda repo_dir: repo_root)
    monkeypatch.setattr(runtime, "_resolve_model_root", lambda repo_root, runtime: model_root)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _StubPipe:
        def __call__(self, *args, **kwargs):
            _ = args
            _ = kwargs
            # Shape: [Batch, Channels, Frames, Height, Width]
            videos = torch.linspace(-1.0, 1.0, steps=1 * 3 * 2 * 96 * 64, dtype=torch.float32).reshape(1, 3, 2, 96, 64)
            return SimpleNamespace(videos=videos)

    monkeypatch.setattr(runtime, "_load_pipeline", lambda repo_root, model_root, device: _StubPipe())

    reference = np.zeros((80, 40, 3), dtype=np.uint8)
    reference[:, :, 2] = 220
    pose_seq = np.zeros((2, 18, 3), dtype=np.float32)
    pose_seq[:, :, 0] = np.linspace(30.0, 55.0, 18)
    pose_seq[:, :, 1] = np.linspace(15.0, 70.0, 18)
    pose_seq[:, :, 2] = 0.95

    out = runtime.generate_frame(
        reference_image_bgr=reference,
        dense_pose_sequence=pose_seq,
        seed=123,
        device="cuda",
    )

    assert out.shape == (96, 64, 3)
    assert out.dtype == np.uint8
    assert int(np.max(out)) > 0


def test_generate_frame_fallback_pose_sequence(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    model_root = tmp_path / "models"
    model_root.mkdir(parents=True)

    monkeypatch.setattr(runtime, "_runtime_config", lambda: runtime._RuntimeConfig(width=32, height=32, inference_steps=3, guidance_scale=1.0, model_variant="finetuned_tennis"))
    monkeypatch.setattr(runtime, "_resolve_repo_root", lambda repo_dir: repo_root)
    monkeypatch.setattr(runtime, "_resolve_model_root", lambda repo_root, runtime: model_root)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "_prepare_pose_sequence", lambda dense_pose_sequence, width, height: [])

    class _StubPipe:
        def __call__(self, *args, **kwargs):
            _ = args
            _ = kwargs
            videos = torch.zeros((1, 3, 1, 32, 32), dtype=torch.float32)
            return SimpleNamespace(videos=videos)

    monkeypatch.setattr(runtime, "_load_pipeline", lambda repo_root, model_root, device: _StubPipe())

    reference = np.zeros((16, 16, 3), dtype=np.uint8)
    pose_seq = np.zeros((1, 18, 3), dtype=np.float32)
    out = runtime.generate_frame(
        reference_image_bgr=reference,
        dense_pose_sequence=pose_seq,
        seed=321,
        device="cpu",
    )
    assert out.shape == (32, 32, 3)


def test_generate_frame_validates_reference_shape() -> None:
    bad_reference = np.zeros((40, 40), dtype=np.uint8)
    pose_seq = np.zeros((1, 18, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        runtime.generate_frame(bad_reference, pose_seq, seed=1, device="cpu")
