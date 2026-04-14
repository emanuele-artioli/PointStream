from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Any
from typing import cast

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class _RuntimeConfig:
    width: int = 512
    height: int = 784
    inference_steps: int = 30
    guidance_scale: float = 3.5
    model_variant: str = "finetuned_tennis"


_PIPELINE: Any | None = None
_PIPELINE_DEVICE: str | None = None
_PIPELINE_DTYPE: torch.dtype | None = None
_PIPELINE_REPO_ROOT: str | None = None
_PIPELINE_MODEL_ROOT: str | None = None
_PIPELINE_LOCK = threading.Lock()


def _runtime_config() -> _RuntimeConfig:
    return _RuntimeConfig(
        width=int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_WIDTH", "512")),
        height=int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_HEIGHT", "512")),
        inference_steps=int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_STEPS", "30")),
        guidance_scale=float(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_CFG", "3.5")),
        model_variant=os.environ.get("POINTSTREAM_ANIMATE_ANYONE_MODEL_VARIANT", "finetuned_tennis"),
    )


def _resolve_repo_root(repo_dir: str | None) -> Path:
    configured = repo_dir or os.environ.get("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR")
    if configured is None:
        raise FileNotFoundError(
            "Animate Anyone backend selected, but POINTSTREAM_ANIMATE_ANYONE_REPO_DIR is not set."
        )

    repo_root = Path(configured).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"Animate Anyone repository path does not exist: {repo_root}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _resolve_model_root(repo_root: Path, runtime: _RuntimeConfig) -> Path:
    explicit_model_dir = os.environ.get("POINTSTREAM_ANIMATE_ANYONE_MODEL_DIR")
    if explicit_model_dir:
        model_root = Path(explicit_model_dir).expanduser()
        if not model_root.is_absolute():
            model_root = (repo_root / model_root).resolve()
        else:
            model_root = model_root.resolve()
    else:
        variant = runtime.model_variant.strip().lower().replace("-", "_")
        alias_map = {
            "original": "original",
            "base": "original",
            "pretrained": "original",
            "finetuned_tennis": "finetuned_tennis",
            "tennis": "finetuned_tennis",
            "finetuned": "finetuned_tennis",
        }
        model_folder = alias_map.get(variant, runtime.model_variant)
        model_root = (repo_root / "Models" / model_folder).resolve()

    required_entries = [
        "stable-diffusion-v1-5",
        "sd-vae-ft-mse",
        "image_encoder",
        "denoising_unet.pth",
        "reference_unet.pth",
        "pose_guider.pth",
        "motion_module.pth",
    ]
    missing = [name for name in required_entries if not (model_root / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"AnimateAnyone model directory is missing required entries: {missing_list}. "
            f"Directory checked: {model_root}"
        )

    return model_root


def _load_pipeline(repo_root: Path, model_root: Path, device: str) -> Any:
    global _PIPELINE, _PIPELINE_DEVICE, _PIPELINE_DTYPE, _PIPELINE_REPO_ROOT, _PIPELINE_MODEL_ROOT

    with _PIPELINE_LOCK:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        if (
            _PIPELINE is not None
            and _PIPELINE_DEVICE == device
            and _PIPELINE_DTYPE == dtype
            and _PIPELINE_REPO_ROOT == str(repo_root)
            and _PIPELINE_MODEL_ROOT == str(model_root)
        ):
            return _PIPELINE

        try:
            from diffusers import AutoencoderKL, DDIMScheduler
            from omegaconf import OmegaConf
            from transformers import CLIPVisionModelWithProjection

            from src.models.pose_guider import PoseGuider
            from src.models.unet_2d_condition import UNet2DConditionModel
            from src.models.unet_3d import UNet3DConditionModel
            from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AnimateAnyone backend dependencies are missing. "
                "Install the Moore-AnimateAnyone runtime dependencies in the active environment."
            ) from exc

        infer_config = OmegaConf.load(str(repo_root / "configs" / "inference" / "inference_v2.yaml"))

        pretrained_vae_path = str(model_root / "sd-vae-ft-mse")
        pretrained_base_model_path = str(model_root / "stable-diffusion-v1-5")
        image_encoder_path = str(model_root / "image_encoder")
        denoising_unet_path = str(model_root / "denoising_unet.pth")
        reference_unet_path = str(model_root / "reference_unet.pth")
        pose_guider_path = str(model_root / "pose_guider.pth")
        motion_module_path = str(model_root / "motion_module.pth")

        vae = AutoencoderKL.from_pretrained(pretrained_vae_path).to(device=device, dtype=dtype)
        reference_unet = UNet2DConditionModel.from_pretrained(
            pretrained_base_model_path,
            subfolder="unet",
        ).to(device=device, dtype=dtype)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_base_model_path,
            motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(device=device, dtype=dtype)
        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(device=device, dtype=dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device=device, dtype=dtype)

        sched_container = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        if not isinstance(sched_container, dict):
            raise ValueError("AnimateAnyone inference config 'noise_scheduler_kwargs' must be a mapping")
        sched_kwargs = cast(dict[str, Any], sched_container)
        scheduler = DDIMScheduler(**sched_kwargs)

        denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
        reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"), strict=False)
        pose_guider.load_state_dict(torch.load(pose_guider_path, map_location="cpu"), strict=False)

        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to(device=device, dtype=dtype)

        _PIPELINE = pipe
        _PIPELINE_DEVICE = device
        _PIPELINE_DTYPE = dtype
        _PIPELINE_REPO_ROOT = str(repo_root)
        _PIPELINE_MODEL_ROOT = str(model_root)
        return pipe


def _render_pose_image_from_dwpose(pose: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if pose.shape != (18, 3):
        return canvas

    valid = pose[:, 2] >= 0.2
    points = pose[:, :2].astype(np.float32)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (1, 5),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (1, 11),
        (11, 12),
        (12, 13),
    ]

    for idx in np.where(valid)[0]:
        x = int(round(points[idx, 0]))
        y = int(round(points[idx, 1]))
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        ax, ay = int(round(points[a, 0])), int(round(points[a, 1]))
        bx, by = int(round(points[b, 0])), int(round(points[b, 1]))
        cv2.line(canvas, (ax, ay), (bx, by), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    return canvas


def _normalize_pose_to_canvas(
    pose: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float = 0.12,
) -> np.ndarray:
    if pose.shape != (18, 3):
        return pose

    normalized = pose.copy().astype(np.float32, copy=False)
    valid = normalized[:, 2] >= 0.2
    if int(np.count_nonzero(valid)) < 2:
        return normalized

    xs = normalized[valid, 0]
    ys = normalized[valid, 1]

    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    src_w = max(1.0, max_x - min_x)
    src_h = max(1.0, max_y - min_y)
    dst_w = max(1.0, float(width) * (1.0 - 2.0 * margin_ratio))
    dst_h = max(1.0, float(height) * (1.0 - 2.0 * margin_ratio))
    scale = min(dst_w / src_w, dst_h / src_h)

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    target_center_x = 0.5 * float(width)
    target_center_y = 0.5 * float(height)

    normalized[:, 0] = (normalized[:, 0] - center_x) * scale + target_center_x
    normalized[:, 1] = (normalized[:, 1] - center_y) * scale + target_center_y
    normalized[:, 0] = np.clip(normalized[:, 0], 0.0, float(width - 1))
    normalized[:, 1] = np.clip(normalized[:, 1], 0.0, float(height - 1))
    return normalized


def _prepare_pose_sequence(dense_pose_sequence: np.ndarray, width: int, height: int) -> list[Any]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for AnimateAnyone backend") from exc

    if dense_pose_sequence.ndim == 2:
        dense_pose_sequence = dense_pose_sequence[np.newaxis, ...]
    if dense_pose_sequence.ndim != 3:
        raise ValueError(f"Expected dense pose sequence [T,18,3] or [18,3], got {dense_pose_sequence.shape}")

    pose_images: list[Any] = []
    for pose in dense_pose_sequence:
        normalized_pose = _normalize_pose_to_canvas(
            pose=np.asarray(pose, dtype=np.float32),
            width=width,
            height=height,
        )
        pose_canvas = _render_pose_image_from_dwpose(pose=normalized_pose, width=width, height=height)
        pose_rgb = cv2.cvtColor(pose_canvas, cv2.COLOR_BGR2RGB)
        pose_images.append(Image.fromarray(pose_rgb))
    return pose_images


def _letterbox_resize_rgb(image_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = image_rgb.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    resized_w = max(1, int(round(float(src_w) * scale)))
    resized_h = max(1, int(round(float(src_h) * scale)))
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = max(0, (target_w - resized_w) // 2)
    off_y = max(0, (target_h - resized_h) // 2)
    canvas[off_y : off_y + resized_h, off_x : off_x + resized_w] = resized
    return canvas


def generate_frame(
    reference_image_bgr: np.ndarray,
    dense_pose_sequence: np.ndarray,
    seed: int,
    device: str = "cuda",
    repo_dir: str | None = None,
) -> np.ndarray:
    """Generate one actor frame via Moore-AnimateAnyone from PointStream-owned runtime glue."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for AnimateAnyone backend") from exc

    if reference_image_bgr.ndim != 3 or reference_image_bgr.shape[2] != 3:
        raise ValueError(f"Expected reference_image_bgr [H,W,3], got {reference_image_bgr.shape}")

    runtime = _runtime_config()
    repo_root = _resolve_repo_root(repo_dir=repo_dir)
    model_root = _resolve_model_root(repo_root=repo_root, runtime=runtime)

    resolved_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    pipe = _load_pipeline(repo_root=repo_root, model_root=model_root, device=resolved_device)

    width = int(runtime.width)
    height = int(runtime.height)

    reference_rgb = cv2.cvtColor(reference_image_bgr, cv2.COLOR_BGR2RGB)
    reference_rgb = _letterbox_resize_rgb(reference_rgb, target_w=width, target_h=height)
    reference_pil = Image.fromarray(reference_rgb)

    dense_pose_sequence = np.asarray(dense_pose_sequence, dtype=np.float32)
    pose_pil_sequence = _prepare_pose_sequence(
        dense_pose_sequence=dense_pose_sequence,
        width=width,
        height=height,
    )
    if not pose_pil_sequence:
        pose_pil_sequence = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))]

    generator = torch.Generator(device=resolved_device).manual_seed(int(seed))
    with torch.no_grad():
        output = pipe(
            reference_pil,
            pose_pil_sequence,
            width,
            height,
            len(pose_pil_sequence),
            int(runtime.inference_steps),
            float(runtime.guidance_scale),
            generator=generator,
        ).videos

    generated = output[0, :, -1].detach().float()
    if float(torch.min(generated)) < 0.0:
        generated = (generated + 1.0) * 0.5
    generated = generated.clamp(0.0, 1.0)

    generated_rgb = (generated.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    generated_bgr = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
    return generated_bgr