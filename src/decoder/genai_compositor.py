from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.shared.tags import gpu_bound


class BaseGenAIStrategy(ABC):
    """Strategy interface for pluggable GenAI actor generation backends."""

    @abstractmethod
    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError


class BaselineControlNetStrategy(BaseGenAIStrategy):
    """Standard SD + ControlNet OpenPose baseline backend."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_openpose",
    ) -> None:
        self._model_id = model_id
        self._controlnet_id = controlnet_id
        self._pipe: Any | None = None

    def _ensure_pipeline(self, device: torch.device) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Diffusers backend requested but dependencies are missing. "
                "Install diffusers, transformers, and accelerate."
            ) from exc

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(self._controlnet_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self._model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        pipe = self._ensure_pipeline(device)
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pillow is required for ControlNet strategy") from exc

        reference_np = _to_numpy_bgr(reference_crop_tensor)
        target_h, target_w = int(reference_np.shape[0]), int(reference_np.shape[1])

        pose_tensor = dense_dwpose_tensor
        if pose_tensor.ndim == 3:
            pose_tensor = pose_tensor[-1]
        pose_image = _render_pose_condition(
            pose_tensor=pose_tensor,
            output_height=target_h,
            output_width=target_w,
        )

        reference_rgb = cv2.cvtColor(reference_np, cv2.COLOR_BGR2RGB)
        pose_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(reference_rgb)
        control_image = Image.fromarray(pose_rgb)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipe(
            prompt="photorealistic tennis player, broadcast sports shot",
            image=init_image,
            control_image=control_image,
            num_inference_steps=20,
            strength=0.65,
            guidance_scale=7.0,
            generator=generator,
        )
        generated_rgb = np.asarray(output.images[0], dtype=np.uint8)
        generated_bgr = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)


class AnimateAnyoneStrategy(BaseGenAIStrategy):
    """Wrapper strategy for local Animate Anyone runtime integration."""

    def __init__(
        self,
        repo_dir: str | None = None,
    ) -> None:
        self._repo_dir = repo_dir or os.environ.get("POINTSTREAM_ANIMATE_ANYONE_REPO_DIR")
        self._runtime_fn: Any | None = None

    def _ensure_runtime(self) -> Any:
        if self._runtime_fn is not None:
            return self._runtime_fn

        if self._repo_dir is None:
            raise FileNotFoundError(
                "Animate Anyone backend selected, but POINTSTREAM_ANIMATE_ANYONE_REPO_DIR is not set."
            )

        repo_path = Path(self._repo_dir)
        if not repo_path.exists() or not repo_path.is_dir():
            raise FileNotFoundError(
                f"Animate Anyone repository path does not exist: {repo_path}"
            )

        from src.decoder.animate_anyone_runtime import generate_frame

        runtime_fn = generate_frame
        self._runtime_fn = runtime_fn
        return runtime_fn

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        runtime_fn = self._ensure_runtime()

        reference_np = _to_numpy_bgr(reference_crop_tensor)
        pose_np = dense_dwpose_tensor.detach().cpu().numpy().astype(np.float32)

        try:
            generated_bgr = runtime_fn(
                reference_image_bgr=reference_np,
                dense_pose_sequence=pose_np,
                seed=int(seed),
                device=str(device),
                repo_dir=self._repo_dir,
            )
        except TypeError:
            # Keep tests/stubs simple when monkeypatching _ensure_runtime.
            generated_bgr = runtime_fn(
                reference_image_bgr=reference_np,
                dense_pose_sequence=pose_np,
                seed=int(seed),
                device=str(device),
            )

        generated_bgr = np.asarray(generated_bgr, dtype=np.uint8)
        if generated_bgr.ndim != 3 or generated_bgr.shape[2] != 3:
            raise ValueError(
                f"Animate Anyone runtime returned invalid frame shape: {tuple(generated_bgr.shape)}"
            )
        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)


class MockCompositor:
    """Lightweight fallback compositor used when GenAI is disabled."""

    def __init__(self, confidence_threshold: float = 0.2) -> None:
        self._confidence_threshold = float(confidence_threshold)

    @gpu_bound
    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
    ) -> torch.Tensor:
        frame_np = self._to_frame_numpy(warped_background_frame)
        pose_np = self._to_pose_numpy(dense_dwpose_tensor)
        crop_np = self._to_crop_numpy(reference_crop_tensor)

        x1, y1, x2, y2 = self._estimate_bbox_from_pose(pose_np=pose_np, frame_height=frame_np.shape[0], frame_width=frame_np.shape[1])

        # Draw a filled placeholder silhouette to prove compositor wiring end-to-end.
        cv2.rectangle(frame_np, (x1, y1), (x2, y2), color=(35, 80, 210), thickness=-1)

        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        resized_crop = cv2.resize(crop_np, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        frame_np[y1:y2, x1:x2] = resized_crop

        return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

    def _to_frame_numpy(self, frame_tensor: torch.Tensor) -> np.ndarray:
        if frame_tensor.ndim != 3:
            raise ValueError(f"Expected frame tensor [C,H,W], got shape {tuple(frame_tensor.shape)}")
        frame_np = frame_tensor.detach().cpu().permute(1, 2, 0).numpy()
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return frame_np.copy()

    def _to_pose_numpy(self, pose_tensor: torch.Tensor) -> np.ndarray:
        pose_np = pose_tensor.detach().cpu().numpy()
        if pose_np.ndim == 3:
            # Use first frame when a temporal tensor [Frames, 18, 3] is provided.
            pose_np = pose_np[0]
        if pose_np.shape != (18, 3):
            raise ValueError(f"Expected pose tensor shape (18, 3), got {tuple(pose_np.shape)}")
        return pose_np.astype(np.float32, copy=False)

    def _to_crop_numpy(self, crop_tensor: torch.Tensor) -> np.ndarray:
        if crop_tensor.ndim != 3:
            raise ValueError(f"Expected crop tensor [C,H,W], got shape {tuple(crop_tensor.shape)}")
        crop_np = crop_tensor.detach().cpu().permute(1, 2, 0).numpy()
        if crop_np.dtype != np.uint8:
            crop_np = np.clip(crop_np, 0, 255).astype(np.uint8)
        if crop_np.shape[2] != 3:
            raise ValueError(f"Expected crop tensor with 3 channels, got shape {tuple(crop_np.shape)}")
        return crop_np

    def _estimate_bbox_from_pose(self, pose_np: np.ndarray, frame_height: int, frame_width: int) -> tuple[int, int, int, int]:
        valid = pose_np[:, 2] >= self._confidence_threshold
        if not np.any(valid):
            cx = frame_width // 2
            cy = frame_height // 2
            half_w = max(8, frame_width // 10)
            half_h = max(12, frame_height // 6)
            return (
                max(0, cx - half_w),
                max(0, cy - half_h),
                min(frame_width, cx + half_w),
                min(frame_height, cy + half_h),
            )

        xs = pose_np[valid, 0]
        ys = pose_np[valid, 1]
        x1 = int(np.floor(np.min(xs)))
        y1 = int(np.floor(np.min(ys)))
        x2 = int(np.ceil(np.max(xs)))
        y2 = int(np.ceil(np.max(ys)))

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(3, int(round(width * 0.15)))
        pad_y = max(4, int(round(height * 0.20)))

        bx1 = max(0, x1 - pad_x)
        by1 = max(0, y1 - pad_y)
        bx2 = min(frame_width, x2 + pad_x)
        by2 = min(frame_height, y2 + pad_y)

        if bx2 <= bx1:
            bx2 = min(frame_width, bx1 + 1)
        if by2 <= by1:
            by2 = min(frame_height, by1 + 1)
        return bx1, by1, bx2, by2


class DiffusersCompositor(MockCompositor):
    """Feature-gated real GenAI compositor with strategy-selectable backends."""

    def __init__(
        self,
        confidence_threshold: float = 0.2,
        backend: str | None = None,
        seed: int = 1337,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        self._seed = int(seed)
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        backend_value = backend if backend is not None else os.environ.get("POINTSTREAM_GENAI_BACKEND")
        if backend_value is None:
            backend_value = "controlnet"
        self._backend = backend_value.strip().lower()
        threshold = int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_TRANSPARENT_THRESHOLD", "8"))
        self._animate_anyone_transparent_threshold = int(np.clip(threshold, 0, 255))
        self._strategy = self._build_strategy(self._backend)

    def _build_strategy(self, backend: str) -> BaseGenAIStrategy:
        if backend in {"controlnet", "baseline", "baseline-controlnet"}:
            return BaselineControlNetStrategy()
        if backend in {"animate-anyone", "animate_anyone", "animateanyone"}:
            return AnimateAnyoneStrategy()
        raise ValueError(f"Unsupported POINTSTREAM_GENAI_BACKEND value: {backend}")

    @gpu_bound
    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
    ) -> torch.Tensor:
        # Deterministic generation is required so residual encoding remains stable.
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        generated_actor = self._strategy.generate(
            reference_crop_tensor=reference_crop_tensor,
            dense_dwpose_tensor=dense_dwpose_tensor,
            seed=self._seed,
            device=self._device,
        )

        frame_np = self._to_frame_numpy(warped_background_frame)
        generated_np = self._to_crop_numpy(generated_actor)
        pose_np = self._to_pose_numpy(dense_dwpose_tensor)

        x1, y1, x2, y2 = self._estimate_bbox_from_pose(
            pose_np=pose_np,
            frame_height=int(frame_np.shape[0]),
            frame_width=int(frame_np.shape[1]),
        )

        target_h = max(1, y2 - y1)
        target_w = max(1, x2 - x1)

        is_animate_anyone = isinstance(self._strategy, AnimateAnyoneStrategy)
        if is_animate_anyone:
            actor_resized = self._resize_actor_with_aspect_recovery(
                actor_bgr=generated_np,
                target_w=target_w,
                target_h=target_h,
            )
            mask = self._segment_black_background(actor_resized)
        else:
            actor_resized = cv2.resize(generated_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = self._segment_foreground(actor_resized)

        roi = frame_np[y1:y2, x1:x2]
        roi[mask] = actor_resized[mask]
        frame_np[y1:y2, x1:x2] = roi
        return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

    def _resize_actor_with_aspect_recovery(self, actor_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        frame_h, frame_w = actor_bgr.shape[:2]
        if target_w <= 0 or target_h <= 0:
            return actor_bgr

        # Inverse of resize-and-pad used by legacy PointStream path.
        if target_w >= target_h:
            content_w = frame_w
            content_h = max(1, int(round(frame_h * (float(target_h) / float(target_w)))))
            content_h = min(content_h, frame_h)
            pad_top = max(0, (frame_h - content_h) // 2)
            pad_bottom = max(0, frame_h - content_h - pad_top)
            cropped = actor_bgr[pad_top:frame_h - pad_bottom, :]
        else:
            content_h = frame_h
            content_w = max(1, int(round(frame_w * (float(target_w) / float(target_h)))))
            content_w = min(content_w, frame_w)
            pad_left = max(0, (frame_w - content_w) // 2)
            pad_right = max(0, frame_w - content_w - pad_left)
            cropped = actor_bgr[:, pad_left:frame_w - pad_right]

        if cropped.size == 0:
            cropped = actor_bgr

        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _segment_black_background(self, actor_bgr: np.ndarray) -> np.ndarray:
        threshold = self._animate_anyone_transparent_threshold
        mask = np.any(actor_bgr > threshold, axis=2)

        mask_u8 = np.asarray(mask.astype(np.uint8) * 255, dtype=np.uint8)
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        closed = cv2.morphologyEx(
            np.asarray(opened, dtype=np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        binary = np.asarray(closed, dtype=np.uint8) > 0

        min_pixels = max(10, actor_bgr.shape[0] * actor_bgr.shape[1] // 80)
        if int(np.count_nonzero(binary)) < min_pixels:
            return np.ones(actor_bgr.shape[:2], dtype=bool)
        return binary

    def _segment_foreground(self, actor_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(actor_bgr, cv2.COLOR_BGR2HSV)
        # Remove near-black/near-gray generated background with a simple color-energy mask.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (val > 24) & (sat > 20)

        mask_u8 = np.asarray(mask.astype(np.uint8) * 255, dtype=np.uint8)
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        closed = cv2.morphologyEx(
            np.asarray(opened, dtype=np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        binary = np.asarray(closed, dtype=np.uint8) > 0
        if int(np.count_nonzero(binary)) < max(10, actor_bgr.shape[0] * actor_bgr.shape[1] // 50):
            return np.ones(actor_bgr.shape[:2], dtype=bool)
        return binary


def _to_numpy_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(image_tensor.shape)}")
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if image_np.dtype != np.uint8:
        image_np = np.asarray(np.clip(image_np, 0, 255), dtype=np.uint8)
    if image_np.shape[2] != 3:
        raise ValueError(f"Expected BGR image tensor with 3 channels, got {tuple(image_np.shape)}")
    return image_np


def _render_pose_condition(pose_tensor: torch.Tensor, output_height: int, output_width: int) -> np.ndarray:
    pose_np = pose_tensor.detach().cpu().numpy()
    if pose_np.shape != (18, 3):
        raise ValueError(f"Expected pose tensor shape (18, 3), got {tuple(pose_np.shape)}")

    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    valid = pose_np[:, 2] >= 0.2
    points = pose_np[:, :2].astype(np.int32)

    for idx in np.where(valid)[0]:
        px = int(np.clip(points[idx, 0], 0, output_width - 1))
        py = int(np.clip(points[idx, 1], 0, output_height - 1))
        cv2.circle(canvas, (px, py), 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    limb_edges = [
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
    for a, b in limb_edges:
        if not (valid[a] and valid[b]):
            continue
        ax = int(np.clip(points[a, 0], 0, output_width - 1))
        ay = int(np.clip(points[a, 1], 0, output_height - 1))
        bx = int(np.clip(points[b, 0], 0, output_width - 1))
        by = int(np.clip(points[b, 1], 0, output_height - 1))
        cv2.line(canvas, (ax, ay), (bx, by), (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return canvas


# Backward-compatible alias used by existing decode tests.
GenAICompositor = MockCompositor
