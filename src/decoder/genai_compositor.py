from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.tags import gpu_bound
from src.shared.torch_dtype import is_cuda_device_usable, resolve_torch_dtype_for_device


_LOGGER = logging.getLogger(__name__)


def _resolve_local_weight_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    assets_candidate = project_root / "assets" / "weights" / model_name
    if assets_candidate.exists():
        return assets_candidate

    return None


def _require_local_or_optin_weight(model_name: str) -> str:
    local_path = _resolve_local_weight_path(model_name)
    if local_path is not None:
        return str(local_path)

    if os.environ.get("POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD", "0") == "1":
        return model_name

    raise FileNotFoundError(
        f"Required model weights not found for '{model_name}'. "
        "Place weights in assets/weights/ or set POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD=1."
    )


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

        dtype = resolve_torch_dtype_for_device(
            device,
            default_cuda=torch.float16,
            allowed_cuda={torch.float16, torch.bfloat16, torch.float32},
        )
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
        actor_identity: str | None = None,
        metadata_mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        _ = actor_identity
        _ = metadata_mask
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
            # Use latest frame when a temporal tensor [Frames, 18, 3] is provided.
            pose_np = pose_np[-1]
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
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")
        backend_value = backend if backend is not None else os.environ.get("POINTSTREAM_GENAI_BACKEND")
        if backend_value is None:
            backend_value = "controlnet"
        self._backend = backend_value.strip().lower()
        threshold = int(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_TRANSPARENT_THRESHOLD", "8"))
        self._animate_anyone_transparent_threshold = int(np.clip(threshold, 0, 255))
        resize_mode = os.environ.get("POINTSTREAM_GENAI_RESIZE_MODE", "aspect-recovery").strip().lower()
        if resize_mode not in {"plain", "aspect-recovery"}:
            resize_mode = "aspect-recovery"
        self._resize_mode = resize_mode

        adaptive_raw = os.environ.get("POINTSTREAM_ANIMATE_ANYONE_ADAPTIVE_THRESHOLD", "1").strip().lower()
        self._use_adaptive_black_threshold = adaptive_raw not in {"0", "false", "off", "no"}

        alpha_smoothing_raw = float(os.environ.get("POINTSTREAM_ANIMATE_ANYONE_ALPHA_SMOOTHING", "0.25"))
        self._alpha_temporal_smoothing = float(np.clip(alpha_smoothing_raw, 0.0, 0.95))
        self._alpha_history_by_actor: dict[str, np.ndarray] = {}

        raw_mask_mode = os.environ.get("POINTSTREAM_COMPOSITING_MASK_MODE", "alpha-heuristic").strip().lower()
        mask_mode_aliases = {
            "alpha": "alpha-heuristic",
            "alpha-heuristic": "alpha-heuristic",
            "heuristic": "alpha-heuristic",
            "metadata-source-mask": "metadata-source-mask",
            "metadata-mask": "metadata-source-mask",
            "source-mask": "metadata-source-mask",
            "postgen-seg-client": "postgen-seg-client",
            "postgen": "postgen-seg-client",
        }
        self._compositing_mask_mode = mask_mode_aliases.get(raw_mask_mode, "alpha-heuristic")

        backend_raw = os.environ.get("POINTSTREAM_POSTGEN_SEGMENTER_BACKEND", "yolo").strip().lower()
        self._postgen_segmenter_backend = backend_raw if backend_raw in {"yolo", "heuristic"} else "yolo"
        self._postgen_segmenter_model = os.environ.get("POINTSTREAM_POSTGEN_SEGMENTER_MODEL", "yolo26n-seg.pt")
        self._postgen_segmenter: Any | None = None
        self._postgen_segmenter_disabled = False

        self._strategy = self._build_strategy(self._backend)

    def _build_strategy(self, backend: str) -> BaseGenAIStrategy:
        if backend in {"controlnet", "baseline", "baseline-controlnet"}:
            return BaselineControlNetStrategy()
        if backend in {"animate-anyone", "animate_anyone", "animateanyone"}:
            return AnimateAnyoneStrategy()
        raise ValueError(f"Unsupported POINTSTREAM_GENAI_BACKEND value: {backend}")

    def uses_temporal_pose_sequence(self) -> bool:
        return isinstance(self._strategy, AnimateAnyoneStrategy)

    @gpu_bound
    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
        actor_identity: str | None = None,
        metadata_mask: np.ndarray | None = None,
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

        is_animate_anyone = self.uses_temporal_pose_sequence()
        if is_animate_anyone and self._resize_mode == "aspect-recovery":
            actor_resized = self._resize_actor_with_aspect_recovery(generated_np, target_w=target_w, target_h=target_h)
        else:
            actor_resized = cv2.resize(generated_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        alpha_mask = self._select_compositing_alpha(
            actor_resized=actor_resized,
            metadata_mask=metadata_mask,
            is_animate_anyone=is_animate_anyone,
        )
        alpha_mask = self._apply_temporal_alpha_smoothing(alpha_mask=alpha_mask, actor_identity=actor_identity)

        if alpha_mask is None or int(np.count_nonzero(alpha_mask > 0.01)) == 0:
            return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

        roi = frame_np[y1:y2, x1:x2]
        alpha_3 = np.asarray(alpha_mask[:, :, None], dtype=np.float32)
        blended = actor_resized.astype(np.float32) * alpha_3 + roi.astype(np.float32) * (1.0 - alpha_3)
        frame_np[y1:y2, x1:x2] = np.asarray(np.clip(blended, 0.0, 255.0), dtype=np.uint8)
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

    def _select_compositing_alpha(
        self,
        actor_resized: np.ndarray,
        metadata_mask: np.ndarray | None,
        is_animate_anyone: bool,
    ) -> np.ndarray | None:
        mode = self._compositing_mask_mode

        if mode == "metadata-source-mask":
            alpha = self._alpha_from_metadata_mask(
                metadata_mask=metadata_mask,
                target_w=int(actor_resized.shape[1]),
                target_h=int(actor_resized.shape[0]),
            )
            if alpha is not None:
                return alpha

        if mode == "postgen-seg-client":
            alpha = self._segment_generated_actor(actor_resized=actor_resized, is_animate_anyone=is_animate_anyone)
            if alpha is not None:
                return alpha

        if is_animate_anyone:
            return self._segment_black_background(actor_resized)
        return self._segment_foreground(actor_resized)

    def _alpha_from_metadata_mask(
        self,
        metadata_mask: np.ndarray | None,
        target_w: int,
        target_h: int,
    ) -> np.ndarray | None:
        if metadata_mask is None:
            return None

        raw = np.asarray(metadata_mask)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        if raw.ndim != 2 or raw.size == 0:
            return None

        if raw.shape[0] != target_h or raw.shape[1] != target_w:
            raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if raw.dtype != np.uint8:
            raw_float = np.asarray(raw, dtype=np.float32)
            raw = np.asarray(raw_float > 0.5, dtype=np.uint8) * 255
        else:
            raw = np.asarray(raw > 127, dtype=np.uint8) * 255

        min_pixels = max(8, target_h * target_w // 120)
        return self._postprocess_binary_mask(np.asarray(raw, dtype=np.uint8), min_pixels=min_pixels)

    def _segment_generated_actor(self, actor_resized: np.ndarray, is_animate_anyone: bool) -> np.ndarray | None:
        if self._postgen_segmenter_backend == "yolo":
            alpha = self._segment_generated_actor_with_yolo(actor_resized=actor_resized)
            if alpha is not None:
                return alpha

        # Always keep a heuristic fallback so ablations remain robust when model runtime is unavailable.
        if is_animate_anyone:
            return self._segment_black_background(actor_resized)
        return self._segment_foreground(actor_resized)

    def _segment_generated_actor_with_yolo(self, actor_resized: np.ndarray) -> np.ndarray | None:
        model = self._ensure_postgen_segmenter()
        if model is None:
            return None

        try:
            results = model.predict(source=actor_resized, classes=[0], verbose=False, conf=0.2)
        except Exception as exc:
            if not self._postgen_segmenter_disabled:
                _LOGGER.warning("Disabling post-generation segmenter after inference failure: %s", exc)
            self._postgen_segmenter_disabled = True
            self._postgen_segmenter = None
            return None

        if not results:
            return None

        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None or len(masks.data) == 0:
            return None

        mask_np = masks.data[0]
        if hasattr(mask_np, "cpu"):
            mask_np = mask_np.cpu().numpy()

        mask_u8 = np.asarray(np.asarray(mask_np, dtype=np.float32) > 0.5, dtype=np.uint8) * 255
        target_h, target_w = actor_resized.shape[:2]
        if mask_u8.shape[:2] != (target_h, target_w):
            mask_u8 = np.asarray(
                cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST),
                dtype=np.uint8,
            )

        min_pixels = max(8, target_h * target_w // 120)
        return self._postprocess_binary_mask(np.asarray(mask_u8, dtype=np.uint8), min_pixels=min_pixels)

    def _ensure_postgen_segmenter(self) -> Any | None:
        if self._postgen_segmenter_disabled:
            return None
        if self._postgen_segmenter is not None:
            return self._postgen_segmenter

        try:
            from ultralytics import YOLO

            weight_ref = _require_local_or_optin_weight(self._postgen_segmenter_model)
            self._postgen_segmenter = YOLO(weight_ref)
            return self._postgen_segmenter
        except Exception as exc:
            _LOGGER.warning(
                "Post-generation segmenter is unavailable; falling back to heuristic alpha extraction: %s",
                exc,
            )
            self._postgen_segmenter_disabled = True
            self._postgen_segmenter = None
            return None

    def _segment_black_background(self, actor_bgr: np.ndarray) -> np.ndarray | None:
        threshold = self._animate_anyone_transparent_threshold
        span_threshold = 6
        if self._use_adaptive_black_threshold:
            threshold, span_threshold = self._estimate_adaptive_black_thresholds(
                actor_bgr=actor_bgr,
                base_threshold=threshold,
                base_span_threshold=span_threshold,
            )

        max_channel = np.max(actor_bgr, axis=2)
        channel_span = np.max(actor_bgr, axis=2) - np.min(actor_bgr, axis=2)
        mask = (max_channel > threshold) | (channel_span > span_threshold)
        mask_u8 = np.asarray(mask.astype(np.uint8) * 255, dtype=np.uint8)

        min_pixels = max(10, actor_bgr.shape[0] * actor_bgr.shape[1] // 80)
        return self._postprocess_binary_mask(mask_u8, min_pixels=min_pixels)

    def _segment_foreground(self, actor_bgr: np.ndarray) -> np.ndarray | None:
        hsv = cv2.cvtColor(actor_bgr, cv2.COLOR_BGR2HSV)
        # Remove near-black/near-gray generated background with a simple color-energy mask.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (val > 24) & ((sat > 16) | (val > 42))

        mask_u8 = np.asarray(mask.astype(np.uint8) * 255, dtype=np.uint8)
        min_pixels = max(10, actor_bgr.shape[0] * actor_bgr.shape[1] // 50)
        return self._postprocess_binary_mask(mask_u8, min_pixels=min_pixels)

    def _postprocess_binary_mask(self, mask_u8: np.ndarray, min_pixels: int) -> np.ndarray | None:
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        closed = cv2.morphologyEx(
            np.asarray(opened, dtype=np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        largest = self._keep_largest_component(np.asarray(closed, dtype=np.uint8))
        filled = self._fill_mask_holes(largest)

        if int(np.count_nonzero(filled)) < int(min_pixels):
            return None
        return self._to_soft_alpha(filled)

    def _keep_largest_component(self, mask_u8: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask_u8 > 0, dtype=np.uint8)
        if int(np.count_nonzero(binary)) == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        kept = np.asarray(labels == best_label, dtype=np.uint8) * 255
        return kept

    def _fill_mask_holes(self, mask_u8: np.ndarray) -> np.ndarray:
        if int(np.count_nonzero(mask_u8)) == 0:
            return np.zeros_like(mask_u8, dtype=np.uint8)

        h, w = mask_u8.shape[:2]
        flood = np.asarray(mask_u8, dtype=np.uint8).copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), (255,))
        flood_inv = cv2.bitwise_not(flood)
        return cv2.bitwise_or(np.asarray(mask_u8, dtype=np.uint8), flood_inv)

    def _to_soft_alpha(self, mask_u8: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(np.asarray(mask_u8, dtype=np.uint8), (5, 5), 0)
        alpha = np.asarray(blurred, dtype=np.float32) / 255.0
        alpha[alpha < 0.05] = 0.0
        alpha[alpha > 0.98] = 1.0
        return np.asarray(np.clip(alpha, 0.0, 1.0), dtype=np.float32)

    def _estimate_adaptive_black_thresholds(
        self,
        actor_bgr: np.ndarray,
        base_threshold: int,
        base_span_threshold: int,
    ) -> tuple[int, int]:
        border_pixels = self._extract_border_pixels(actor_bgr)
        if border_pixels.size == 0:
            return int(base_threshold), int(base_span_threshold)

        border_max = np.max(border_pixels, axis=1)
        border_span = np.max(border_pixels, axis=1) - np.min(border_pixels, axis=1)

        adaptive_threshold = int(np.clip(np.percentile(border_max, 95) + 3.0, 0.0, 255.0))
        adaptive_span = int(np.clip(np.percentile(border_span, 95) + 2.0, 0.0, 255.0))

        return max(int(base_threshold), adaptive_threshold), max(int(base_span_threshold), adaptive_span)

    def _extract_border_pixels(self, actor_bgr: np.ndarray) -> np.ndarray:
        h, w = actor_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.empty((0, 3), dtype=np.uint8)

        border = max(1, min(h, w) // 32)
        top = actor_bgr[:border, :, :].reshape(-1, 3)
        bottom = actor_bgr[h - border :, :, :].reshape(-1, 3)
        left = actor_bgr[:, :border, :].reshape(-1, 3)
        right = actor_bgr[:, w - border :, :].reshape(-1, 3)
        return np.concatenate([top, bottom, left, right], axis=0)

    def _apply_temporal_alpha_smoothing(
        self,
        alpha_mask: np.ndarray | None,
        actor_identity: str | None,
    ) -> np.ndarray | None:
        key = actor_identity if actor_identity is not None else "__default_actor__"

        if alpha_mask is None:
            self._alpha_history_by_actor.pop(key, None)
            return None

        current = np.asarray(alpha_mask, dtype=np.float32)
        smoothing = float(self._alpha_temporal_smoothing)
        if smoothing <= 0.0:
            self._alpha_history_by_actor[key] = current
            return current

        previous = self._alpha_history_by_actor.get(key)
        if previous is None or previous.shape != current.shape:
            self._alpha_history_by_actor[key] = current
            return current

        blended = np.asarray(previous * smoothing + current * (1.0 - smoothing), dtype=np.float32)
        self._alpha_history_by_actor[key] = blended
        return blended


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

    return draw_dwpose_canvas(
        height=int(output_height),
        width=int(output_width),
        people_dw=pose_np[np.newaxis, ...],
        confidence_threshold=0.2,
    )


# Backward-compatible alias used by existing decode tests.
GenAICompositor = MockCompositor
