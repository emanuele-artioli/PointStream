from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.genai_debug import export_compositor_artifacts
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


def _require_local_or_optin_weight(model_name: str, allow_download: bool = False) -> str:
    local_path = _resolve_local_weight_path(model_name)
    if local_path is not None:
        return str(local_path)

    if allow_download:
        return model_name

    raise FileNotFoundError(
        f"Required model weights not found for '{model_name}'. "
        "Place weights in assets/weights/ or set allow-auto-model-download to true."
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
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_debug_inputs(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        return {}



class BaselineControlNetStrategy(BaseGenAIStrategy):
    """Standard SD + ControlNet OpenPose baseline backend."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_openpose",
        config: Any = None,
    ) -> None:
        self._model_id = model_id
        self._controlnet_id = controlnet_id
        self.config = config
        self._pipe: Any | None = None
        
        self._width = int(config.controlnet_width) if config and hasattr(config, "controlnet_width") else 512
        self._height = int(config.controlnet_height) if config and hasattr(config, "controlnet_height") else 512
        self._steps = int(config.controlnet_steps) if config and hasattr(config, "controlnet_steps") else 20
        self._strength = float(config.controlnet_strength) if config and hasattr(config, "controlnet_strength") else 0.65
        self._cfg = float(config.controlnet_cfg) if config and hasattr(config, "controlnet_cfg") else 7.0

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
            config_dtype=self.config.gpu_dtype if self.config and hasattr(self.config, "gpu_dtype") else None,
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

    def get_debug_inputs(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        artifacts = {}
        ref_np = _to_numpy_bgr(reference_crop_tensor)
        artifacts["00_reference_crop.png"] = ref_np
        
        try:
            pose_tensor = dense_dwpose_tensor.clone()
            if pose_tensor.ndim == 3:
                pose_tensor = pose_tensor[-1]
            pose_np_raw = pose_tensor.cpu().numpy()
            
            valid = pose_np_raw[:, 2] >= 0.2
            if np.any(valid):
                xs = pose_np_raw[valid, 0]
                ys = pose_np_raw[valid, 1]
                x1 = int(np.floor(np.min(xs)))
                y1 = int(np.floor(np.min(ys)))
                x2 = int(np.ceil(np.max(xs)))
                y2 = int(np.ceil(np.max(ys)))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                gen_h = max(8, (bh // 8) * 8)
                gen_w = max(8, (bw // 8) * 8)
                
                pose_tensor[..., 0] -= x1
                pose_tensor[..., 1] -= y1
                pose_tensor[..., 0] *= float(gen_w) / float(bw)
                pose_tensor[..., 1] *= float(gen_h) / float(bh)
                target_h, target_w = gen_h, gen_w
            else:
                target_h = int(reference_crop_tensor.shape[1])
                target_w = int(reference_crop_tensor.shape[2])
                
            pose_np = _render_pose_condition(pose_tensor, output_height=target_h, output_width=target_w)
            pose_np = cv2.cvtColor(pose_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            _LOGGER.warning(f"Failed to render pose for debug output: {e}")
            pose_np = dense_dwpose_tensor.cpu().numpy()
            if pose_np.ndim == 3:
                pose_np = pose_np[-1]
            pose_np = np.asarray(pose_np * 255.0, dtype=np.uint8) if pose_np.dtype != np.uint8 else pose_np
            if len(pose_np.shape) == 3 and pose_np.shape[0] == 3:
                pose_np = np.transpose(pose_np, (1, 2, 0))
        artifacts["01_pose_condition.png"] = pose_np
        return artifacts

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        pipe = self._ensure_pipeline(device)
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("Pillow is required for ControlNet strategy") from exc

        reference_np = _to_numpy_bgr(reference_crop_tensor)

        if metadata_bbox is not None:
            x1, y1, x2, y2 = metadata_bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            
            scale = float(max(self._width, self._height)) / max(bh, bw)
            scaled_h = int(bh * scale)
            scaled_w = int(bw * scale)
            
            offset_x = (self._width - scaled_w) // 2
            offset_y = (self._height - scaled_h) // 2
            
            pose_tensor = dense_dwpose_tensor.clone()
            pose_tensor[..., 0] -= x1
            pose_tensor[..., 1] -= y1
            pose_tensor[..., 0] *= float(scaled_w) / float(bw)
            pose_tensor[..., 1] *= float(scaled_h) / float(bh)
            pose_tensor[..., 0] += offset_x
            pose_tensor[..., 1] += offset_y
            
        else:
            bh, bw = int(reference_np.shape[0]), int(reference_np.shape[1])
            scale = float(max(self._width, self._height)) / max(bh, bw)
            scaled_h = int(bh * scale)
            scaled_w = int(bw * scale)
            
            offset_x = (self._width - scaled_w) // 2
            offset_y = (self._height - scaled_h) // 2
            
            pose_tensor = dense_dwpose_tensor.clone()
            pose_tensor[..., 0] *= float(scaled_w) / float(bw)
            pose_tensor[..., 1] *= float(scaled_h) / float(bh)
            pose_tensor[..., 0] += offset_x
            pose_tensor[..., 1] += offset_y

        if pose_tensor.ndim == 3:
            pose_tensor = pose_tensor[-1]
            
        pose_image = _render_pose_condition(
            pose_tensor=pose_tensor,
            output_height=self._height,
            output_width=self._width,
        )

        reference_rgb = cv2.cvtColor(reference_np, cv2.COLOR_BGR2RGB)
        if reference_rgb.shape[0] != scaled_h or reference_rgb.shape[1] != scaled_w:
            reference_rgb = cv2.resize(reference_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            
        padded_reference = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        padded_reference[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = reference_rgb
            
        # _render_pose_condition returns an RGB canvas natively. Do not swap to BGR.
        pose_rgb = pose_image
        init_image = Image.fromarray(padded_reference)
        control_image = Image.fromarray(pose_rgb)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipe(
            prompt="photorealistic tennis player, broadcast sports shot",
            image=init_image,
            control_image=control_image,
            height=self._height,
            width=self._width,
            num_inference_steps=self._steps,
            strength=self._strength,
            guidance_scale=self._cfg,
            generator=generator,
        )
        generated_rgb = np.asarray(output.images[0], dtype=np.uint8)
        generated_cropped = generated_rgb[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w]
        generated_bgr = cv2.cvtColor(generated_cropped, cv2.COLOR_RGB2BGR)
        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)


class AnimateAnyoneStrategy(BaseGenAIStrategy):
    """Wrapper strategy for local Animate Anyone runtime integration."""

    def __init__(
        self,
        repo_dir: str | None = None,
        config: Any = None,
    ) -> None:
        self.config = config
        self._repo_dir = repo_dir or (self.config.animate_anyone_repo_dir if self.config else None)
        
        self._steps = int(self.config.animate_anyone_steps if self.config and hasattr(self.config, "animate_anyone_steps") else 3)
        self._cfg = float(self.config.animate_anyone_cfg if self.config and hasattr(self.config, "animate_anyone_cfg") else 7.5)
        self._width = int(self.config.animate_anyone_width if self.config and hasattr(self.config, "animate_anyone_width") else 256)
        self._height = int(self.config.animate_anyone_height if self.config and hasattr(self.config, "animate_anyone_height") else 256)
        self._window = int(self.config.animate_anyone_window) if self.config and getattr(self.config, "animate_anyone_window", None) else None
        self._model_dir = getattr(self.config, "animate_anyone_model_dir", None) if self.config else None
        self._model_variant = getattr(self.config, "animate_anyone_model_variant", "finetuned_tennis") if self.config else "finetuned_tennis"
        self._gpu_dtype = getattr(self.config, "gpu_dtype", None) if self.config else None
        
        self._runtime_fn: Any | None = None

    def _ensure_runtime(self) -> Any:
        if self._runtime_fn is not None:
            return self._runtime_fn

        from src.decoder.animate_anyone_runtime import generate_frame

        runtime_fn = generate_frame
        self._runtime_fn = runtime_fn
        return runtime_fn

    def get_debug_inputs(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        artifacts = {}
        ref_np = _to_numpy_bgr(reference_crop_tensor)
        artifacts["00_reference_crop.png"] = ref_np
        
        try:
            pose_tensor = dense_dwpose_tensor.clone()
            if pose_tensor.ndim == 3:
                pose_tensor = pose_tensor[-1]
            pose_np_raw = pose_tensor.cpu().numpy()
            
            valid = pose_np_raw[:, 2] >= 0.2
            if np.any(valid):
                xs = pose_np_raw[valid, 0]
                ys = pose_np_raw[valid, 1]
                x1 = int(np.floor(np.min(xs)))
                y1 = int(np.floor(np.min(ys)))
                x2 = int(np.ceil(np.max(xs)))
                y2 = int(np.ceil(np.max(ys)))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                gen_h = max(8, (bh // 8) * 8)
                gen_w = max(8, (bw // 8) * 8)
                
                pose_tensor[..., 0] -= x1
                pose_tensor[..., 1] -= y1
                pose_tensor[..., 0] *= float(gen_w) / float(bw)
                pose_tensor[..., 1] *= float(gen_h) / float(bh)
                target_h, target_w = gen_h, gen_w
            else:
                target_h = int(reference_crop_tensor.shape[1])
                target_w = int(reference_crop_tensor.shape[2])
                
            pose_np = _render_pose_condition(pose_tensor, output_height=target_h, output_width=target_w)
            pose_np = cv2.cvtColor(pose_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            _LOGGER.warning(f"Failed to render pose for debug output: {e}")
            pose_np = dense_dwpose_tensor.cpu().numpy()
            if pose_np.ndim == 3:
                pose_np = pose_np[-1]
            pose_np = np.asarray(pose_np * 255.0, dtype=np.uint8) if pose_np.dtype != np.uint8 else pose_np
            if len(pose_np.shape) == 3 and pose_np.shape[0] == 3:
                pose_np = np.transpose(pose_np, (1, 2, 0))
        artifacts["01_pose_condition.png"] = pose_np
        return artifacts

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
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
                steps=self._steps,
                cfg=self._cfg,
                width=self._width,
                height=self._height,
                model_dir=self._model_dir,
                model_variant=self._model_variant,
                gpu_dtype=self._gpu_dtype,
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

    def generate_sequence(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        from src.decoder.animate_anyone_runtime import generate_sequence

        reference_np = _to_numpy_bgr(reference_crop_tensor)
        pose_np = dense_dwpose_tensor.detach().cpu().numpy().astype(np.float32)

        try:
            generated_bgr = generate_sequence(
                reference_image_bgr=reference_np,
                dense_pose_sequence=pose_np,
                seed=int(seed),
                device=str(device),
                repo_dir=self._repo_dir,
                steps=self._steps,
                cfg=self._cfg,
                width=self._width,
                height=self._height,
                window=self._window,
                model_dir=self._model_dir,
                model_variant=self._model_variant,
                gpu_dtype=self._gpu_dtype,
            )
        except TypeError:
            generated_bgr = generate_sequence(
                reference_image_bgr=reference_np,
                dense_pose_sequence=pose_np,
                seed=int(seed),
                device=str(device),
            )

        generated_bgr = np.asarray(generated_bgr, dtype=np.uint8)
        if generated_bgr.ndim != 4 or generated_bgr.shape[-1] != 3:
            raise ValueError(
                f"Animate Anyone runtime returned invalid sequence shape: {tuple(generated_bgr.shape)}"
            )
        return torch.from_numpy(generated_bgr).permute(0, 3, 1, 2).contiguous().to(torch.uint8)


class BaseCompositor:
    """Base compositor providing helper methods."""

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
        metadata_bbox: tuple[int, int, int, int] | None = None,
        debug_dir: str | Path | None = None,
        frame_idx: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("BaseCompositor does not implement process. Subclasses must implement it.")

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



class DiffusersCompositor(BaseCompositor):
    """Feature-gated real GenAI compositor with strategy-selectable backends."""

    def __init__(
        self,
        confidence_threshold: float = 0.2,
        backend: str | None = None,
        seed: int = 1337,
        device: str | torch.device | None = None,
        config: Any = None,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        self.config = config
        self._seed = int(seed)
        self._debug_stage = "unknown"
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        if self._device.type == "cuda" and not is_cuda_device_usable(self._device):
            self._device = torch.device("cpu")
            
        backend_value = backend if backend is not None else (self.config.genai_backend if self.config else None)
        if backend_value is None:
            backend_value = "controlnet"
        self._backend = backend_value.strip().lower()
        
        threshold = self.config.animate_anyone_transparent_threshold if self.config else 8
        self._animate_anyone_transparent_threshold = int(np.clip(threshold, 0, 255))
        
        resize_mode = (self.config.genai_resize_mode if self.config else "aspect-recovery").strip().lower()
        if resize_mode not in {"plain", "aspect-recovery"}:
            resize_mode = "aspect-recovery"
        self._resize_mode = resize_mode

        self._use_adaptive_black_threshold = bool(self.config.animate_anyone_adaptive_threshold) if self.config else True

        alpha_smoothing_raw = float(self.config.animate_anyone_alpha_smoothing if self.config else 0.25)
        self._alpha_temporal_smoothing = float(np.clip(alpha_smoothing_raw, 0.0, 0.95))
        self._alpha_history_by_actor: dict[str, np.ndarray] = {}

        raw_mask_mode = (self.config.compositing_mask_mode if self.config else "postgen-seg-client").strip().lower()
        mask_mode_aliases = {
            "alpha": "alpha-heuristic",
            "alpha-heuristic": "alpha-heuristic",
            "heuristic": "alpha-heuristic",
            "metadata-source-mask": "metadata-source-mask",
            "metadata-mask": "metadata-source-mask",
            "source-mask": "metadata-source-mask",
            "postgen-seg-client": "postgen-seg-client",
            "postgen": "postgen-seg-client",
            "pose-heuristic-mask": "pose-heuristic-mask",
            "pose-hull": "pose-heuristic-mask",
        }
        self._compositing_mask_mode = mask_mode_aliases.get(raw_mask_mode, "postgen-seg-client")

        backend_raw = (self.config.postgen_segmenter_backend if self.config else "yolo").strip().lower()
        self._postgen_segmenter_backend = backend_raw if backend_raw in {"yolo", "heuristic"} else "yolo"
        self._postgen_segmenter_model = self.config.postgen_segmenter_model if self.config and self.config.postgen_segmenter_model else "yolo26n-seg.pt"
        self._postgen_segmenter: Any | None = None
        self._postgen_segmenter_disabled = False
        self._current_debug_artifacts: dict[str, np.ndarray] | None = None

        self._strategy = self._build_strategy(self._backend)

    def set_debug_stage(self, stage: str) -> None:
        self._debug_stage = str(stage)

    def clear_history(self) -> None:
        """Reset temporal state such as alpha smoothing history."""
        self._alpha_history_by_actor.clear()

    def _build_strategy(self, backend: str) -> BaseGenAIStrategy:
        if backend in {"controlnet", "baseline", "baseline-controlnet"}:
            return BaselineControlNetStrategy(config=self.config)
        if backend in {"animate-anyone", "animate_anyone", "animateanyone"}:
            return AnimateAnyoneStrategy(config=self.config)
        if backend in {"caption-controlnet", "caption_controlnet"}:
            from src.decoder.controlnet_engine import CaptionControlNetStrategy
            return CaptionControlNetStrategy(config=self.config)
        if backend in {"mock-caption-controlnet", "mock_caption_controlnet"}:
            from src.decoder.controlnet_engine import MockCaptionControlNetStrategy
            return MockCaptionControlNetStrategy(config=self.config)
        if backend in {"ip-adapter-controlnet", "ip_adapter_controlnet"}:
            from src.decoder.controlnet_engine import IPAdapterControlNetStrategy
            return IPAdapterControlNetStrategy(config=self.config)
        if backend in {"canny-controlnet", "canny_controlnet"}:
            from src.decoder.controlnet_engine import CannyControlNetStrategy
            return CannyControlNetStrategy(config=self.config)
        if backend in {"seg-controlnet", "seg_controlnet"}:
            from src.decoder.controlnet_engine import SegControlNetStrategy
            return SegControlNetStrategy(config=self.config)
        raise ValueError(f"Unsupported genai-backend value in config: {backend}")

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
        metadata_bbox: tuple[int, int, int, int] | None = None,
        debug_dir: str | Path | None = None,
        frame_idx: int | None = None,
    ) -> torch.Tensor:
        # Deterministic generation is required so residual encoding remains stable.
        if self._seed is not None:
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

        x1, y1, x2, y2 = self._resolve_target_bbox(
            pose_np=self._to_pose_numpy(dense_dwpose_tensor),
            frame_height=int(warped_background_frame.shape[1]),
            frame_width=int(warped_background_frame.shape[2]),
            metadata_bbox=metadata_bbox,
        )
        resolved_bbox = (x1, y1, x2, y2)
        # For canny/seg controlnet, we pass the mask through the dense_dwpose_tensor argument.
        # Check if the backend is one of these and a mask was provided.
        is_mask_backend = self._backend in {"canny-controlnet", "seg-controlnet", "canny_controlnet", "seg_controlnet"}
        control_tensor = dense_dwpose_tensor
        if is_mask_backend and metadata_mask is not None:
            # metadata_mask is [H, W] or [H, W, 1]. Create [1, H, W] tensor.
            mask_np = np.asarray(metadata_mask, dtype=np.uint8)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            control_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self._device)

        if debug_dir is not None and frame_idx is not None and actor_identity is not None:
            self._current_debug_artifacts = self._strategy.get_debug_inputs(
                reference_crop_tensor=reference_crop_tensor,
                dense_dwpose_tensor=control_tensor,
            )
            bg_np = warped_background_frame.cpu().numpy()
            if len(bg_np.shape) == 3 and bg_np.shape[0] == 3:
                bg_np = np.transpose(bg_np, (1, 2, 0))
            self._current_debug_artifacts["02_warped_background.png"] = bg_np
        else:
            self._current_debug_artifacts = None

        generated_actor = self._strategy.generate(
            reference_crop_tensor=reference_crop_tensor,
            dense_dwpose_tensor=control_tensor,
            seed=self._seed,
            device=self._device,
            metadata_bbox=resolved_bbox,
        )
        
        if self._current_debug_artifacts is not None:
            actor_np = generated_actor.cpu().numpy()
            if len(actor_np.shape) == 3 and actor_np.shape[0] == 3:
                actor_np = np.transpose(actor_np, (1, 2, 0))
            self._current_debug_artifacts["03_generated_actor.png"] = actor_np

        return self._composite_actor_frame(
            generated_actor=generated_actor,
            dense_dwpose_tensor=dense_dwpose_tensor,
            warped_background_frame=warped_background_frame,
            actor_identity=actor_identity,
            metadata_mask=metadata_mask,
            metadata_bbox=metadata_bbox,
            debug_dir=debug_dir,
            frame_idx=frame_idx,
        )

    @gpu_bound
    def process_sequence(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frames: torch.Tensor,
        actor_identity: str | None = None,
        metadata_masks: list[np.ndarray | None] | None = None,
        metadata_bboxes: list[tuple[int, int, int, int] | None] | None = None,
        debug_dir: str | Path | None = None,
        global_frame_ids: list[int] | None = None,
    ) -> torch.Tensor:
        if not self.uses_temporal_pose_sequence():
            raise RuntimeError("process_sequence is only supported for temporal GenAI backends")
        if not hasattr(self._strategy, "generate_sequence"):
            raise RuntimeError("GenAI strategy does not support sequence generation")

        if dense_dwpose_tensor.ndim == 2:
            dense_dwpose_tensor = dense_dwpose_tensor.unsqueeze(0)
        if dense_dwpose_tensor.ndim != 3:
            raise ValueError(
                f"Expected dense pose sequence [Frames,18,3], got {tuple(dense_dwpose_tensor.shape)}"
            )
        if warped_background_frames.ndim != 4:
            raise ValueError(
                f"Expected background frames [Frames,C,H,W], got {tuple(warped_background_frames.shape)}"
            )

        frame_count = int(dense_dwpose_tensor.shape[0])
        if int(warped_background_frames.shape[0]) != frame_count:
            raise ValueError("Dense pose sequence length must match background frame count")

        # Shape: [Frames, Channels, Height, Width]
        generated_sequence = self._strategy.generate_sequence(
            reference_crop_tensor=reference_crop_tensor,
            dense_dwpose_tensor=dense_dwpose_tensor,
            seed=self._seed,
            device=self._device,
        )
        if generated_sequence.ndim != 4 or int(generated_sequence.shape[0]) != frame_count:
            raise ValueError(
                f"Generated sequence shape mismatch: expected {frame_count} frames, got {tuple(generated_sequence.shape)}"
            )

        out_frames: list[torch.Tensor] = []
        for frame_idx in range(frame_count):
            global_frame_id = global_frame_ids[frame_idx] if global_frame_ids is not None else None
            
            if debug_dir is not None and global_frame_id is not None and actor_identity is not None:
                self._current_debug_artifacts = self._strategy.get_debug_inputs(
                    reference_crop_tensor=reference_crop_tensor,
                    dense_dwpose_tensor=dense_dwpose_tensor[frame_idx],
                )
                bg_np = warped_background_frames[frame_idx].cpu().numpy()
                if len(bg_np.shape) == 3 and bg_np.shape[0] == 3:
                    bg_np = np.transpose(bg_np, (1, 2, 0))
                self._current_debug_artifacts["02_warped_background.png"] = bg_np
            else:
                self._current_debug_artifacts = None
                
            if self._current_debug_artifacts is not None:
                actor_np = generated_sequence[frame_idx].cpu().numpy()
                if len(actor_np.shape) == 3 and actor_np.shape[0] == 3:
                    actor_np = np.transpose(actor_np, (1, 2, 0))
                self._current_debug_artifacts["03_generated_actor.png"] = actor_np
                
            metadata_mask = None
            if metadata_masks is not None and frame_idx < len(metadata_masks):
                metadata_mask = metadata_masks[frame_idx]
            metadata_bbox = None
            if metadata_bboxes is not None and frame_idx < len(metadata_bboxes):
                metadata_bbox = metadata_bboxes[frame_idx]

            out_frames.append(
                self._composite_actor_frame(
                    generated_actor=generated_sequence[frame_idx],
                    dense_dwpose_tensor=dense_dwpose_tensor[frame_idx],
                    warped_background_frame=warped_background_frames[frame_idx],
                    actor_identity=actor_identity,
                    metadata_mask=metadata_mask,
                    metadata_bbox=metadata_bbox,
                    debug_dir=debug_dir,
                    frame_idx=global_frame_id,
                )
            )

        return torch.stack(out_frames, dim=0)

    def _composite_actor_frame(
        self,
        generated_actor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
        actor_identity: str | None = None,
        metadata_mask: np.ndarray | None = None,
        metadata_bbox: tuple[int, int, int, int] | None = None,
        debug_dir: str | Path | None = None,
        frame_idx: int | None = None,
    ) -> torch.Tensor:
        frame_np = self._to_frame_numpy(warped_background_frame)
        generated_np = self._to_crop_numpy(generated_actor)
        pose_np = self._to_pose_numpy(dense_dwpose_tensor)

        x1, y1, x2, y2 = self._resolve_target_bbox(
            pose_np=pose_np,
            frame_height=int(frame_np.shape[0]),
            frame_width=int(frame_np.shape[1]),
            metadata_bbox=metadata_bbox,
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
            pose_np=pose_np,
            target_bbox=(x1, y1, x2, y2),
        )
        if is_animate_anyone:
            alpha_mask = self._apply_temporal_alpha_smoothing(alpha_mask=alpha_mask, actor_identity=actor_identity)

        if _LOGGER.isEnabledFor(logging.DEBUG):
            if alpha_mask is None:
                _LOGGER.debug(
                    "GenAI alpha mask missing stage=%s actor=%s mode=%s backend=%s",
                    self._debug_stage,
                    actor_identity,
                    self._compositing_mask_mode,
                    self._backend,
                )
            else:
                nonzero = float(np.count_nonzero(alpha_mask > 0.01))
                total = float(alpha_mask.size)
                _LOGGER.debug(
                    "GenAI alpha mask stats stage=%s actor=%s mode=%s backend=%s nonzero=%.1f%%",
                    self._debug_stage,
                    actor_identity,
                    self._compositing_mask_mode,
                    self._backend,
                    100.0 * nonzero / max(1.0, total),
                )

        if alpha_mask is None or int(np.count_nonzero(alpha_mask > 0.01)) == 0:
            _LOGGER.debug(
                "GenAI compositor produced empty alpha stage=%s actor=%s mode=%s backend=%s",
                self._debug_stage,
                actor_identity,
                self._compositing_mask_mode,
                self._backend,
            )
            return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)
            
        if self._current_debug_artifacts is not None:
            alpha_out = np.asarray(alpha_mask * 255.0, dtype=np.uint8)
            if len(alpha_out.shape) == 3:
                alpha_out = alpha_out[:, :, 0]
            self._current_debug_artifacts["04_alpha_mask.png"] = alpha_out

        roi = frame_np[y1:y2, x1:x2]
        alpha_3 = np.asarray(alpha_mask[:, :, None], dtype=np.float32)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            try:
                delta = np.abs(actor_resized.astype(np.float32) - roi.astype(np.float32))
                _LOGGER.debug(
                    "GenAI actor vs ROI stage=%s actor=%s mean=%.2f max=%.2f",
                    self._debug_stage,
                    actor_identity,
                    float(delta.mean()),
                    float(delta.max()),
                )
            except Exception as exc:
                _LOGGER.debug(
                    "GenAI actor vs ROI stage=%s actor=%s diff unavailable: %s",
                    self._debug_stage,
                    actor_identity,
                    exc,
                )
        blended = actor_resized.astype(np.float32) * alpha_3 + roi.astype(np.float32) * (1.0 - alpha_3)
        frame_np[y1:y2, x1:x2] = np.asarray(np.clip(blended, 0.0, 255.0), dtype=np.uint8)
        
        if self._current_debug_artifacts is not None:
            self._current_debug_artifacts["05_composited_frame.png"] = frame_np
            export_compositor_artifacts(
                debug_dir=debug_dir,
                stage=self._debug_stage,
                frame_idx=frame_idx,
                actor_id=actor_identity,
                artifacts=self._current_debug_artifacts,
            )
            self._current_debug_artifacts = None
            
        return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

    def _resolve_target_bbox(
        self,
        pose_np: np.ndarray,
        frame_height: int,
        frame_width: int,
        metadata_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        is_mask_backend = self._backend in {"canny-controlnet", "seg-controlnet", "canny_controlnet", "seg_controlnet"}
        if (self._compositing_mask_mode == "metadata-source-mask" or is_mask_backend) and metadata_bbox is not None:
            x1, y1, x2, y2 = metadata_bbox
            clipped_x1 = max(0, min(frame_width - 1, int(x1)))
            clipped_y1 = max(0, min(frame_height - 1, int(y1)))
            clipped_x2 = max(clipped_x1 + 1, min(frame_width, int(x2)))
            clipped_y2 = max(clipped_y1 + 1, min(frame_height, int(y2)))
            return clipped_x1, clipped_y1, clipped_x2, clipped_y2

        if is_mask_backend and metadata_bbox is None:
            # Fallback if no metadata_bbox is provided but we need it for mask backend
            raise ValueError(f"metadata_bbox is required for {self._backend} but was not provided.")

        return self._estimate_bbox_from_pose(
            pose_np=pose_np,
            frame_height=frame_height,
            frame_width=frame_width,
        )

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
        pose_np: np.ndarray,
        target_bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        mode = self._compositing_mask_mode

        if mode == "pose-heuristic-mask":
            return self._alpha_from_pose_hull(
                pose_np=pose_np,
                target_bbox=target_bbox,
                target_w=int(actor_resized.shape[1]),
                target_h=int(actor_resized.shape[0]),
            )

        if mode == "metadata-source-mask":
            metadata_alpha = self._alpha_from_metadata_mask(
                metadata_mask=metadata_mask,
                target_w=int(actor_resized.shape[1]),
                target_h=int(actor_resized.shape[0]),
            )
            if metadata_alpha is not None:
                generated_alpha = self._segment_generated_actor(
                    actor_resized=actor_resized,
                    is_animate_anyone=is_animate_anyone,
                )
                if generated_alpha is None:
                    return metadata_alpha

                if generated_alpha.shape != metadata_alpha.shape:
                    generated_alpha = np.asarray(
                        cv2.resize(
                            np.asarray(generated_alpha, dtype=np.float32),
                            (int(metadata_alpha.shape[1]), int(metadata_alpha.shape[0])),
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        dtype=np.float32,
                    )

                return np.asarray(
                    np.clip(
                        np.asarray(metadata_alpha, dtype=np.float32)
                        * np.asarray(generated_alpha, dtype=np.float32),
                        0.0,
                        1.0,
                    ),
                    dtype=np.float32,
                )

        if mode == "postgen-seg-client":
            alpha = self._segment_generated_actor(actor_resized=actor_resized, is_animate_anyone=is_animate_anyone)
            if alpha is not None:
                return alpha
            metadata_alpha = self._alpha_from_metadata_mask(
                metadata_mask=metadata_mask,
                target_w=int(actor_resized.shape[1]),
                target_h=int(actor_resized.shape[0]),
            )
            if metadata_alpha is not None:
                return metadata_alpha

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

            weight_ref = _require_local_or_optin_weight(self._postgen_segmenter_model, allow_download=self.config.allow_auto_model_download if self.config else False)
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

    def _alpha_from_pose_hull(
        self, pose_np: np.ndarray, target_bbox: tuple[int, int, int, int], target_w: int, target_h: int
    ) -> np.ndarray | None:
        valid = pose_np[:, 2] >= self._confidence_threshold
        if not np.any(valid):
            return None
            
        x1, y1, x2, y2 = target_bbox
        
        # Shift keypoints to local crop coordinates
        xs = pose_np[valid, 0] - x1
        ys = pose_np[valid, 1] - y1
        
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        hull = cv2.convexHull(points).astype(np.int32)
        
        mask: np.ndarray = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, (255,))
        
        # Adaptive dilation based on target bounds
        dilation = float(self.config.pose_heuristic_mask_dilation if self.config and hasattr(self.config, "pose_heuristic_mask_dilation") else 0.15)
        kernel_x = max(3, int(target_w * dilation))
        kernel_y = max(3, int(target_h * dilation))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
        
        dilated_mask: np.ndarray = cv2.dilate(mask, kernel, iterations=1)
        
        # Return strict sharp edge mask 
        return dilated_mask.astype(np.float32) / 255.0

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
GenAICompositor = BaseCompositor
