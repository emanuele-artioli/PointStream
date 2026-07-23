"""The generative strategies a run can select.

Selected by name string from config, so the mapping between a config
value and the class chosen here is part of the config contract."""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Any
import cv2
import numpy as np
import torch
from src.shared.torch_dtype import resolve_torch_dtype_for_device
from src.decoder.compositing.pose_render import _render_pose_condition, _to_numpy_bgr
_LOGGER = logging.getLogger(__name__)


class BaseGenAIStrategy(ABC):
    """Strategy interface for pluggable GenAI actor generation backends."""

    @abstractmethod
    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: Any,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
        init_image_override: Any = None,
        strength_override: float | None = None,
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
        model_id: str = "assets/weights/stable-diffusion-v1-5",
        controlnet_id: str = "assets/weights/control_v11p_sd15_openpose",
        config: Any = None,
    ) -> None:
        self.config = config
        self._model_id = model_id
        self._controlnet_id = getattr(config, "controlnet_id", controlnet_id) if config else controlnet_id
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
        init_image_override: Any = None,
        strength_override: float | None = None,
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
        if init_image_override is not None:
            init_image = init_image_override
        else:
            init_image = Image.fromarray(padded_reference)
            
        control_image = Image.fromarray(pose_rgb)

        strength = strength_override if strength_override is not None else self._strength

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipe(
            prompt="photorealistic tennis player, broadcast sports shot",
            image=init_image,
            control_image=control_image,
            height=self._height,
            width=self._width,
            num_inference_steps=self._steps,
            strength=strength,
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
        init_image_override: Any = None,
        strength_override: float | None = None,
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
