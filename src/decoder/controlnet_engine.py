from __future__ import annotations

import logging
from typing import Any
import torch
import cv2
import numpy as np
from PIL import Image

from src.decoder.genai_compositor import BaseGenAIStrategy, _to_numpy_bgr, _render_pose_condition
from src.shared.torch_dtype import resolve_torch_dtype_for_device

_LOGGER = logging.getLogger(__name__)


class MockCaptionControlNetStrategy(BaseGenAIStrategy):
    """Mock implementation returning deterministic dummy tensors for pipeline validation."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        # Validate inputs
        if reference_crop_tensor.ndim != 3 or reference_crop_tensor.shape[0] != 3:
            raise ValueError(f"Invalid reference crop shape: {tuple(reference_crop_tensor.shape)}")
        if dense_dwpose_tensor.ndim not in (2, 3) or dense_dwpose_tensor.shape[-1] != 3:
            raise ValueError(f"Invalid pose tensor shape: {tuple(dense_dwpose_tensor.shape)}")

        # Shape: [Channels, Height, Width]
        c, h, w = reference_crop_tensor.shape

        # Create deterministic dummy output based on seed
        torch.manual_seed(seed)
        dummy_tensor = torch.randint(0, 256, (c, h, w), dtype=torch.uint8, device=device)
        return dummy_tensor


class CaptionControlNetStrategy(BaseGenAIStrategy):
    """ControlNet strategy with BLIP auto-captioning for the reference image."""

    def __init__(
        self,
        model_id: str = "assets/weights/stable-diffusion-v1-5",
        controlnet_id: str = "assets/weights/control_v11p_sd15_openpose",
        vlm_id: str = "assets/weights/blip-image-captioning-base",
        config: Any = None,
    ) -> None:
        self._model_id = model_id
        self._controlnet_id = controlnet_id
        self._vlm_id = vlm_id
        self.config = config
        
        self._pipe: Any | None = None
        self._vlm_processor: Any | None = None
        self._vlm_model: Any | None = None
        
        self._cached_prompt: str | None = None
        self._cached_reference_hash: int | None = None
        
        self._width = int(config.controlnet_width) if config and hasattr(config, "controlnet_width") else 512
        self._height = int(config.controlnet_height) if config and hasattr(config, "controlnet_height") else 512
        self._steps = int(config.controlnet_steps) if config and hasattr(config, "controlnet_steps") else 20
        self._strength = float(config.controlnet_strength) if config and hasattr(config, "controlnet_strength") else 0.65
        self._cfg = float(config.controlnet_cfg) if config and hasattr(config, "controlnet_cfg") else 7.0

    def _ensure_vlm(self, device: torch.device) -> tuple[Any, Any]:
        if self._vlm_processor is not None and self._vlm_model is not None:
            return self._vlm_processor, self._vlm_model

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for BLIP VLM strategy") from exc

        _LOGGER.info(f"Loading VLM model {self._vlm_id}...")
        self._vlm_processor = BlipProcessor.from_pretrained(self._vlm_id)
        self._vlm_model = BlipForConditionalGeneration.from_pretrained(self._vlm_id).to(device)
        return self._vlm_processor, self._vlm_model

    def _ensure_pipeline(self, device: torch.device) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError("diffusers is required for ControlNet strategy") from exc

        dtype = resolve_torch_dtype_for_device(
            device,
            default_cuda=torch.float16,
            allowed_cuda={torch.float16, torch.bfloat16, torch.float32},
            config_dtype=self.config.gpu_dtype if self.config and hasattr(self.config, "gpu_dtype") else None,
        )
        
        _LOGGER.info(f"Loading ControlNet model {self._controlnet_id}...")
        controlnet = ControlNetModel.from_pretrained(self._controlnet_id, torch_dtype=dtype)
        
        _LOGGER.info(f"Loading Stable Diffusion model {self._model_id}...")
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

    def _generate_caption(self, reference_image: Image.Image, device: torch.device) -> str:
        # Cache captioning per run assuming reference stays identical to avoid redundant VLM calls
        ref_hash = hash(reference_image.tobytes())
        if self._cached_prompt is not None and self._cached_reference_hash == ref_hash:
            return self._cached_prompt

        processor, model = self._ensure_vlm(device)
        inputs = processor(reference_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        base_prompt = "photorealistic tennis player, broadcast sports shot"
        full_prompt = f"{caption}, {base_prompt}"
        _LOGGER.info(f"Generated ControlNet prompt via BLIP: {full_prompt}")
        
        self._cached_prompt = full_prompt
        self._cached_reference_hash = ref_hash
        return full_prompt

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        pipe = self._ensure_pipeline(device)

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

        prompt = self._generate_caption(init_image, device)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipe(
            prompt=prompt,
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
        
        # Shape: [Channels, Height, Width]
        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)


class IPAdapterControlNetStrategy(BaseGenAIStrategy):
    """ControlNet strategy with IP-Adapter visual prompting (txt2img)."""

    def __init__(
        self,
        model_id: str = "assets/weights/stable-diffusion-v1-5",
        controlnet_id: str = "assets/weights/control_v11p_sd15_openpose",
        config: Any = None,
    ) -> None:
        self._model_id = model_id
        self._controlnet_id = controlnet_id
        self.config = config
        
        self._pipe: Any | None = None
        
        self._width = int(config.controlnet_width) if config and hasattr(config, "controlnet_width") else 512
        self._height = int(config.controlnet_height) if config and hasattr(config, "controlnet_height") else 512
        self._steps = int(config.controlnet_steps) if config and hasattr(config, "controlnet_steps") else 20
        self._cfg = float(config.controlnet_cfg) if config and hasattr(config, "controlnet_cfg") else 7.0
        
        self._ip_adapter_repo = getattr(config, "ip_adapter_repo", "h94/IP-Adapter") if config else "h94/IP-Adapter"
        self._ip_adapter_subfolder = getattr(config, "ip_adapter_subfolder", "models") if config else "models"
        self._ip_adapter_weight = getattr(config, "ip_adapter_weight", "ip-adapter_sd15.bin") if config else "ip-adapter_sd15.bin"
        self._ip_adapter_scale = float(getattr(config, "ip_adapter_scale", 0.5)) if config else 0.5

    def _ensure_pipeline(self, device: torch.device) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError("diffusers is required for IP-Adapter ControlNet strategy") from exc

        dtype = resolve_torch_dtype_for_device(
            device,
            default_cuda=torch.float16,
            allowed_cuda={torch.float16, torch.bfloat16, torch.float32},
            config_dtype=self.config.gpu_dtype if self.config and hasattr(self.config, "gpu_dtype") else None,
        )
        
        _LOGGER.info(f"Loading ControlNet model {self._controlnet_id}...")
        controlnet = ControlNetModel.from_pretrained(self._controlnet_id, torch_dtype=dtype)
        
        _LOGGER.info(f"Loading Stable Diffusion model {self._model_id}...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self._model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe = pipe.to(device)
        
        _LOGGER.info(f"Loading IP-Adapter from {self._ip_adapter_repo} ({self._ip_adapter_weight})...")
        pipe.load_ip_adapter(
            self._ip_adapter_repo,
            subfolder=self._ip_adapter_subfolder,
            weight_name=self._ip_adapter_weight,
        )
        pipe.set_ip_adapter_scale(self._ip_adapter_scale)
        
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        pipe = self._ensure_pipeline(device)

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
            
        pose_rgb = pose_image
        init_image = Image.fromarray(padded_reference)
        control_image = Image.fromarray(pose_rgb)

        prompt = "photorealistic tennis player, broadcast sports shot"

        generator = torch.Generator(device=device).manual_seed(int(seed))
        output = pipe(
            prompt=prompt,
            image=control_image,
            ip_adapter_image=init_image,
            height=self._height,
            width=self._width,
            num_inference_steps=self._steps,
            guidance_scale=self._cfg,
            generator=generator,
        )
        generated_rgb = np.asarray(output.images[0], dtype=np.uint8)
        generated_cropped = generated_rgb[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w]
        generated_bgr = cv2.cvtColor(generated_cropped, cv2.COLOR_RGB2BGR)
        
        # Shape: [Channels, Height, Width]
        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)
