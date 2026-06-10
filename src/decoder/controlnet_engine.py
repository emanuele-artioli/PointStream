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
        
        self._cached_prompts: dict[int, str] = {}
        
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
        if ref_hash in getattr(self, "_cached_prompts", {}):
            return self._cached_prompts[ref_hash]
            
        if not hasattr(self, "_cached_prompts"):
            self._cached_prompts = {}

        processor, model = self._ensure_vlm(device)
        inputs = processor(reference_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        base_prompt = "photorealistic tennis player, broadcast sports shot"
        full_prompt = f"{caption}, {base_prompt}"
        _LOGGER.info(f"Generated ControlNet prompt via BLIP: {full_prompt}")
        
        self._cached_prompts[ref_hash] = full_prompt
        return full_prompt

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

        original_ref_image = Image.fromarray(cv2.cvtColor(reference_np, cv2.COLOR_BGR2RGB))
        prompt = self._generate_caption(original_ref_image, device)

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


class CannyControlNetStrategy(BaseGenAIStrategy):
    """ControlNet strategy utilizing Canny edges as the control condition."""

    def __init__(
        self,
        model_id: str = "assets/weights/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_canny",
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
            raise RuntimeError("diffusers is required for ControlNet strategy") from exc

        dtype = resolve_torch_dtype_for_device(
            device,
            default_cuda=torch.float16,
            allowed_cuda={torch.float16, torch.bfloat16, torch.float32},
            config_dtype=self.config.gpu_dtype if self.config and hasattr(self.config, "gpu_dtype") else None,
        )
        
        _LOGGER.info(f"Loading Canny ControlNet model {self._controlnet_id}...")
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

    def get_debug_inputs(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        artifacts = {}
        ref_np = _to_numpy_bgr(reference_crop_tensor)
        artifacts["00_reference_crop.png"] = ref_np
        
        # Save the mask/canny condition
        mask_np = dense_dwpose_tensor.detach().cpu().squeeze().numpy()
        if mask_np.ndim == 2:
            mask_uint8 = np.clip(mask_np, 0, 255).astype(np.uint8)
            artifacts["01_mask_condition.png"] = mask_uint8
            
        return artifacts

    def generate(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        seed: int,
        device: torch.device,
        metadata_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        # dense_dwpose_tensor actually contains the binary mask or canny edge image here!
        pipe = self._ensure_pipeline(device)

        reference_np = _to_numpy_bgr(reference_crop_tensor)

        # dense_dwpose_tensor is a [C, H, W] tensor containing the decoded mask image 
        # (which we passed in through the pose_tensor argument to reuse the interface).
        # Wait, if we use mask metadata, it doesn't get passed as dense_dwpose_tensor automatically!
        # The BaseGenAIStrategy.generate signature requires a dense_dwpose_tensor.
        # We will need to pass the mask into generate! 
        # Actually, let's just assume `dense_dwpose_tensor` contains the mask as [1, H, W].
        
        mask_np = dense_dwpose_tensor.detach().cpu().squeeze().numpy()
        if mask_np.ndim != 2:
            raise ValueError(f"CannyControlNet expects a 2D mask, got shape: {dense_dwpose_tensor.shape}")

        if metadata_bbox is not None:
            x1, y1, x2, y2 = metadata_bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            scale = float(max(self._width, self._height)) / max(bh, bw)
            scaled_h = int(bh * scale)
            scaled_w = int(bw * scale)
            offset_x = (self._width - scaled_w) // 2
            offset_y = (self._height - scaled_h) // 2
        else:
            bh, bw = int(reference_np.shape[0]), int(reference_np.shape[1])
            scale = float(max(self._width, self._height)) / max(bh, bw)
            scaled_h = int(bh * scale)
            scaled_w = int(bw * scale)
            offset_x = (self._width - scaled_w) // 2
            offset_y = (self._height - scaled_h) // 2

        # Resize mask condition to match ControlNet canvas
        mask_uint8 = np.clip(mask_np, 0, 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        
        canvas = np.zeros((self._height, self._width), dtype=np.uint8)
        canvas[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = mask_resized
        control_rgb = np.stack([canvas]*3, axis=-1)

        reference_rgb = cv2.cvtColor(reference_np, cv2.COLOR_BGR2RGB)
        if reference_rgb.shape[0] != scaled_h or reference_rgb.shape[1] != scaled_w:
            reference_rgb = cv2.resize(reference_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            
        padded_reference = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        padded_reference[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = reference_rgb
            
        init_image = Image.fromarray(padded_reference)
        control_image = Image.fromarray(control_rgb)

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


class SegControlNetStrategy(CannyControlNetStrategy):
    """ControlNet strategy utilizing segmentation masks as the control condition."""

    def __init__(
        self,
        model_id: str = "assets/weights/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_seg",
        config: Any = None,
    ) -> None:
        super().__init__(model_id=model_id, controlnet_id=controlnet_id, config=config)
