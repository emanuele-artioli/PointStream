"""Spade4Tennis inference engine for Pointstream GenAI backend.

Implements ``Spade4TennisStrategy`` which loads the SPADE-conditioned
ResNet-9 generator and produces player+racket synthesis from a skeleton
pose and a reference crop image.
"""
import logging
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.decoder.genai_compositor import (
    BaseGenAIStrategy,
    _render_pose_condition,
    _require_local_or_optin_weight,
    _to_numpy_bgr,
)
from src.shared.spade4tennis_arch import SPADEResNet9Generator

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPADE Normalization (inference-only copy, matches train_spade4tennis.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inference Strategy
# ---------------------------------------------------------------------------

class Spade4TennisStrategy(BaseGenAIStrategy):
    """Spade4Tennis GenAI backend for Pointstream."""

    def __init__(self, config: Any = None):
        self.config = config
        self._model: SPADEResNet9Generator | None = None
        self._width = int(config.controlnet_width) if config and hasattr(config, "controlnet_width") else 512
        self._height = int(config.controlnet_height) if config and hasattr(config, "controlnet_height") else 512

    def _ensure_model(self, device: torch.device) -> SPADEResNet9Generator:
        if self._model is not None:
            return self._model

        # Try lite first, then fall back to full
        try:
            model_path = _require_local_or_optin_weight("spade4tennis_lite_generator.pt", allow_download=False)
        except FileNotFoundError:
            model_path = _require_local_or_optin_weight("spade4tennis_full_generator.pt", allow_download=False)

        generator = SPADEResNet9Generator()
        state_dict = torch.load(model_path, map_location="cpu")
        generator.load_state_dict(state_dict)
        generator.to(device)
        generator.eval()
        self._model = generator

        param_count = sum(p.numel() for p in generator.parameters())
        _LOGGER.info(f"Loaded Spade4Tennis generator ({param_count/1e6:.1f}M params) from {model_path}")

        return self._model

    def get_debug_inputs(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Produce debug visualisations of the conditioning inputs."""
        artifacts: dict[str, np.ndarray] = {}
        ref_np = _to_numpy_bgr(reference_crop_tensor)
        artifacts["00_reference_crop.png"] = ref_np

        try:
            pose_tensor = dense_dwpose_tensor.clone()
            if pose_tensor.ndim == 3:
                pose_tensor = pose_tensor[-1]

            pose_np_raw = pose_tensor.cpu().numpy()  # Shape: [18, 3]
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
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a player+racket image from skeleton pose and reference."""
        model = self._ensure_model(device)

        # --- Compute target canvas and remap pose ---
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
            bh, bw = int(reference_crop_tensor.shape[1]), int(reference_crop_tensor.shape[2])
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

        # --- Render skeleton ---
        pose_image = _render_pose_condition(
            pose_tensor=pose_tensor,
            output_height=self._height,
            output_width=self._width,
        )  # Shape: [H, W, 3] uint8 RGB

        # Skeleton → tensor [-1, 1]
        skeleton_tensor = transforms.ToTensor()(pose_image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
        skeleton_tensor = (skeleton_tensor - 0.5) * 2.0  # Shape: [1, 3, H, W]

        # Reference → tensor [-1, 1]
        ref_image = transforms.ToPILImage()(reference_crop_tensor)
        ref_image = transforms.Resize(
            (self._height, self._width), transforms.InterpolationMode.BICUBIC
        )(ref_image)
        ref_tensor = transforms.ToTensor()(ref_image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
        ref_tensor = (ref_tensor - 0.5) * 2.0  # Shape: [1, 3, H, W]

        # --- Forward pass (separate skeleton and reference inputs for SPADE) ---
        with torch.no_grad():
            generated = model(skeleton_tensor, ref_tensor)  # Shape: [1, 3, H, W]

        # Post-process: [-1, 1] → uint8 RGB
        generated = (generated.squeeze(0) + 1.0) / 2.0  # Shape: [3, H, W]
        generated_rgb = (generated * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        generated_rgb = generated_rgb.transpose(1, 2, 0)  # Shape: [H, W, 3]

        # Crop back the valid region
        generated_cropped = generated_rgb[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w]
        generated_bgr = cv2.cvtColor(generated_cropped, cv2.COLOR_RGB2BGR)

        return torch.from_numpy(generated_bgr).permute(2, 0, 1).contiguous().to(torch.uint8)
