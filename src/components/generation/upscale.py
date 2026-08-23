"""Upsample + refine. No diffusion. The cheap baseline every generative model must beat."""

from __future__ import annotations

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.contracts.capabilities import CONDITION_APPEARANCE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


class UpscaleRefineGenerator(BaseFrameGenerator):
    """Bicubic upsample followed by an unsharp refine.

    Deterministic, no weights. Applied even when the appearance is already at
    the target size, so the output is never a silent identity — a generative
    model that merely copies the reference would match identity and lose to
    this, not tie it.
    """

    required = (CONDITION_APPEARANCE,)

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        sharpen: float = 0.5,
        scale: int = 2,
    ) -> None:
        self.width = width
        self.height = height
        self.sharpen = sharpen
        self.scale = max(1, scale)

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        del seed, device
        if cv2 is None:
            raise RuntimeError("opencv is required for upscale-refine.")
        appearance = as_hwc(conditioning.appearance)
        if params.width is None and params.height is None:
            src_h, src_w = appearance.shape[:2]
            target_w = src_w * self.scale
            target_h = src_h * self.scale
        else:
            target_w, target_h = self.canvas_size(params)
        upsampled = cv2.resize(
            appearance, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        # Unsharp mask: output = image + amount * (image - blur). A constant
        # image is unchanged; any spatial content moves. That is the point.
        blur = cv2.GaussianBlur(upsampled, (0, 0), sigmaX=1.0)
        refined = cv2.addWeighted(upsampled, 1.0 + self.sharpen, blur, -self.sharpen, 0)
        return as_chw(np.clip(refined, 0, 255).astype(np.uint8))
