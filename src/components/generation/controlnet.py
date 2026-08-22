"""ControlNet variants: canny, seg, pose, ip-adapter, multi.

Each variant is the same class with a different ``variant`` default in the
registry. Conditioning is read from named bundle fields; the compositor no
longer string-matches the backend name to decide whether the overloaded
parameter was a pose, a mask, or a tuple.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.components.generation.pose import letterbox_from_bbox, letterbox_image
from src.contracts.capabilities import (
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_POSE,
)
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

VARIANT_REQUIRES: dict[str, tuple[str, ...]] = {
    "canny": (CONDITION_CANNY, CONDITION_APPEARANCE),
    "seg": (CONDITION_MASK, CONDITION_APPEARANCE),
    "pose": (CONDITION_POSE, CONDITION_APPEARANCE),
    "ip-adapter": (CONDITION_APPEARANCE, CONDITION_POSE),
    "multi": (CONDITION_POSE, CONDITION_MASK, CONDITION_APPEARANCE),
}

_KNOWN = frozenset(VARIANT_REQUIRES)


class ControlNetGenerator(BaseFrameGenerator):
    """SD-ControlNet img2img, one class per variant via registry defaults."""

    def __init__(
        self,
        variant: str = "pose",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        strength: float = 0.65,
        guidance: float = 7.0,
        checkpoint: str | None = None,
        pipeline: Any = None,
    ) -> None:
        if variant not in _KNOWN:
            raise ValueError(
                f"Unknown ControlNet variant {variant!r}. Known: {', '.join(sorted(_KNOWN))}."
            )
        self.variant = variant
        self.required = VARIANT_REQUIRES[variant]
        self.width = width
        self.height = height
        self.steps = steps
        self.strength = strength
        self.guidance = guidance
        self.checkpoint = checkpoint
        self._pipeline = pipeline

    def prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        """Letterbox appearance and every declared condition onto one canvas.

        Public so the rescale fix is testable without loading diffusers.
        """
        canvas_width, canvas_height = self.canvas_size(params)
        appearance = as_hwc(conditioning.appearance)
        src_h, src_w = appearance.shape[:2]
        box = letterbox_from_bbox(
            conditioning.bbox, src_w, src_h, canvas_width, canvas_height
        )
        prepared: dict[str, Any] = {
            "appearance": letterbox_image(appearance, box),
            "letterbox": box,
        }
        for name in ("pose", "mask", "canny"):
            value = getattr(conditioning, name)
            if value is None:
                continue
            image = as_hwc(value)
            if image.shape[:2] != (src_h, src_w):
                image = _resize_to(image, src_w, src_h)
            prepared[name] = letterbox_image(image, box)
        return prepared

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        prepared = self.prepare(conditioning, params)
        pipeline = self._pipeline if self._pipeline is not None else self._load_pipeline(device)
        steps = params.steps if params.steps is not None else self.steps
        strength = params.strength if params.strength is not None else self.strength
        guidance = params.guidance_scale if params.guidance_scale is not None else self.guidance
        width, height = self.canvas_size(params)

        appearance = prepared["appearance"]
        init = params.init_image
        if init is not None:
            init_hwc = as_hwc(init)
        else:
            init_hwc = appearance

        control = self._control_image(prepared)
        output = pipeline(
            prompt="photorealistic tennis player, broadcast sports shot",
            image=init_hwc,
            control_image=control,
            height=height,
            width=width,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=guidance,
            generator_seed=seed,
            ip_adapter_image=appearance if self.variant == "ip-adapter" else None,
        )
        return as_chw(_coerce_output(output))

    def _control_image(self, prepared: dict[str, Any]) -> np.ndarray | list[np.ndarray]:
        if self.variant == "canny":
            return prepared["canny"]
        if self.variant == "seg":
            return prepared["mask"]
        if self.variant == "multi":
            return [prepared["pose"], prepared["mask"]]
        return prepared["pose"]

    def _load_pipeline(self, device: Device) -> Any:
        raise RuntimeError(
            f"{self.variant}-controlnet has no pipeline loaded. Pass a test double "
            f"as pipeline=... or a local checkpoint once weights are available. "
            f"Requested device={device!r}, checkpoint={self.checkpoint!r}."
        )


def _resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    interp = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def _coerce_output(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    images = getattr(output, "images", None)
    if images:
        return np.asarray(images[0])
    if isinstance(output, (tuple, list)) and output:
        return np.asarray(output[0])
    raise TypeError(f"ControlNet pipeline returned unusable output: {type(output)!r}.")
