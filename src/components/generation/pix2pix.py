"""Pix2Pix UNet wrapper. Pose + appearance, per frame. Weights optional."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.components.generation.pose import letterbox_from_bbox, letterbox_image
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams


class Pix2PixGenerator(BaseFrameGenerator):
    """6-channel (pose RGB + appearance RGB) UNet, matching the training script."""

    required = (CONDITION_POSE, CONDITION_APPEARANCE)

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        checkpoint: str | None = None,
        model: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.checkpoint = checkpoint
        self._model = model

    def prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        canvas_width, canvas_height = self.canvas_size(params)
        appearance = as_hwc(conditioning.appearance)
        src_h, src_w = appearance.shape[:2]
        box = letterbox_from_bbox(
            conditioning.bbox, src_w, src_h, canvas_width, canvas_height
        )
        pose = as_hwc(conditioning.pose)
        if pose.shape[:2] != (src_h, src_w):
            import cv2

            pose = cv2.resize(pose, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        return {
            "appearance": letterbox_image(appearance, box),
            "pose": letterbox_image(pose, box),
            "letterbox": box,
        }

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        del seed
        prepared = self.prepare(conditioning, params)
        model = self._model if self._model is not None else self._load_model(device)
        pose = prepared["pose"]
        appearance = prepared["appearance"]
        stacked = np.concatenate([as_chw(pose), as_chw(appearance)], axis=0)
        output = model(stacked)
        return as_chw(output)

    def _load_model(self, device: Device) -> Any:
        raise RuntimeError(
            f"pix2pix has no model loaded. Pass model=... for tests, or a local "
            f"checkpoint once weights are available. device={device!r}, "
            f"checkpoint={self.checkpoint!r}."
        )
