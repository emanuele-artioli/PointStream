"""Animate-Anyone: temporal pose-to-video.

The shipped checkpoint was fine-tuned on scenes from a **single tennis match**.
Every score it produces is scoped to that match; the caveat travels with the
registry summary and with ``TENNIS_MATCH_FINETUNE_CAVEAT`` on the class so a
caller dumping numbers cannot miss it.

This wrapper makes the backend evaluable. A full retrain is a later decision.
``scripts/eval_checkpoint.py`` still has no ``animate-anyone`` entry in
``ARCH_CHOICES`` — that script is owned elsewhere; the required change is
reported with this stream, not made here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.components.generation.pose import letterbox_from_bbox, letterbox_image
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

TENNIS_MATCH_FINETUNE_CAVEAT = (
    "This checkpoint was fine-tuned on scenes from a single tennis match. "
    "Any score it posts is scoped to that match, not a general human model."
)


class AnimateAnyoneGenerator(BaseFrameGenerator):
    """Sequence generator. Temporal capability is declared, not inferred."""

    required = (CONDITION_POSE, CONDITION_APPEARANCE)
    caveat = TENNIS_MATCH_FINETUNE_CAVEAT

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        steps: int = 10,
        guidance: float = 7.5,
        checkpoint: str | None = None,
        runtime: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.checkpoint = checkpoint
        self._runtime = runtime

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        if not conditioning:
            raise ValueError("generate_sequence needs at least one ConditioningBundle.")
        for bundle in conditioning:
            bundle.require(*self.required)
            bundle.validate_shapes()
        if self._runtime is not None:
            output = self._runtime(
                list(conditioning), seed=seed, device=device, params=params
            )
            return tuple(as_chw(frame) for frame in output)
        return self._run_runtime(tuple(conditioning), seed=seed, device=device, params=params)

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        # Single-frame path still letterboxes through the shared geometry so a
        # per-frame call and a length-1 sequence cannot disagree about layout.
        prepared = self._prepare(conditioning, params)
        if self._runtime is not None:
            output = self._runtime(
                [conditioning], seed=seed, device=device, params=params, prepared=prepared
            )
            frame = output[0] if isinstance(output, (list, tuple)) else output
            return as_chw(frame)
        return self._run_runtime((conditioning,), seed=seed, device=device, params=params)[0]

    def _prepare(
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
            "caveat": TENNIS_MATCH_FINETUNE_CAVEAT,
        }

    def _run_runtime(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        del seed, device, params
        raise RuntimeError(
            "animate-anyone runtime is not loaded. "
            + TENNIS_MATCH_FINETUNE_CAVEAT
            + " Pass runtime=... for tests, or a local Moore-AnimateAnyone "
            "checkpoint once eval_checkpoint.py lists this arch. "
            f"Bundles={len(conditioning)}, checkpoint={self.checkpoint!r}."
        )
