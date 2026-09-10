"""Shared FrameGenerator surface: typed bundle, keyword-only extras."""

from __future__ import annotations

import numpy as np

from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams


class BaseFrameGenerator:
    """One generate() shape for every backend.

    ``conditioning`` is the only positional argument. Seed, device and params
    are keyword-only so a mask cannot land in a slot named for a pose.
    Subclasses set ``required`` to the conditioning kinds they declared in the
    registry and implement ``_generate``.
    """

    required: tuple[str, ...] = ()
    width: int = 512
    height: int = 512

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        conditioning.require(*self.required)
        conditioning.validate_shapes()
        return self._generate(conditioning, seed=seed, device=device, params=params)

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        raise NotImplementedError

    def canvas_size(self, params: GenerationParams) -> tuple[int, int]:
        width = params.width if params.width is not None else self.width
        height = params.height if params.height is not None else self.height
        return width, height
