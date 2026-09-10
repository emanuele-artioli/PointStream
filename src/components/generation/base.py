"""Shared FrameGenerator surface: typed bundle, keyword-only extras."""

from __future__ import annotations

from typing import Any

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


class RunnerGeneratorAdapter:
    """Adapts a FrameGenerator / SequenceGenerator from CHW (components convention)
    to HWC (runner/dispatch convention)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.required = getattr(backend, "required", ())
        self.width = getattr(backend, "width", 512)
        self.height = getattr(backend, "height", 512)

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        from src.components.generation._numpy import as_hwc

        out = self.backend.generate(conditioning, seed=seed, device=device, params=params)
        return as_hwc(out)

    def generate_sequence(
        self,
        conditioning: Any,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Any:
        from src.components.generation._numpy import as_hwc

        if hasattr(self.backend, "generate_sequence"):
            output = self.backend.generate_sequence(
                conditioning, seed=seed, device=device, params=params
            )
            return tuple(as_hwc(f) for f in output)
        return tuple(
            as_hwc(self.backend.generate(b, seed=seed, device=device, params=params))
            for b in conditioning
        )


def as_runner_ref(
    backend: Any,
    name: str = "injected",
    capabilities: frozenset[str] | None = None,
    requires: frozenset[str] | None = None,
) -> Any:
    """Wrap any generator backend into a runner-compatible GeneratorRef."""
    from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE
    from src.pipeline.reconstruction.dispatch import GeneratorRef

    raw_caps: Any = (
        capabilities
        if capabilities is not None
        else getattr(backend, "capabilities", frozenset())
    )
    caps: set[str] = set(raw_caps)
    if hasattr(backend, "generate_sequence"):
        caps.add(CAP_TEMPORAL_SEQUENCE)
    raw_reqs: Any = (
        requires
        if requires is not None
        else getattr(backend, "required", frozenset())
    )
    reqs: frozenset[str] = frozenset(raw_reqs)
    adapter = RunnerGeneratorAdapter(backend)
    return GeneratorRef(
        backend=adapter,
        capabilities=frozenset(caps),
        requires=reqs,
        name=name,
    )

