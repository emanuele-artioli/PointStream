"""Runner-specific adaptation of generation backends.

Generation components speak CHW arrays.  The reconstruction dispatch contract
uses HWC arrays, so this adapter belongs above the components layer rather than
in ``src.components.generation``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.generation._numpy import as_hwc
from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams
from src.pipeline.reconstruction.dispatch import GeneratorRef


class RunnerGeneratorAdapter:
    """Adapt a frame or sequence backend from CHW to runner HWC arrays."""

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
        output = self.backend.generate(conditioning, seed=seed, device=device, params=params)
        return as_hwc(output)

    def generate_sequence(
        self,
        conditioning: Any,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> tuple[np.ndarray, ...]:
        if hasattr(self.backend, "generate_sequence"):
            output = self.backend.generate_sequence(
                conditioning, seed=seed, device=device, params=params
            )
            return tuple(as_hwc(frame) for frame in output)
        return tuple(
            as_hwc(self.backend.generate(bundle, seed=seed, device=device, params=params))
            for bundle in conditioning
        )


def as_runner_ref(
    backend: Any,
    name: str = "injected",
    capabilities: frozenset[str] | None = None,
    requires: frozenset[str] | None = None,
) -> GeneratorRef:
    """Wrap a component backend in the runner's reconstruction contract."""
    backend_capabilities = getattr(backend, "capabilities", frozenset[str]())
    caps = set(capabilities if capabilities is not None else backend_capabilities)
    if hasattr(backend, "generate_sequence"):
        caps.add(CAP_TEMPORAL_SEQUENCE)
    backend_requires = getattr(backend, "required", frozenset[str]())
    raw_requires = requires if requires is not None else backend_requires
    return GeneratorRef(
        backend=RunnerGeneratorAdapter(backend),
        capabilities=frozenset(caps),
        requires=frozenset(raw_requires),
        name=name,
    )
