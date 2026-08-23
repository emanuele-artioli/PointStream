"""Generative dispatch through an injected protocol, never a class or name.

The runner owns the registry and hands this layer a ``GeneratorRef``: a
callable that satisfies ``FrameGenerator`` plus the capabilities it declared.
Temporal vs per-frame is read from ``CAP_TEMPORAL_SEQUENCE`` on that
declaration. A backend that forgot to declare the capability is driven
per-frame even if it also happens to implement a sequence method; a backend
that declared it is driven as a sequence even if it is not the historical
temporal class. The pre-rewrite compositor selected the sequence path by
concrete class identity. This module does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE
from src.contracts.conditioning import (
    ConditioningBundle,
    FrameGenerator,
    GenerationParams,
    SequenceGenerator,
    require_sequence,
)
from src.contracts.errors import ConfigValueError, UnsupportedCapabilityError
from src.contracts.registry import BackendSpec
from src.pipeline.reconstruction.device import DeviceDecision, DevicePolicy


@dataclass(frozen=True)
class GeneratorRef:
    """An injected generator plus the capabilities the registry declared for it.

    ``name`` is for error messages only. Dispatch never reads it, never
    substring-matches it, and never branches on the backend's type.
    """

    backend: FrameGenerator
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    name: str = "injected"

    def supports_sequence(self) -> bool:
        """Whether the *declaration* says this backend generates sequences."""
        return CAP_TEMPORAL_SEQUENCE in self.capabilities


def from_spec(spec: BackendSpec[object], backend: FrameGenerator) -> GeneratorRef:
    """Build a ref from a registry spec. C3 calls this; reconstruction does not look up."""
    return GeneratorRef(
        backend=backend,
        capabilities=spec.capabilities,
        requires=spec.requires,
        name=spec.name,
    )


def dispatch(
    generator: GeneratorRef,
    bundles: Sequence[ConditioningBundle],
    *,
    seed: int,
    params: GenerationParams | None = None,
    spec: BackendSpec[object] | None = None,
    policy: DevicePolicy | None = None,
) -> tuple[tuple[np.ndarray, ...], DeviceDecision]:
    """Generate one crop per bundle, routing on declared capabilities.

    Raises:
        ConfigValueError: Empty bundle list — generation ran with nothing to draw.
        MissingConditioningError: A declared requirement is absent from a bundle.
        UnsupportedCapabilityError: Declared temporal but has no ``generate_sequence``.
    """
    if not bundles:
        raise ConfigValueError(
            "reconstruction.generation",
            "generation is enabled but no conditioning bundles were supplied. "
            "The generator would have nothing to draw; that work belongs in the residual, "
            "or the generation stage should be off.",
        )
    settings = params if params is not None else GenerationParams()
    device_policy = policy if policy is not None else DevicePolicy()

    for bundle in bundles:
        if generator.requires:
            bundle.require(*sorted(generator.requires))
        bundle.validate_shapes()

    def _run(device: str) -> tuple[np.ndarray, ...]:
        if generator.supports_sequence():
            return _as_sequence(
                _generate_sequence(
                    generator,
                    bundles,
                    seed=seed,
                    device=device,
                    params=settings,
                    spec=spec,
                )
            )
        return tuple(
            _as_frame(
                generator.backend.generate(
                    bundle, seed=seed, device=device, params=settings
                )
            )
            for bundle in bundles
        )

    crops, decision = device_policy.run(_run)
    if len(crops) != len(bundles):
        raise ValueError(
            f"generator {generator.name!r} returned {len(crops)} frames for "
            f"{len(bundles)} bundles."
        )
    return crops, decision


def _generate_sequence(
    generator: GeneratorRef,
    bundles: Sequence[ConditioningBundle],
    *,
    seed: int,
    device: str,
    params: GenerationParams,
    spec: BackendSpec[object] | None,
) -> Sequence[object]:
    """Sequence path. Capability was already checked; this verifies the method exists."""
    backend = generator.backend
    if spec is not None:
        require_sequence(spec, backend)
    if not isinstance(backend, SequenceGenerator):
        raise UnsupportedCapabilityError(
            CAP_TEMPORAL_SEQUENCE,
            f"{generator.name} (declares it, but has no generate_sequence method)",
            sorted(generator.capabilities),
        )
    return backend.generate_sequence(bundles, seed=seed, device=device, params=params)


def _as_frame(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"generator must return a frame (H, W, 3); got shape {array.shape}."
        )
    return np.asarray(array, dtype=np.uint8)


def _as_sequence(values: Sequence[object]) -> tuple[np.ndarray, ...]:
    return tuple(_as_frame(item) for item in values)
