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
from src.pipeline.reconstruction.dispatch import dispatch
from src.pipeline.reconstruction.device import DeviceDecision, DevicePolicy


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


def dispatch_by_object_identity(
    generator: GeneratorRef,
    bundles: tuple[ConditioningBundle, ...] | list[ConditioningBundle],
    *,
    seed: int,
    params: GenerationParams | None = None,
    policy: DevicePolicy | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[DeviceDecision, ...]]:
    """Dispatch temporal bundles as separate stable-object sequences.

    A sequence generator must never receive frames from different players in a
    single appearance timeline. Per-frame generators retain their efficient
    batched path because their outputs have no cross-frame state.
    """
    if not bundles:
        return (), ()
    if not generator.supports_sequence():
        crops, decision = dispatch(
            generator, bundles, seed=seed, params=params, policy=policy
        )
        return crops, (decision,)
    groups: dict[str, list[int]] = {}
    for index, bundle in enumerate(bundles):
        if not bundle.object_id:
            raise ValueError(
                "temporal generation requires an explicit object_id for every bundle"
            )
        groups.setdefault(bundle.object_id, []).append(index)
    ordered: list[np.ndarray | None] = [None] * len(bundles)
    decisions: list[DeviceDecision] = []
    for object_id, indices in groups.items():
        indices.sort(
            key=lambda index: (
                bundles[index].frame_index if bundles[index].frame_index is not None else index,
                index,
            )
        )
        sequence = tuple(bundles[index] for index in indices)
        crops, decision = dispatch(
            generator, sequence, seed=seed, params=params, policy=policy
        )
        if len(crops) != len(indices):
            raise ValueError(
                f"temporal generator returned {len(crops)} crops for object {object_id!r} "
                f"with {len(indices)} requested frames"
            )
        decisions.append(decision)
        for index, crop in zip(indices, crops, strict=True):
            ordered[index] = crop
    if any(crop is None for crop in ordered):
        raise RuntimeError("object-grouped generation left an output placement unfilled")
    return tuple(crop for crop in ordered if crop is not None), tuple(decisions)


__all__ = ["RunnerGeneratorAdapter", "as_runner_ref", "dispatch_by_object_identity"]
