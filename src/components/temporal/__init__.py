"""Temporal-policy backends.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("temporal")

_TARGET = "src.components.temporal.policy:ConfigurableTemporalPolicy"

REGISTRY.register(
    BackendSpec(
        name="config",
        target=_TARGET,
        summary="Sparsity levels taken from TemporalConfig.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="none",
        target=_TARGET,
        aliases=("dense",),
        defaults={
            "metadata_sparsity": False,
            "generation_sparsity": False,
            "pipeline_sparsity": False,
        },
        summary="Every frame fully processed.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="metadata-sparsity",
        target=_TARGET,
        defaults={
            "metadata_sparsity": True,
            "generation_sparsity": False,
            "pipeline_sparsity": False,
        },
        summary="Transmit motion only at keyframes; interpolate between.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="generation-sparsity",
        target=_TARGET,
        defaults={
            "metadata_sparsity": False,
            "generation_sparsity": True,
            "pipeline_sparsity": False,
        },
        summary="Run the generator only at keyframes.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="pipeline-sparsity",
        target=_TARGET,
        defaults={
            "metadata_sparsity": False,
            "generation_sparsity": False,
            "pipeline_sparsity": True,
        },
        summary="Skip detection, segmentation and pose on low-motion frames.",
    )
)
