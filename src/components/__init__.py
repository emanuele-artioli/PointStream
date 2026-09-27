"""Pluggable backends, loaded from cheap registry exports on demand.

The registry package must remain importable in dataset-specific environments
that do not install every model's optional dependencies. Accessing a named
registry imports only that axis; :func:`all_registries` explicitly loads them
all for the CLI and full application.
"""

from __future__ import annotations

from importlib import import_module
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.contracts.registry import Registry

_REGISTRY_MODULES = {
    "APPEARANCE": ("appearance", "appearance"),
    "BACKGROUND": ("background", "background"),
    "CODECS": ("codec", "codec"),
    "DETECTORS": ("detection", "detector"),
    "DOMAINS": ("domain", "domain"),
    "GENERATORS": ("generation", "generator"),
    "METRICS": ("metrics", "metric"),
    "MOTION": ("motion", "motion"),
    "POSE": ("pose", "pose"),
    "RIGID": ("rigid", "rigid"),
    "SCENE": ("scene", "scene"),
    "SEGMENTERS": ("segmentation", "segmenter"),
    "SELECTION": ("selection", "selection"),
    "TEMPORAL": ("temporal", "temporal"),
    "TRACKING": ("tracking", "tracking"),
    "TRANSPORT": ("transport", "transport"),
}

__all__ = [
    "APPEARANCE",
    "BACKGROUND",
    "CODECS",
    "DETECTORS",
    "DOMAINS",
    "GENERATORS",
    "METRICS",
    "MOTION",
    "POSE",
    "RIGID",
    "SCENE",
    "SEGMENTERS",
    "SELECTION",
    "TEMPORAL",
    "TRACKING",
    "TRANSPORT",
    "all_registries",
    "describe_all",
    "validate_config",
]


def __getattr__(name: str) -> object:
    target = _REGISTRY_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, _ = target
    value = getattr(import_module(f"src.components.{module_name}"), "REGISTRY")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def all_registries() -> dict[str, "Registry[object]"]:
    """Load each axis registry, keyed by the name used in error messages."""
    return {
        axis: getattr(sys.modules[__name__], exported)
        for exported, (_, axis) in _REGISTRY_MODULES.items()
    }


def describe_all() -> str:
    """Readable table of every backend on every axis."""
    return "\n\n".join(registry.describe() for registry in all_registries().values())


def validate_config(config: object) -> None:
    """Validate a config against all loaded registries."""
    from src.contracts.config import PointstreamConfig, validate_backends

    if not isinstance(config, PointstreamConfig):
        raise TypeError(f"expected PointstreamConfig, got {type(config).__name__}")
    validate_backends(
        config,
        generators=getattr(sys.modules[__name__], "GENERATORS"),
        registries=all_registries(),
    )
