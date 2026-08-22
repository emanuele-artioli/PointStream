"""Pluggable backends. One subpackage per axis, each with its own Registry.

Importing this package must stay cheap: registry tables hold import strings, not
classes. Heavy backends load only when ``Registry.build`` is called.

Per-axis packages are owned by Phase B workstreams. This module only re-exports
their ``REGISTRY`` objects so one command can list every backend. Do not add
implementations here.
"""

from __future__ import annotations

from src.components.appearance import REGISTRY as APPEARANCE
from src.components.background import REGISTRY as BACKGROUND
from src.components.codec import REGISTRY as CODECS
from src.components.detection import REGISTRY as DETECTORS
from src.components.domain import REGISTRY as DOMAINS
from src.components.generation import REGISTRY as GENERATORS
from src.components.metrics import REGISTRY as METRICS
from src.components.motion import REGISTRY as MOTION
from src.components.pose import REGISTRY as POSE
from src.components.rigid import REGISTRY as RIGID
from src.components.scene import REGISTRY as SCENE
from src.components.segmentation import REGISTRY as SEGMENTERS
from src.components.selection import REGISTRY as SELECTION
from src.components.temporal import REGISTRY as TEMPORAL
from src.components.tracking import REGISTRY as TRACKING
from src.components.transport import REGISTRY as TRANSPORT
from src.contracts.registry import Registry

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
]


def all_registries() -> dict[str, Registry[object]]:
    """Every axis registry, keyed by the axis name used in error messages."""
    return {
        "appearance": APPEARANCE,
        "background": BACKGROUND,
        "codec": CODECS,
        "detector": DETECTORS,
        "domain": DOMAINS,
        "generator": GENERATORS,
        "metric": METRICS,
        "motion": MOTION,
        "pose": POSE,
        "rigid": RIGID,
        "scene": SCENE,
        "segmenter": SEGMENTERS,
        "selection": SELECTION,
        "temporal": TEMPORAL,
        "tracking": TRACKING,
        "transport": TRANSPORT,
    }


def describe_all() -> str:
    """Readable table of every registered backend on every axis."""
    return "\n\n".join(registry.describe() for registry in all_registries().values())
