"""Rebuild and verify the PointStream probe set.

The view is the source of truth; the manifest is walked off it. Schema
``pointstream.probe_set.v2`` uses track-local frame indices and records the
global offset so the mapping is recoverable.
"""

from __future__ import annotations

from experiments.probe_set.materialize import regenerate
from experiments.probe_set.schema import (
    COORDINATE_SYSTEM,
    HELD_OUT_VIDEOS,
    SCHEMA_ID,
    TRAINING_SPLIT_VIDEOS,
    ProbeSetError,
)
from experiments.probe_set.verify import collect_violations, verify

__all__ = [
    "COORDINATE_SYSTEM",
    "HELD_OUT_VIDEOS",
    "SCHEMA_ID",
    "TRAINING_SPLIT_VIDEOS",
    "ProbeSetError",
    "collect_violations",
    "regenerate",
    "verify",
]
