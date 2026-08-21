"""The contracts every other layer is written against.

This package is the shared context for the whole project: protocols, schemas,
capability declarations, the layer list, and config validation. It is the one
place where "what a component must satisfy" is written down, and CI checks it,
so no prose description of the architecture can drift from the code.

**It imports nothing heavy.** No torch, no cv2, no ffmpeg, and nothing from any
other layer. Validating a configuration — including checking that every named
backend exists and that the chosen combination is decodable — must be possible
without the runtime dependencies of the backends themselves installed.
"""

from __future__ import annotations

from src.contracts.codecs import CodecCapabilities, EncodeRequest, RateControl
from src.contracts.conditioning import (
    ConditioningBundle,
    ConditioningPlan,
    GenerationParams,
)
from src.contracts.config import PointstreamConfig, load, parse, validate
from src.contracts.domain import CameraMotion, DomainProfile, SalientClass
from src.contracts.errors import (
    CodecConstraintError,
    ConfigError,
    ConfigKeyError,
    ConfigValueError,
    ContractError,
    DuplicateBackendError,
    LayerViolationError,
    MissingConditioningError,
    UndecodableStreamError,
    UnknownBackendError,
    UnsupportedCapabilityError,
)
from src.contracts.keypoints import KeypointSchema
from src.contracts.lattice import StageLattice
from src.contracts.metrics import MetricSpec
from src.contracts.registry import BackendSpec, Registry

__all__ = [
    "BackendSpec",
    "CameraMotion",
    "CodecCapabilities",
    "CodecConstraintError",
    "ConditioningBundle",
    "ConditioningPlan",
    "ConfigError",
    "ConfigKeyError",
    "ConfigValueError",
    "ContractError",
    "DomainProfile",
    "DuplicateBackendError",
    "EncodeRequest",
    "GenerationParams",
    "KeypointSchema",
    "LayerViolationError",
    "MetricSpec",
    "MissingConditioningError",
    "PointstreamConfig",
    "RateControl",
    "Registry",
    "SalientClass",
    "StageLattice",
    "UndecodableStreamError",
    "UnknownBackendError",
    "UnsupportedCapabilityError",
    "load",
    "parse",
    "validate",
]
