"""Reconstruction: background, placement, generative dispatch, quality.

The pipeline never knows which backend was chosen. A generator is injected
as a protocol plus its declared capabilities. Device/OOM fallback is one
policy. Quality is scored on every path, region-scoped when masks exist.
"""

from src.pipeline.reconstruction.background import (
    MODE_DELTA,
    MODE_FULL,
    MODE_NONE,
    BackgroundModelView,
    BackgroundResolver,
    apply_plate_delta,
    warp_plate,
)
from src.pipeline.reconstruction.compositor import Placement, composite_clip, composite_frame
from src.pipeline.reconstruction.device import DeviceDecision, DevicePolicy, is_out_of_memory
from src.pipeline.reconstruction.dispatch import GeneratorRef, dispatch, from_spec
from src.pipeline.reconstruction.quality import (
    Closeness,
    NumpyPsnrEvaluator,
    QualityReport,
    bit_identical,
    closeness,
    measure_symmetry,
    score,
)
from src.pipeline.reconstruction.reconstruct import (
    ObjectRequest,
    ReconstructionRequest,
    ReconstructionResult,
    reconstruct,
)

__all__ = [
    "BackgroundModelView",
    "BackgroundResolver",
    "Closeness",
    "DeviceDecision",
    "DevicePolicy",
    "GeneratorRef",
    "MODE_DELTA",
    "MODE_FULL",
    "MODE_NONE",
    "NumpyPsnrEvaluator",
    "ObjectRequest",
    "Placement",
    "QualityReport",
    "ReconstructionRequest",
    "ReconstructionResult",
    "apply_plate_delta",
    "bit_identical",
    "closeness",
    "composite_clip",
    "composite_frame",
    "dispatch",
    "from_spec",
    "is_out_of_memory",
    "measure_symmetry",
    "reconstruct",
    "score",
    "warp_plate",
]
