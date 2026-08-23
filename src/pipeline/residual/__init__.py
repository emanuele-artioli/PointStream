"""The corrective residual: coarseness spectrum, compute, apply.

The residual absorbs whatever disabled stages would have handled. Absent
reports unaided reconstruction quality. Lossless is a ceiling calibration.
"""

from src.pipeline.residual.signal import (
    ResidualPayload,
    ResidualResult,
    apply_residual,
    apply_signed,
    block_activity_gate,
    compute_residual,
    decode_lossy,
    downscale_background,
    encode_lossy,
    l1_energy,
    signed_residual,
)
from src.pipeline.residual.spectrum import (
    Coarseness,
    ResidualPoint,
    ResidualVariant,
    coarseness_ladder,
    infer_lossy_rung,
    point_for,
    variant_for,
)

__all__ = [
    "Coarseness",
    "ResidualPayload",
    "ResidualPoint",
    "ResidualResult",
    "ResidualVariant",
    "apply_residual",
    "apply_signed",
    "block_activity_gate",
    "coarseness_ladder",
    "compute_residual",
    "decode_lossy",
    "downscale_background",
    "encode_lossy",
    "infer_lossy_rung",
    "l1_energy",
    "point_for",
    "signed_residual",
    "variant_for",
]
