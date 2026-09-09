"""The corrective residual: coarseness spectrum, compute, apply.

The residual absorbs whatever disabled stages would have handled. Absent
reports unaided reconstruction quality. Lossless is a ceiling calibration.
"""

from src.pipeline.residual.codec import (
    TransmittedResidual,
    decode_residual_stream,
    encode_residual_to_bitstream,
)
from src.pipeline.residual.lossy import (
    OFFSET,
    block_activity_gate,
    decode_lossy,
    downscale_background,
    encode_lossy,
    invert_clipped_addition,
    subsample_chroma,
)
from src.pipeline.residual.signal import (
    ResidualPayload,
    ResidualResult,
    apply_residual,
    apply_signed,
    compute_residual,
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
    "OFFSET",
    "Coarseness",
    "ResidualPayload",
    "ResidualPoint",
    "ResidualResult",
    "ResidualVariant",
    "TransmittedResidual",
    "apply_residual",
    "apply_signed",
    "block_activity_gate",
    "coarseness_ladder",
    "compute_residual",
    "decode_lossy",
    "decode_residual_stream",
    "downscale_background",
    "encode_lossy",
    "encode_residual_to_bitstream",
    "infer_lossy_rung",
    "invert_clipped_addition",
    "l1_energy",
    "point_for",
    "signed_residual",
    "subsample_chroma",
    "variant_for",
]
