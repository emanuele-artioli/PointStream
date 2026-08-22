"""Bit-identity versus closeness.

Deterministic stages (panorama warp, residual arithmetic) must match bit for
bit. Generative stages will not, even when they are working — asserting
identity on a sampler fails for reasons that have nothing to do with
correctness.
"""

from __future__ import annotations

import numpy as np

from src.components.metrics.frames import paired


def bit_identical(reference: np.ndarray, predicted: np.ndarray) -> bool:
    """True when every sample matches exactly."""
    ref, pred = paired(reference, predicted)
    return bool(np.array_equal(ref, pred))


def close(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    atol: float = 1.0,
    rtol: float = 0.0,
) -> bool:
    """True when frames agree within a tolerance. For generative stages."""
    ref, pred = paired(reference, predicted)
    return bool(np.allclose(ref, pred, atol=atol, rtol=rtol))
