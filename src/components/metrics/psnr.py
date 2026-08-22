"""Peak signal-to-noise ratio. The always-on floor."""

from __future__ import annotations

import math

import numpy as np

from src.components.metrics.frames import paired
from src.contracts.metrics import PSNR

_PEAK = 255.0


class PsnrMetric:
    """Mean per-frame PSNR in dB. Identical frames score ``inf``."""

    name = PSNR.name

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        values = [_frame_psnr(ref[index], pred[index]) for index in range(ref.shape[0])]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            return math.inf
        return float(sum(finite) / len(finite))


def _frame_psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    mse = float(np.mean((reference - predicted) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((_PEAK**2) / mse)
