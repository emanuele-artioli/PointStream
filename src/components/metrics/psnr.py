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
        return _mean_finite(values)

    def score_masked(
        self, reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray
    ) -> float:
        """PSNR over True pixels of ``mask`` shaped ``(H, W)`` or ``(T, H, W)``."""
        return masked_psnr(reference, predicted, mask)


def masked_psnr(reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-frame PSNR restricted to a boolean mask. Identical region → ``inf``."""
    ref, pred = paired(reference, predicted)
    selected = _align_mask(mask, ref.shape)
    values = [
        _masked_frame_psnr(ref[index], pred[index], selected[index])
        for index in range(ref.shape[0])
    ]
    return _mean_finite(values)


def _frame_psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    mse = float(np.mean((reference - predicted) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((_PEAK**2) / mse)


def _masked_frame_psnr(
    reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> float:
    mse = float(np.mean((reference[mask] - predicted[mask]) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((_PEAK**2) / mse)


def _align_mask(mask: np.ndarray, clip_shape: tuple[int, ...]) -> np.ndarray:
    frames, height, width, _channels = clip_shape
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if array.shape != (height, width):
            raise ValueError(
                f"mask shape {array.shape} does not match frame {(height, width)}"
            )
        return np.broadcast_to(array, (frames, height, width))
    if array.shape != (frames, height, width):
        raise ValueError(
            f"mask shape {array.shape} does not match clip {(frames, height, width)}"
        )
    return array


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.inf
    return float(sum(finite) / len(finite))
