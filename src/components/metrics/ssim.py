"""Structural similarity (Wang et al.), averaged over frames and channels."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.components.metrics.frames import paired
from src.contracts.metrics import SSIM

_K1 = 0.01
_K2 = 0.03
_PEAK = 255.0
_WINDOW = 11
_SIGMA = 1.5


class SsimMetric:
    """Mean SSIM in ``[0, 1]``. Identical frames score 1."""

    name = SSIM.name

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        values = [_frame_ssim(ref[index], pred[index]) for index in range(ref.shape[0])]
        return float(sum(values) / len(values))


def _frame_ssim(reference: np.ndarray, predicted: np.ndarray) -> float:
    channels = [
        _channel_ssim(reference[:, :, channel], predicted[:, :, channel])
        for channel in range(reference.shape[-1])
    ]
    return float(sum(channels) / len(channels))


def _channel_ssim(reference: np.ndarray, predicted: np.ndarray) -> float:
    c1 = (_K1 * _PEAK) ** 2
    c2 = (_K2 * _PEAK) ** 2
    if min(reference.shape[:2]) < _WINDOW:
        return _global_ssim(reference, predicted, c1, c2)
    return _windowed_ssim(reference, predicted, c1, c2)


def _global_ssim(reference: np.ndarray, predicted: np.ndarray, c1: float, c2: float) -> float:
    mu_x = float(reference.mean())
    mu_y = float(predicted.mean())
    var_x = float(reference.var())
    var_y = float(predicted.var())
    cov = float(((reference - mu_x) * (predicted - mu_y)).mean())
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


def _windowed_ssim(reference: np.ndarray, predicted: np.ndarray, c1: float, c2: float) -> float:
    mu_x = ndimage.gaussian_filter(reference, _SIGMA)
    mu_y = ndimage.gaussian_filter(predicted, _SIGMA)
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x2 = ndimage.gaussian_filter(reference**2, _SIGMA) - mu_x2
    sigma_y2 = ndimage.gaussian_filter(predicted**2, _SIGMA) - mu_y2
    sigma_xy = ndimage.gaussian_filter(reference * predicted, _SIGMA) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))
