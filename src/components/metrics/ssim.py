"""Structural similarity (Wang et al.), averaged over frames and channels."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from src.components.metrics.frames import paired
from src.contracts.metrics import SSIM

_K1 = 0.01
_K2 = 0.03
_PEAK = 255.0
_WINDOW = 11
_SIGMA = 1.5

ScratchBuffers = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _allocate_buffers(shape: tuple[int, int]) -> ScratchBuffers:
    """Preallocate 4 reusable float64 buffers and fault memory pages eagerly."""
    bufs = (
        np.empty(shape, dtype=np.float64),
        np.empty(shape, dtype=np.float64),
        np.empty(shape, dtype=np.float64),
        np.empty(shape, dtype=np.float64),
    )
    for b in bufs:
        b.fill(0.0)
    return bufs


class _ThreadBuffers(threading.local):
    """Thread-local scratch buffers to ensure complete thread isolation."""

    def __init__(self) -> None:
        self.buffers: ScratchBuffers | None = None
        self.shape: tuple[int, int] | None = None

    def get(self, shape: tuple[int, int]) -> ScratchBuffers:
        if self.buffers is None or self.shape != shape:
            self.shape = shape
            self.buffers = _allocate_buffers(shape)
        return self.buffers


_THREAD_BUFFERS = _ThreadBuffers()


class SsimMetric:
    """Mean SSIM in ``[0, 1]``. Identical frames score 1."""

    name = SSIM.name

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        n_frames = int(ref.shape[0])
        h, w = int(ref.shape[1]), int(ref.shape[2])
        is_windowed = min(h, w) >= _WINDOW

        env_workers = int(os.environ.get("SSIM_THREADS", "0"))
        max_workers = env_workers if env_workers > 0 else min(8, n_frames, os.cpu_count() or 1)

        if not is_windowed or max_workers <= 1 or n_frames <= 1:
            buffers = _THREAD_BUFFERS.get((h, w)) if is_windowed else None
            values = [
                _frame_ssim(ref[index], pred[index], buffers=buffers)
                for index in range(n_frames)
            ]
            return float(sum(values) / len(values))

        def _worker_task(index: int) -> float:
            buffers = _THREAD_BUFFERS.get((h, w))
            return _frame_ssim(ref[index], pred[index], buffers=buffers)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            values = list(pool.map(_worker_task, range(n_frames)))
        return float(sum(values) / len(values))

    def score_masked(
        self, reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray
    ) -> float:
        """Global SSIM over True pixels of ``mask``. Not interchangeable with a crop."""
        return masked_ssim(reference, predicted, mask)


def masked_ssim(reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-frame global SSIM restricted to a boolean mask."""
    ref, pred = paired(reference, predicted)
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim == 2:
        selected = np.broadcast_to(selected, (ref.shape[0], *selected.shape))
    if selected.shape != ref.shape[:3]:
        raise ValueError(
            f"mask shape {np.asarray(mask).shape} does not match clip {ref.shape[:3]}"
        )
    values = [
        _frame_ssim_masked(ref[index], pred[index], selected[index])
        for index in range(ref.shape[0])
    ]
    return float(sum(values) / len(values))


def _frame_ssim(
    reference: np.ndarray,
    predicted: np.ndarray,
    buffers: ScratchBuffers | None = None,
) -> float:
    channels = [
        _channel_ssim(
            reference[:, :, channel],
            predicted[:, :, channel],
            buffers=buffers,
        )
        for channel in range(reference.shape[-1])
    ]
    return float(sum(channels) / len(channels))


def _channel_ssim(
    reference: np.ndarray,
    predicted: np.ndarray,
    buffers: ScratchBuffers | None = None,
) -> float:
    c1 = (_K1 * _PEAK) ** 2
    c2 = (_K2 * _PEAK) ** 2
    if min(reference.shape[:2]) < _WINDOW:
        return _global_ssim(reference, predicted, c1, c2)
    return _windowed_ssim(reference, predicted, c1, c2, buffers=buffers)


def _frame_ssim_masked(
    reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> float:
    c1 = (_K1 * _PEAK) ** 2
    c2 = (_K2 * _PEAK) ** 2
    channels = [
        _global_ssim(reference[:, :, channel][mask], predicted[:, :, channel][mask], c1, c2)
        for channel in range(reference.shape[-1])
    ]
    return float(sum(channels) / len(channels))


def _global_ssim(reference: np.ndarray, predicted: np.ndarray, c1: float, c2: float) -> float:
    mu_x = float(reference.mean())
    mu_y = float(predicted.mean())
    var_x = float(reference.var())
    var_y = float(predicted.var())
    cov = float(((reference - mu_x) * (predicted - mu_y)).mean())
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


def _windowed_ssim(
    reference: np.ndarray,
    predicted: np.ndarray,
    c1: float,
    c2: float,
    buffers: ScratchBuffers | None = None,
) -> float:
    if buffers is None:
        buf_a = np.empty(reference.shape, dtype=np.float64)
        buf_b = np.empty(reference.shape, dtype=np.float64)
        buf_c = np.empty(reference.shape, dtype=np.float64)
        buf_d = np.empty(reference.shape, dtype=np.float64)
    else:
        buf_a, buf_b, buf_c, buf_d = buffers

    ndimage.gaussian_filter(reference, _SIGMA, output=buf_a)
    ndimage.gaussian_filter(predicted, _SIGMA, output=buf_b)

    np.multiply(buf_a, buf_b, out=buf_c)
    buf_c *= 2.0
    buf_c += c1

    np.multiply(reference, predicted, out=buf_d)
    ndimage.gaussian_filter(buf_d, _SIGMA, output=buf_d)
    np.multiply(buf_a, buf_b, out=buf_a)
    buf_d -= buf_a
    buf_d *= 2.0
    buf_d += c2

    buf_c *= buf_d

    np.multiply(reference, reference, out=buf_d)
    ndimage.gaussian_filter(buf_d, _SIGMA, output=buf_d)

    np.multiply(predicted, predicted, out=buf_a)
    ndimage.gaussian_filter(buf_a, _SIGMA, output=buf_a)
    buf_d += buf_a

    ndimage.gaussian_filter(reference, _SIGMA, output=buf_a)
    np.multiply(buf_a, buf_a, out=buf_a)

    ndimage.gaussian_filter(predicted, _SIGMA, output=buf_b)
    np.multiply(buf_b, buf_b, out=buf_b)

    buf_d -= buf_a
    buf_d -= buf_b
    buf_d += c2

    buf_a += buf_b
    buf_a += c1

    buf_d *= buf_a
    buf_c /= buf_d
    return float(np.mean(buf_c))
