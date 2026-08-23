"""Fréchet Video Motion Distance — temporal coherence, not FVD appearance.

Published FVMD (Liu et al. 2024) tracks points with PIPs++, forms velocity and
acceleration histograms, and reports the Fréchet distance between those motion
features. This backend keeps that recipe and makes the tracker injectable:
tests mock it, the default uses Lucas–Kanade on a grid so a run without PIPs++
still measures motion rather than I3D appearance (the leftover FVD path).

A single reconstructed clip versus its reference is one sample each, so the
Gaussian collapses and the distance is Euclidean in motion-feature space.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg

from src.components.metrics.frames import paired, to_clip
from src.contracts.metrics import FVMD

Tracker = Callable[[np.ndarray], np.ndarray]

N_ANGLE_BINS = 8
N_MAGNITUDE_BINS = 9


class FvmdMetric:
    """Motion-feature Fréchet distance. Lower is better. 0 if motion matches."""

    name = FVMD.name

    def __init__(self, tracker: Tracker | None = None) -> None:
        self._tracker = tracker

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        if ref.shape[0] < 2:
            raise ValueError(
                "FVMD is a temporal metric and needs a sequence of at least two "
                f"frames; got T={ref.shape[0]}."
            )
        track = self._tracker or lk_grid_tracker
        features_ref = motion_feature(track(ref))
        features_pred = motion_feature(track(pred))
        stacked_ref = np.atleast_2d(features_ref)
        stacked_pred = np.atleast_2d(features_pred)
        mu_ref, sigma_ref = feature_statistics(stacked_ref)
        mu_pred, sigma_pred = feature_statistics(stacked_pred)
        return frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)


def motion_feature(trajectories: np.ndarray) -> np.ndarray:
    """Concatenate velocity and acceleration polar histograms.

    ``trajectories`` is ``(T, N, 2)`` pixel positions. T=2 yields a zero
    acceleration histogram rather than failing — velocity still distinguishes
    a static clip from a sliding one.
    """
    tracks = np.asarray(trajectories, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"expected trajectories (T, N, 2), got {tracks.shape}")
    if tracks.shape[0] < 2:
        raise ValueError("need at least two frames of trajectories")
    velocity = np.diff(tracks, axis=0)
    if tracks.shape[0] >= 3:
        acceleration = np.diff(velocity, axis=0)
    else:
        acceleration = np.zeros_like(velocity)
    return np.concatenate([_polar_histogram(velocity), _polar_histogram(acceleration)])


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian fit: ``(mean, covariance)`` on ``[N, D]`` feature rows."""
    if features.ndim != 2:
        raise ValueError(f"expected features [N, D], got {features.shape}")
    mean = np.mean(features, axis=0)
    if features.shape[0] > 1:
        covariance = np.atleast_2d(np.cov(features, rowvar=False))
    else:
        dim = features.shape[1]
        covariance = np.zeros((dim, dim), dtype=np.float64)
    return mean.astype(np.float64), covariance.astype(np.float64)


def frechet_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two Gaussians (sqrt of the FID formula)."""
    mu_a = np.atleast_1d(np.asarray(mean_a, dtype=np.float64))
    mu_b = np.atleast_1d(np.asarray(mean_b, dtype=np.float64))
    sigma_a = np.atleast_2d(np.asarray(cov_a, dtype=np.float64))
    sigma_b = np.atleast_2d(np.asarray(cov_b, dtype=np.float64))
    if mu_a.shape != mu_b.shape:
        raise ValueError(f"mean shape mismatch: {mu_a.shape} vs {mu_b.shape}")
    if sigma_a.shape != sigma_b.shape:
        raise ValueError(f"covariance shape mismatch: {sigma_a.shape} vs {sigma_b.shape}")

    delta = mu_a - mu_b
    if np.allclose(sigma_a, 0.0) and np.allclose(sigma_b, 0.0):
        return float(np.linalg.norm(delta))
    covmean, _ = linalg.sqrtm(sigma_a @ sigma_b, disp=False)
    if not np.isfinite(np.asarray(covmean)).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset) @ (sigma_b + offset))
    covmean_arr = np.asarray(covmean)
    if np.iscomplexobj(covmean_arr):
        covmean_arr = covmean_arr.real
    trace = float(np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean_arr))
    squared = float(delta.dot(delta)) + trace
    return float(np.sqrt(max(squared, 0.0)))


def lk_grid_tracker(clip: np.ndarray, grid_step: int = 4) -> np.ndarray:
    """Sparse Lucas–Kanade tracks on a regular grid. Default stand-in for PIPs++."""
    import cv2

    frames = np.clip(np.rint(to_clip(clip)), 0, 255).astype(np.uint8)
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    height, width = gray[0].shape
    step = max(1, grid_step)
    rows, cols = np.mgrid[step // 2 : height : step, step // 2 : width : step]
    points = np.stack([cols.ravel(), rows.ravel()], axis=-1).astype(np.float32)
    if points.size == 0:
        points = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
    path = [points.copy()]
    previous = gray[0]
    for frame in gray[1:]:
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, frame, points, points.copy())
        if nxt is None or status is None:
            path.append(points.copy())
            continue
        good = status.reshape(-1) == 1
        points = np.where(good[:, None], nxt, points)
        path.append(points.copy())
        previous = frame
    return np.stack(path, axis=0)


def _polar_histogram(vectors: np.ndarray) -> np.ndarray:
    flat = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    magnitude = np.linalg.norm(flat, axis=1)
    angle = np.arctan2(flat[:, 1], flat[:, 0])
    angle_bins = np.floor(((angle + np.pi) / (2.0 * np.pi)) * N_ANGLE_BINS).astype(int)
    angle_bins = np.clip(angle_bins, 0, N_ANGLE_BINS - 1)
    mag_scale = float(np.max(magnitude)) if magnitude.size else 1.0
    mag_scale = max(mag_scale, 1e-6)
    mag_bins = np.floor((magnitude / mag_scale) * N_MAGNITUDE_BINS).astype(int)
    mag_bins = np.clip(mag_bins, 0, N_MAGNITUDE_BINS - 1)
    hist = np.zeros((N_ANGLE_BINS, N_MAGNITUDE_BINS), dtype=np.float64)
    for angle_bin, mag_bin, weight in zip(angle_bins, mag_bins, magnitude):
        hist[angle_bin, mag_bin] += weight
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.ravel()
