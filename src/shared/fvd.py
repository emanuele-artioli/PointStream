"""Frechet Video Distance (FVD) — a spatiotemporal perceptual quality metric.

FVD is the video analogue of FID: instead of comparing single-image Inception
features, it compares clip-level features from a pretrained action-recognition
network (I3D) between a reference and a predicted/reconstructed video, and
reports the Frechet distance between the two Gaussian-fitted feature
distributions. Lower is better; 0 means the feature distributions are
identical (e.g. comparing a video against itself).

Backbone: I3D R50 ("Quo Vadis, Action Recognition?", Carreira & Zisserman),
pretrained on Kinetics-400 via the `pytorchvideo` model zoo (8-frame clips,
temporal stride 8 — the "8x8" checkpoint). The pretrained weights are
symlinked into `assets/weights/i3d_r50_kinetics.pyth` (see
`scripts/download_weights.py`) from the shared host model cache, per the
project's weights convention; never reference the host path in docs.

The math (Frechet distance over feature mean/covariance, matrix square root)
is architecture-agnostic and is factored out from I3D/video I/O so it can be
unit-tested cheaply with synthetic feature arrays (mock-first) without
loading the ~215MB checkpoint or decoding real video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import linalg

# I3D "8x8" pretraining convention: 8 frames sampled at stride 8 (i.e. one
# clip spans 64 raw source frames). Kept as module constants so the sampling
# behaviour used to build training-consistent clips is visible and testable.
I3D_CLIP_LEN = 8
I3D_SAMPLING_RATE = 8
I3D_FRAME_SIZE = 224

# Kinetics normalization constants used by the pytorchvideo model zoo
# (SlowFast/I3D family) — RGB order.
_KINETICS_MEAN = (0.45, 0.45, 0.45)
_KINETICS_STD = (0.225, 0.225, 0.225)

_FEATURE_DIM = 2048  # I3D R50 pre-classifier pooled feature width.

# Cache of loaded I3D feature extractors keyed by (weights_path, device) so a
# process computing FVD across many runs (e.g. benchmark_matrix sweeps) does
# not reload the checkpoint every call.
_MODEL_CACHE: dict[tuple[str, str], "I3DFeatureExtractor"] = {}


def default_weights_path() -> Path:
    """Resolve the project-relative path to the symlinked I3D checkpoint."""
    return Path(__file__).resolve().parents[2] / "assets" / "weights" / "i3d_r50_kinetics.pyth"


# ---------------------------------------------------------------------------
# Pure math: Frechet distance over feature statistics. No torch/model
# dependency, so this is cheap to unit test with synthetic arrays.
# ---------------------------------------------------------------------------


def compute_feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian to a set of feature vectors: return (mean, covariance).

    features: [NumClips, FeatureDim]
    """
    if features.ndim != 2:
        raise ValueError(f"expected features of shape [NumClips, FeatureDim], got {features.shape}")

    mu = np.mean(features, axis=0)
    if features.shape[0] > 1:
        sigma = np.cov(features, rowvar=False)
        # np.cov collapses to a 0-d array when FeatureDim == 1; keep it 2D.
        sigma = np.atleast_2d(sigma)
    else:
        # A single clip carries no covariance information; treat it as a
        # degenerate (zero-variance) Gaussian. FVD then reduces to the
        # Euclidean distance between the two mean feature vectors.
        dim = features.shape[1]
        sigma = np.zeros((dim, dim), dtype=np.float64)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between two multivariate Gaussians (the FID/FVD formula).

    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Follows the numerically-stabilized formulation standard in FID
    implementations: regularize with `eps * I` if the covariance product's
    matrix square root is singular, and drop any residual imaginary
    component introduced by floating-point error in `scipy.linalg.sqrtm`.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"mean vectors must have the same shape: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariance matrices must have the same shape: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    squared_distance = float(diff.dot(diff)) + float(trace_term)
    # Clamp tiny negative values caused by floating-point error near d^2 == 0.
    return float(np.sqrt(max(squared_distance, 0.0)))


# ---------------------------------------------------------------------------
# Clip sampling: turn a stack of decoded frames into I3D-ready clip tensors.
# ---------------------------------------------------------------------------


def sample_clip_frame_indices(
    num_frames: int,
    clip_len: int = I3D_CLIP_LEN,
    sampling_rate: int = I3D_SAMPLING_RATE,
) -> list[list[int]]:
    """Pick frame indices for one or more I3D clips out of `num_frames`.

    When there are enough frames for at least one full training-consistent
    window (`clip_len * sampling_rate` raw frames), split the video into
    non-overlapping such windows and take every `sampling_rate`-th frame from
    each (matching the "8x8" pretraining convention). Short videos (e.g. a
    few-second smoke-test clip) fall back to a single clip built by evenly
    subsampling whatever frames exist, so the metric stays computable — at
    the cost of a training-distribution mismatch and a degenerate
    (single-clip) covariance estimate; both are surfaced in `fvd_num_clips`.
    """
    if num_frames <= 0:
        return []

    window = clip_len * sampling_rate
    if num_frames >= window:
        num_clips = num_frames // window
        clips = []
        for clip_idx in range(num_clips):
            start = clip_idx * window
            clips.append([start + i * sampling_rate for i in range(clip_len)])
        return clips

    # Fallback: evenly subsample clip_len frames from the whole clip.
    if num_frames == 1:
        return [[0] * clip_len]
    indices = np.linspace(0, num_frames - 1, num=clip_len)
    return [np.round(indices).astype(int).tolist()]


def preprocess_frames_for_i3d(
    frames_bgr01: torch.Tensor,
    clip_len: int = I3D_CLIP_LEN,
    sampling_rate: int = I3D_SAMPLING_RATE,
    frame_size: int = I3D_FRAME_SIZE,
) -> torch.Tensor:
    """Build I3D-ready clip tensors from decoded video frames.

    frames_bgr01: [Frames, Channels(BGR), Height, Width], float32 in [0, 1]
        (this is exactly `video_io.decode_video_to_tensor(...).tensor`).
    Returns: [NumClips, Channels(RGB), ClipLen, FrameSize, FrameSize],
        Kinetics-normalized.
    """
    num_frames = int(frames_bgr01.shape[0])
    clip_indices = sample_clip_frame_indices(num_frames, clip_len=clip_len, sampling_rate=sampling_rate)
    if not clip_indices:
        raise ValueError("no frames available to build an I3D clip")

    # BGR -> RGB (Kinetics/I3D backbones are trained on RGB frames).
    frames_rgb01 = frames_bgr01.flip(dims=[1])  # Shape: [Frames, Channels(RGB), Height, Width]

    mean = torch.tensor(_KINETICS_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_KINETICS_STD, dtype=torch.float32).view(1, 3, 1, 1)

    clips = []
    for indices in clip_indices:
        clip_frames = frames_rgb01[indices]  # Shape: [ClipLen, Channels(RGB), Height, Width]
        clip_frames = torch.nn.functional.interpolate(
            clip_frames,
            size=(frame_size, frame_size),
            mode="bilinear",
            align_corners=False,
        )  # Shape: [ClipLen, Channels(RGB), FrameSize, FrameSize]
        clip_frames = (clip_frames - mean) / std
        clips.append(clip_frames.permute(1, 0, 2, 3))  # Shape: [Channels(RGB), ClipLen, FrameSize, FrameSize]

    return torch.stack(clips, dim=0)  # Shape: [NumClips, Channels(RGB), ClipLen, FrameSize, FrameSize]


# ---------------------------------------------------------------------------
# I3D feature extraction (real weights; heavy — kept behind lazy loading).
# ---------------------------------------------------------------------------


class I3DFeatureExtractor:
    """Loads I3D R50 (Kinetics-400) and extracts pre-classifier pooled features."""

    def __init__(self, weights_path: Path | None = None, device: str | None = None) -> None:
        self._weights_path = weights_path or default_weights_path()
        self._device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None

    @property
    def device(self) -> str:
        return self._device

    def _load_model(self) -> torch.nn.Module:  # pragma: no cover - requires real I3D weights, exercised by integration tests
        if self._model is not None:
            return self._model

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"I3D weights not found at '{self._weights_path}'. "
                "Run scripts/download_weights.py, or place/symlink the I3D R50 "
                "Kinetics-400 checkpoint (pytorchvideo 'i3d_r50') at that path."
            )

        # Imported lazily: pytorchvideo/fvcore/iopath are only needed when a
        # real FVD computation is requested, keeping the module importable
        # (e.g. for the pure-math unit tests) without those deps loaded.
        import pytorchvideo.models.hub as hub

        model = hub.i3d_r50(pretrained=False)
        checkpoint = torch.load(str(self._weights_path), map_location="cpu")
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

        # Strip the classification head down to the pooled 2048-d feature:
        # skip the Kinetics-400 linear projection so ResNetBasicHead just
        # pools spatiotemporally and returns pre-classifier features.
        classification_head = model.blocks[-1]
        classification_head.proj = torch.nn.Identity()

        model.eval()
        model.to(self._device)
        self._model = model
        return model

    @torch.no_grad()
    def extract_features(  # pragma: no cover - requires real I3D weights, exercised by integration tests
        self, clips: torch.Tensor, batch_size: int = 4
    ) -> np.ndarray:
        """clips: [NumClips, Channels(RGB), ClipLen, FrameSize, FrameSize] -> [NumClips, FeatureDim]."""
        model = self._load_model()
        features: list[np.ndarray] = []
        for start in range(0, clips.shape[0], batch_size):
            batch = clips[start : start + batch_size].to(self._device)
            batch_features = model(batch)  # Shape: [Batch, FeatureDim]
            features.append(batch_features.detach().cpu().numpy())
        return np.concatenate(features, axis=0) if features else np.zeros((0, _FEATURE_DIM), dtype=np.float32)


def get_cached_extractor(weights_path: Path | None = None, device: str | None = None) -> I3DFeatureExtractor:
    resolved_weights = str(weights_path or default_weights_path())
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_key = (resolved_weights, resolved_device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = I3DFeatureExtractor(weights_path=weights_path, device=resolved_device)
    return _MODEL_CACHE[cache_key]


# ---------------------------------------------------------------------------
# High-level orchestration: two decoded videos -> a single FVD score.
# ---------------------------------------------------------------------------


def compute_fvd_from_frames(  # pragma: no cover - orchestrates real I3D inference, exercised by integration tests
    reference_frames_bgr01: torch.Tensor,
    predicted_frames_bgr01: torch.Tensor,
    weights_path: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Compute FVD between two already-decoded frame stacks.

    Each input: [Frames, Channels(BGR), Height, Width], float32 in [0, 1].
    """
    reference_clips = preprocess_frames_for_i3d(reference_frames_bgr01)
    predicted_clips = preprocess_frames_for_i3d(predicted_frames_bgr01)

    extractor = get_cached_extractor(weights_path=weights_path, device=device)
    reference_features = extractor.extract_features(reference_clips)
    predicted_features = extractor.extract_features(predicted_clips)

    mu_ref, sigma_ref = compute_feature_statistics(reference_features)
    mu_pred, sigma_pred = compute_feature_statistics(predicted_features)
    score = frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)

    note = None
    min_clips = min(reference_features.shape[0], predicted_features.shape[0])
    if min_clips < 2:
        note = (
            "fewer than 2 clips available on at least one side; FVD's covariance "
            "term is degenerate and the score reduces to a feature-mean distance"
        )

    return {
        "fvd": score,
        "fvd_num_clips_reference": int(reference_features.shape[0]),
        "fvd_num_clips_predicted": int(predicted_features.shape[0]),
        "fvd_backbone": "i3d_r50_kinetics400",
        "note": note,
    }
