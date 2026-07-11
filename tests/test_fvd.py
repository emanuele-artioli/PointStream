from __future__ import annotations

import numpy as np
import pytest
import torch

from src.shared.fvd import (
    I3DFeatureExtractor,
    compute_feature_statistics,
    default_weights_path,
    frechet_distance,
    get_cached_extractor,
    preprocess_frames_for_i3d,
    sample_clip_frame_indices,
)


def test_frechet_distance_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(seed=0)
    features = rng.normal(size=(16, 32)).astype(np.float64)
    mu, sigma = compute_feature_statistics(features)

    distance = frechet_distance(mu, sigma, mu, sigma)

    assert distance == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_grows_with_mean_shift() -> None:
    rng = np.random.default_rng(seed=1)
    features_a = rng.normal(size=(16, 32)).astype(np.float64)
    features_b = features_a + 5.0  # Large, deterministic shift in feature space.

    mu_a, sigma_a = compute_feature_statistics(features_a)
    mu_b, sigma_b = compute_feature_statistics(features_b)

    distance_close = frechet_distance(mu_a, sigma_a, mu_a, sigma_a)
    distance_far = frechet_distance(mu_a, sigma_a, mu_b, sigma_b)

    assert distance_far > distance_close
    assert distance_far == pytest.approx(np.sqrt(32) * 5.0, rel=1e-3)


def test_default_weights_path_points_at_assets_weights() -> None:
    path = default_weights_path()

    assert path.name == "i3d_r50_kinetics.pyth"
    assert path.parent.name == "weights"
    assert path.parent.parent.name == "assets"


def test_compute_feature_statistics_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        compute_feature_statistics(np.zeros((4, 4, 4)))


def test_frechet_distance_rejects_mismatched_mean_shapes() -> None:
    with pytest.raises(ValueError):
        frechet_distance(np.zeros(4), np.eye(4), np.zeros(5), np.eye(5))


def test_frechet_distance_rejects_mismatched_covariance_shapes() -> None:
    with pytest.raises(ValueError):
        frechet_distance(np.zeros(4), np.eye(4), np.zeros(4), np.eye(5))


def test_get_cached_extractor_reuses_instance_per_key(tmp_path) -> None:
    weights_a = tmp_path / "a.pyth"
    weights_b = tmp_path / "b.pyth"

    first = get_cached_extractor(weights_path=weights_a, device="cpu")
    second = get_cached_extractor(weights_path=weights_a, device="cpu")
    third = get_cached_extractor(weights_path=weights_b, device="cpu")

    assert first is second
    assert first is not third
    assert isinstance(first, I3DFeatureExtractor)
    assert first.device == "cpu"


def test_frechet_distance_symmetric() -> None:
    rng = np.random.default_rng(seed=2)
    features_a = rng.normal(size=(10, 8)).astype(np.float64)
    features_b = rng.normal(loc=1.0, size=(10, 8)).astype(np.float64)
    mu_a, sigma_a = compute_feature_statistics(features_a)
    mu_b, sigma_b = compute_feature_statistics(features_b)

    forward = frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
    backward = frechet_distance(mu_b, sigma_b, mu_a, sigma_a)

    assert forward == pytest.approx(backward, rel=1e-6)


def test_compute_feature_statistics_single_clip_gives_zero_covariance() -> None:
    features = np.array([[1.0, 2.0, 3.0]])

    mu, sigma = compute_feature_statistics(features)

    assert mu.tolist() == [1.0, 2.0, 3.0]
    assert sigma.shape == (3, 3)
    assert np.allclose(sigma, 0.0)


def test_sample_clip_frame_indices_full_windows() -> None:
    # 128 frames, clip_len=8, sampling_rate=8 -> two 64-frame windows.
    indices = sample_clip_frame_indices(num_frames=128, clip_len=8, sampling_rate=8)

    assert len(indices) == 2
    assert indices[0] == [0, 8, 16, 24, 32, 40, 48, 56]
    assert indices[1] == [64, 72, 80, 88, 96, 104, 112, 120]


def test_sample_clip_frame_indices_short_video_falls_back_to_one_clip() -> None:
    # Fewer frames than one full clip_len*sampling_rate window (64).
    indices = sample_clip_frame_indices(num_frames=10, clip_len=8, sampling_rate=8)

    assert len(indices) == 1
    assert len(indices[0]) == 8
    assert max(indices[0]) < 10


def test_sample_clip_frame_indices_empty_video() -> None:
    assert sample_clip_frame_indices(num_frames=0) == []


def test_sample_clip_frame_indices_single_frame_repeats_it() -> None:
    indices = sample_clip_frame_indices(num_frames=1, clip_len=8, sampling_rate=8)

    assert indices == [[0] * 8]


def test_preprocess_frames_for_i3d_shape_and_normalization() -> None:
    num_frames, height, width = 10, 64, 96
    frames = torch.rand(num_frames, 3, height, width)  # Shape: [Frames, Channels(BGR), Height, Width]

    clips = preprocess_frames_for_i3d(frames, clip_len=8, sampling_rate=8, frame_size=32)

    # Shape: [NumClips, Channels(RGB), ClipLen, FrameSize, FrameSize]
    assert clips.shape == (1, 3, 8, 32, 32)
    assert clips.dtype == torch.float32


def test_preprocess_frames_for_i3d_bgr_to_rgb_channel_swap() -> None:
    # A single frame, all channels distinct, so we can check the swap directly.
    frames = torch.zeros(64, 3, 4, 4)
    frames[:, 0, :, :] = 0.1  # B
    frames[:, 1, :, :] = 0.5  # G
    frames[:, 2, :, :] = 0.9  # R

    clips = preprocess_frames_for_i3d(frames, clip_len=8, sampling_rate=8, frame_size=4)

    # After BGR->RGB swap and normalization, channel 0 should reflect the
    # original R value (0.9) rather than B (0.1).
    mean_r, std_r = 0.45, 0.225
    expected_channel0 = (0.9 - mean_r) / std_r
    assert torch.allclose(clips[0, 0], torch.full((8, 4, 4), expected_channel0), atol=1e-5)
