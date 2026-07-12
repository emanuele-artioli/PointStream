"""Unit tests for the pure-tensor math in src/shared/lpips_metric.py.

No VGG weights are loaded here (see test_lpips_metric_integration.py for the
real-backbone path) — these exercise perceptual_distance_from_features and
normalize_for_vgg directly with synthetic tensors, mirroring test_fvd.py's
split between pure math and real-model tests.
"""
from __future__ import annotations

import pytest
import torch

from src.shared.lpips_metric import (
    default_weights_path,
    normalize_for_vgg,
    perceptual_distance_from_features,
)


def test_perceptual_distance_zero_for_identical_features() -> None:
    feats = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    assert perceptual_distance_from_features(feats, feats) == pytest.approx(0.0, abs=1e-6)


def test_perceptual_distance_grows_with_feature_difference() -> None:
    feats_a = [torch.zeros(1, 3, 4, 4)]
    feats_b_small = [torch.full((1, 3, 4, 4), 0.1)]
    feats_b_large = [torch.full((1, 3, 4, 4), 1.0)]

    small = perceptual_distance_from_features(feats_a, feats_b_small)
    large = perceptual_distance_from_features(feats_a, feats_b_large)
    assert 0.0 < small < large


def test_perceptual_distance_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        perceptual_distance_from_features([torch.zeros(1, 3, 2, 2)], [])


def test_perceptual_distance_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no feature layers"):
        perceptual_distance_from_features([], [])


def test_perceptual_distance_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        perceptual_distance_from_features([torch.zeros(1, 3, 2, 2)], [torch.zeros(1, 3, 4, 4)])


def test_normalize_for_vgg_shape_preserved() -> None:
    frames = torch.rand(3, 3, 16, 16)
    normalized = normalize_for_vgg(frames)
    assert normalized.shape == frames.shape


def test_normalize_for_vgg_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match=r"\[N, 3, H, W\]"):
        normalize_for_vgg(torch.rand(3, 4, 16, 16))


def test_default_weights_path_points_at_assets_weights() -> None:
    path = default_weights_path()
    assert path.name == "vgg19-bn.pth"
    assert path.parent.name == "weights"
