"""Appearance representations: JPEG quality and downscale are not equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.appearance import REGISTRY as APPEARANCE
from src.components.appearance.compressed import CompressedImageAppearance, resolve_downscale
from src.components.appearance.embedding import ImageEmbeddingAppearance
from src.components.appearance.latent import DiffusionLatentAppearance
from src.contracts.capabilities import (
    APPEARANCE_COMPRESSED_IMAGE,
    APPEARANCE_DIFFUSION_LATENT,
    APPEARANCE_IMAGE_EMBEDDING,
)


def _ramp() -> np.ndarray:
    """A crop with both spatial structure and high-frequency detail."""
    y, x = np.mgrid[0:64, 0:64]
    r = (x * 4).astype(np.uint8)
    g = (y * 4).astype(np.uint8)
    b = ((x + y) % 256).astype(np.uint8)
    checker = np.zeros((64, 64), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    return np.stack([r, g, np.bitwise_or(b, checker)], axis=2)


def test_jpeg_quality_changes_bytes_without_changing_transmitted_size():
    crop = _ramp()
    encoder = CompressedImageAppearance()
    low, bytes_low = encoder.encode(crop, quality=20, downscale=1.0)
    high, bytes_high = encoder.encode(crop, quality=95, downscale=1.0)
    assert low.kind == APPEARANCE_COMPRESSED_IMAGE
    assert low.transmitted_size == high.transmitted_size == (64, 64)
    assert low.quality != high.quality
    assert bytes_low != bytes_high
    assert low.downscale == high.downscale == 1.0


def test_downscale_changes_transmitted_size_at_fixed_quality():
    crop = _ramp()
    encoder = CompressedImageAppearance()
    full, _ = encoder.encode(crop, quality=90, downscale=1.0)
    half, payload_half = encoder.encode(crop, quality=90, downscale=0.5)
    assert full.quality == half.quality == 90
    assert full.transmitted_size == (64, 64)
    assert half.transmitted_size == (32, 32)
    decoded_half = encoder.decode(payload_half)
    assert decoded_half.shape[0] == 32
    assert decoded_half.shape[1] == 32


def test_jpeg_and_downscale_are_independent_mechanisms():
    """Quantisation at full res is not the same crop as a sharp half-res JPEG."""
    crop = _ramp()
    encoder = CompressedImageAppearance()
    _, quantized = encoder.encode(crop, quality=10, downscale=1.0)
    _, half = encoder.encode(crop, quality=95, downscale=0.5)
    decoded_q = encoder.decode(quantized)
    decoded_half = encoder.decode(half)
    assert decoded_q.shape != decoded_half.shape
    assert quantized != half


def test_config_divisor_and_linear_factor_agree_on_half_resolution():
    assert resolve_downscale(2) == 0.5
    assert resolve_downscale(0.5) == 0.5
    assert resolve_downscale(1) == 1.0
    with pytest.raises(ValueError, match="downscale"):
        resolve_downscale(0.0)


def test_image_embedding_differs_for_different_crops():
    encoder = ImageEmbeddingAppearance()
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    bright = np.full((32, 32, 3), 200, dtype=np.uint8)
    desc_a, payload_a = encoder.encode(dark)
    desc_b, payload_b = encoder.encode(bright)
    assert desc_a.kind == APPEARANCE_IMAGE_EMBEDDING
    assert desc_a.dimensions == desc_b.dimensions
    assert payload_a != payload_b


def test_diffusion_latent_states_an_exact_grid_cost():
    encoder = DiffusionLatentAppearance()
    desc, payload = encoder.encode(_ramp())
    assert desc.kind == APPEARANCE_DIFFUSION_LATENT
    assert desc.channels == 4
    assert desc.height == 8
    assert desc.width == 8
    assert desc.cost().byte_count == 4 * 8 * 8 * 2
    assert len(payload) == desc.cost().byte_count


def test_registry_names_match_the_capability_vocabulary():
    for name in (
        APPEARANCE_COMPRESSED_IMAGE,
        APPEARANCE_DIFFUSION_LATENT,
        APPEARANCE_IMAGE_EMBEDDING,
    ):
        built = APPEARANCE.build(name)
        assert built.kind == name
