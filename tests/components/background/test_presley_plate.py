"""Tests for Presley-style compact background plate encoding and restoration."""

import numpy as np
import pytest

from src.components.background.presley_plate import (
    PresleyPlateConfig,
    PresleyPlateDecoder,
    PresleyPlateEncoder,
)
from src.components.background.scale import unpack_header
from src.components.background.types import BackgroundArtifact


def _make_synthetic_tennis_court(height: int = 2160, width: int = 3840) -> np.ndarray:
    """Generate synthetic 4K court: green turf, blue court, white lines, sensor noise."""
    court = np.zeros((height, width, 3), dtype=np.uint8)
    # Green outer turf (BGR: [34, 139, 34])
    court[:, :] = [34, 139, 34]
    # Blue inner court (BGR: [180, 105, 30])
    margin_y, margin_x = height // 6, width // 6
    court[margin_y : height - margin_y, margin_x : width - margin_x] = [180, 105, 30]
    # White baseline (BGR: [255, 255, 255])
    court[margin_y : margin_y + 12, margin_x : width - margin_x] = [255, 255, 255]
    # White center service line
    mid_x = width // 2
    court[margin_y : height - margin_y, mid_x - 6 : mid_x + 6] = [255, 255, 255]
    # Add subtle sensor noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, court.shape).astype(np.int16)
    noisy = np.clip(court.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# Group 1: Behaviour Tests
def test_default_config_initialization():
    config = PresleyPlateConfig()
    assert config.scale == 0.5
    assert config.filter_type == "bilateral"
    assert config.codec == "webp"
    assert config.quality == 40


def test_pre_filter_bilateral_smooths_noise_and_preserves_edges():
    encoder = PresleyPlateEncoder(PresleyPlateConfig(filter_type="bilateral"))
    court = _make_synthetic_tennis_court(360, 640)
    filtered = encoder.pre_filter(court)

    assert filtered.shape == court.shape
    assert filtered.dtype == np.uint8

    # Edge preservation: line center should remain high intensity
    mid_x = 320
    assert filtered[100, mid_x, 0] > 220


def test_downsample_and_geometry_header_packed():
    encoder = PresleyPlateEncoder(PresleyPlateConfig(scale=0.5))
    plate = _make_synthetic_tennis_court(1080, 1920)
    artifact = encoder.encode(plate, scene_id="scene_001", chunk_id="chunk_0")

    assert isinstance(artifact, BackgroundArtifact)
    assert artifact.method == "presley-compact"
    assert artifact.geometry_header != b""

    header = unpack_header(artifact.geometry_header)
    assert header.original_width == 1920
    assert header.original_height == 1080
    assert header.coded_width == 960
    assert header.coded_height == 540
    assert header.scale == 0.5


def test_encode_decode_roundtrip_restores_original_dimensions():
    encoder = PresleyPlateEncoder(PresleyPlateConfig(scale=0.5, quality=40))
    decoder = PresleyPlateDecoder()

    orig_h, orig_w = 720, 1280
    plate = _make_synthetic_tennis_court(orig_h, orig_w)
    artifact = encoder.encode(plate)

    restored = decoder.decode(artifact)
    assert restored.shape == (orig_h, orig_w, 3)
    assert restored.dtype == np.uint8

    # Check that restored court has good fidelity (PSNR >= 28 dB)
    mse = np.mean((plate.astype(float) - restored.astype(float)) ** 2)
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-10))
    assert psnr >= 28.0


def test_wire_budget_under_six_kilobytes():
    encoder = PresleyPlateEncoder(PresleyPlateConfig(scale=0.5, quality=35))
    # Test on full 4K plate
    plate = _make_synthetic_tennis_court(2160, 3840)
    artifact = encoder.encode(plate)

    # Invariant: wire budget <= 6.0 kB (6144 bytes)
    assert len(artifact.payload) <= 6144, f"Payload {len(artifact.payload)} exceeds 6144 bytes"


# Group 2: Plausible Misuse Tests
def test_unsupported_scale_raises_value_error():
    with pytest.raises(ValueError, match="scale"):
        PresleyPlateEncoder(PresleyPlateConfig(scale=0.75))


def test_unsupported_filter_raises_value_error():
    with pytest.raises(ValueError, match="filter"):
        PresleyPlateEncoder(PresleyPlateConfig(filter_type="magic"))


def test_unsupported_codec_raises_value_error():
    with pytest.raises(ValueError, match="codec"):
        PresleyPlateEncoder(PresleyPlateConfig(codec="mp3"))


def test_decode_corrupt_payload_raises_error():
    decoder = PresleyPlateDecoder()
    bad_artifact = BackgroundArtifact(
        method="presley-compact",
        codec="webp",
        codec_id="webp:q40",
        mode="full",
        payload=b"not a valid image bitstream",
        geometry_header=b"",
    )
    with pytest.raises(ValueError):
        decoder.decode(bad_artifact)


def test_decode_invalid_geometry_header_raises_error():
    encoder = PresleyPlateEncoder(PresleyPlateConfig(scale=0.5))
    decoder = PresleyPlateDecoder()
    plate = _make_synthetic_tennis_court(360, 640)
    artifact = encoder.encode(plate)

    corrupted_artifact = BackgroundArtifact(
        method=artifact.method,
        codec=artifact.codec,
        codec_id=artifact.codec_id,
        mode=artifact.mode,
        payload=artifact.payload,
        geometry_header=b"garbage_header",
    )
    with pytest.raises(Exception):
        decoder.decode(corrupted_artifact)
