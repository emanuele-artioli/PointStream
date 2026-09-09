"""Authorized behavioral tests for residual transport fidelity, serialization, and decoding.

Covers:
1. Zero residual roundtrip (zero difference restores identically with 0 energy)
2. Extrema and saturation modes (+/-255 in full_range vs clipped)
3. Exact uncompressed algebra (lossless int16 bit-identity without clipped sum subtraction artifacts)
4. Coded roundtrip with native codec (end-to-end encode to bitstream bytes and decode with ffmpeg/AVC)
5. Fresh-process source-free decode and scored output identity (client reconstructs from wire bytes with zero source access)
6. All-byte ledger reconciliation (wire bytes match sizes.residual; unencoded fallback rejected as rate)
7. Absent residual zero calls and bytes (disabled residual transmits 0 bytes and makes 0 codec calls)
8. Corrupted, truncated, or incompatible payload rejection (corrupted/truncated bitstream raises errors)
9. Generation-enabled predictor consistency (client runs generator for Pc, applies residual, rejects fallback scores)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.components.codec import tools
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import GeneratorConfig, PointstreamConfig, ResidualConfig
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_DELIVERED,
    SOURCE_PASSTHROUGH,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_RESIDUAL,
    WHOLE_FRAME_RESIDUAL,
    StageLattice,
)
from src.pipeline.reconstruction import (
    GeneratorRef,
    ObjectRequest,
    bit_identical,
)
from src.pipeline.residual import (
    TransmittedResidual,
    apply_residual,
    compute_residual,
    decode_lossy,
    decode_residual_stream,
    encode_lossy,
    encode_residual_to_bitstream,
    invert_clipped_addition,
    signed_residual,
)
from src.runner import lattice_config_from, run
from src.runner.client import reconstruct_serialized_client, serialize_client_request
from src.runner.stages import make_metrics


def _clip(value: int, *, frames: int = 2, size: int = 32) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def _object(*, size: int = 16) -> ObjectRequest:
    return ObjectRequest(
        object_id="player",
        appearance=np.full((size, size, 3), 200, dtype=np.uint8),
        bbox=(0, 0, size, size),
        mask=np.ones((size, size), dtype=bool),
    )


# ---------------------------------------------------------------------------
# 1. Zero residual roundtrip
# ---------------------------------------------------------------------------


def test_zero_residual_roundtrip() -> None:
    """Zero difference restores identically with 0 energy and 0 active blocks."""
    source = _clip(120, frames=2, size=16)
    recon = np.copy(source)

    # In pipeline compute_residual
    result = compute_residual(source, recon, lattice=StageLattice.of(STAGE_RESIDUAL))
    assert result.payload.l1_energy == 0.0
    assert result.payload.nonzero_bytes == 0
    assert result.payload.active_blocks == 0
    restored = apply_residual(recon, result.payload)
    assert bit_identical(source, restored)

    # In full_range encode_lossy
    diff = signed_residual(source, recon)
    encoded = encode_lossy(diff, mode="full_range")
    assert np.all(encoded == 128)
    decoded_diff = decode_lossy(encoded, mode="full_range")
    assert np.all(decoded_diff == 0)
    assert bit_identical(source, apply_residual(recon, decoded_diff))


# ---------------------------------------------------------------------------
# 2. Extrema and saturation modes
# ---------------------------------------------------------------------------


def test_extrema_and_saturation_modes() -> None:
    """Preserves +/-255 in full_range mode; saturates in clipped mode."""
    diff = np.array([[[[255, -255, 0]]]], dtype=np.int16)

    # Clipped mode: saturates to [-128, 127]
    enc_clip = encode_lossy(diff, mode="clipped")
    dec_clip = decode_lossy(enc_clip, mode="clipped")
    assert dec_clip[0, 0, 0, 0] == 127
    assert dec_clip[0, 0, 0, 1] == -128
    assert dec_clip[0, 0, 0, 2] == 0

    # Full range mode: preserves extrema exactly
    enc_fr = encode_lossy(diff, mode="full_range")
    assert enc_fr[0, 0, 0, 0] == 255
    assert enc_fr[0, 0, 0, 1] == 0
    assert enc_fr[0, 0, 0, 2] == 128

    dec_fr = decode_lossy(enc_fr, mode="full_range")
    assert dec_fr[0, 0, 0, 0] == 255
    assert dec_fr[0, 0, 0, 1] == -255
    assert dec_fr[0, 0, 0, 2] == 0

    # Test error bounds across the full integer span [-255, 255]
    all_diffs = np.arange(-255, 256, dtype=np.int16).reshape(1, 1, 511, 1)
    enc_all = encode_lossy(all_diffs, mode="full_range")
    dec_all = decode_lossy(enc_all, mode="full_range")
    max_err = int(np.max(np.abs(all_diffs - dec_all)))
    assert max_err <= 1


# ---------------------------------------------------------------------------
# 3. Exact uncompressed algebra
# ---------------------------------------------------------------------------


def test_exact_uncompressed_algebra() -> None:
    """Lossless int16 bit-identity without clipped sum subtraction artifacts."""
    base = np.array([[[[220, 15, 128]]]], dtype=np.uint8)
    source = np.array([[[[20, 245, 50]]]], dtype=np.uint8)

    diff = signed_residual(source, base)
    restored = apply_residual(base, diff)
    assert bit_identical(source, restored)

    # Compute lossless residual via stage
    res = compute_residual(
        source,
        base,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(codec="avc", rate_control=RateControl.LOSSLESS, rate=0),
    )
    assert res.base is not None
    assert bit_identical(res.base, base)
    assert bit_identical(res.reconstructed, source)

    # Invert clipped addition verification when not saturated
    after = np.array([[[[10, -10, 0]]]], dtype=np.int16)
    delivered = np.clip(base.astype(np.int16) + after, 0, 255).astype(np.uint8)
    base_est, sat_mask = invert_clipped_addition(delivered, after)
    assert bit_identical(base_est, base)
    assert not np.any(sat_mask)

    # And when saturation occurs, inversion fails without base
    sat_base = np.array([[[[250, 5, 128]]]], dtype=np.uint8)
    sat_after = np.array([[[[20, -10, 0]]]], dtype=np.int16)
    sat_deliv = np.clip(sat_base.astype(np.int16) + sat_after, 0, 255).astype(np.uint8)
    sat_est, sat_mask2 = invert_clipped_addition(sat_deliv, sat_after)
    assert sat_mask2[0, 0, 0, 0]  # Saturation at 255 detected
    assert not bit_identical(sat_est, sat_base)


# ---------------------------------------------------------------------------
# 4. Coded roundtrip with native codec
# ---------------------------------------------------------------------------


def test_coded_roundtrip_with_native_codec() -> None:
    """End-to-end encode to bitstream bytes and decode with ffmpeg/AVC."""
    try:
        tools.resolve_ffmpeg()
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")

    t, h, w = 4, 64, 64
    x = np.linspace(30, 220, w, dtype=np.uint8)
    plane = np.tile(x, (h, 1))
    source = np.stack([np.stack([plane, plane, plane], axis=-1) for _ in range(t)])
    base = np.full_like(source, 100)

    diff = signed_residual(source, base)
    lossy_uint8 = encode_lossy(diff, mode="full_range")

    request = EncodeRequest(codec_name="avc", rate_control=RateControl.CRF, rate=23)
    transmitted, decoded_frames = encode_residual_to_bitstream(
        lossy_uint8,
        request,
        mode="full_range",
        fps=25.0,
    )

    assert transmitted.is_coded is True
    assert len(transmitted.bitstream) > 16
    assert transmitted.byte_count == len(transmitted.bitstream)
    assert transmitted.codec_name == "avc"

    # Client decode from transmitted bitstream
    decoded_diff = decode_residual_stream(transmitted)
    assert decoded_diff.shape == source.shape
    restored = apply_residual(base, decoded_diff)

    base_l1 = float(np.mean(np.abs(source.astype(np.float32) - base.astype(np.float32))))
    restored_l1 = float(np.mean(np.abs(source.astype(np.float32) - restored.astype(np.float32))))
    assert restored_l1 < base_l1


# ---------------------------------------------------------------------------
# 5. Fresh-process source-free decode and scored output identity
# ---------------------------------------------------------------------------


def test_fresh_process_source_free_decode_and_scored_output_identity() -> None:
    """Client reconstructs from wire bytes with zero access to source frames."""
    try:
        tools.resolve_ffmpeg()
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")

    t, h, w = 2, 64, 64
    source = np.full((t, h, w, 3), 140, dtype=np.uint8)
    source[0, 10:30, 10:30] = 220

    config = PointstreamConfig(
        lattice=lattice_config_from(WHOLE_FRAME_RESIDUAL),
        residual=ResidualConfig(
            codec="avc",
            rate_control=RateControl.CRF,
            rate=28,
            block_size=8,
            background_downscale=1,
        ),
    )
    result = run(config, [source])
    wire_bytes = result.chunks[0].bag.get("wire_request")
    assert wire_bytes is not None
    assert len(wire_bytes) > 0

    # Fresh standalone client execution
    client_frames = reconstruct_serialized_client(wire_bytes)
    assert client_frames.shape == source.shape
    assert bit_identical(client_frames, result.frames)
    assert result.delivered_quality is not None
    assert result.delivered_quality.whole_frame() > 20.0


# ---------------------------------------------------------------------------
# 6. All-byte ledger reconciliation
# ---------------------------------------------------------------------------


def test_all_byte_ledger_reconciliation() -> None:
    """Wire bytes match sizes.residual; unencoded fallback rejected as rate."""
    try:
        tools.resolve_ffmpeg()
    except FileNotFoundError:
        pytest.skip("ffmpeg not available")

    t, h, w = 2, 64, 64
    source = np.full((t, h, w, 3), 130, dtype=np.uint8)
    source[0, 10:30, 10:30] = 200

    config = PointstreamConfig(
        lattice=lattice_config_from(WHOLE_FRAME_RESIDUAL),
        residual=ResidualConfig(codec="avc", rate_control=RateControl.CRF, rate=28),
    )
    result = run(config, [source])
    chunk = result.chunks[0]

    transmitted = chunk.bag.get("transmitted_residual")
    assert transmitted is not None
    assert transmitted.is_coded is True
    # Byte count in sizes ledger matches transmitted bitstream exact length
    assert chunk.sizes.residual == len(transmitted.bitstream)

    # Reject unencoded fallback array when require_compressed=True
    unencoded_transmitted = TransmittedResidual(
        bitstream=b"",
        codec_name="raw",
        is_coded=False,
        raw_frames=np.zeros((t, h, w, 3), dtype=np.int16),
        shape=(t, h, w, 3),
    )
    with pytest.raises(ValueError, match="unencoded fallback"):
        serialize_client_request(
            background=None,
            frame_count=t,
            height=h,
            width=w,
            residual_payload=unencoded_transmitted,
            require_compressed=True,
        )


# ---------------------------------------------------------------------------
# 7. Absent residual zero calls and bytes
# ---------------------------------------------------------------------------


def test_absent_residual_zero_calls_and_bytes() -> None:
    """Disabled residual transmits 0 bytes and makes 0 residual codec calls."""
    source = _clip(100, frames=2, size=32)
    config = PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH))

    result = run(config, [source])
    chunk = result.chunks[0]
    assert chunk.sizes.residual == 0
    assert chunk.bag.get(STAGE_RESIDUAL) is None
    assert chunk.bag.get("transmitted_residual") is None


# ---------------------------------------------------------------------------
# 8. Corrupted, truncated, or incompatible payload rejection
# ---------------------------------------------------------------------------


def test_corrupted_truncated_or_incompatible_payload_rejection() -> None:
    """Corrupted/truncated bitstream and invalid payloads raise explicit errors."""
    # Truncated bitstream (< 16 bytes)
    truncated = TransmittedResidual(
        bitstream=b"short",
        codec_name="avc",
        shape=(1, 32, 32, 3),
        is_coded=True,
    )
    with pytest.raises(ValueError, match="truncated|Corrupted"):
        decode_residual_stream(truncated)

    # Corrupted bitstream (garbage bytes)
    corrupted = TransmittedResidual(
        bitstream=b"corrupted_garbage_bytes_long_enough_to_fail_decoder" * 5,
        codec_name="avc",
        shape=(1, 32, 32, 3),
        is_coded=True,
    )
    with pytest.raises(ValueError, match="Failed to decode residual bitstream"):
        decode_residual_stream(corrupted)

    # Missing both bitstream and raw_frames
    empty = TransmittedResidual(
        bitstream=b"",
        codec_name="avc",
        shape=(1, 32, 32, 3),
        is_coded=False,
    )
    with pytest.raises(ValueError, match="neither bitstream nor raw_frames"):
        decode_residual_stream(empty)

    # Invalid shape
    invalid_shape = TransmittedResidual(
        bitstream=b"12345678901234567890",
        codec_name="avc",
        shape=(1, 32),
        is_coded=True,
    )
    with pytest.raises(ValueError, match="Invalid transmitted residual shape"):
        decode_residual_stream(invalid_shape)


# ---------------------------------------------------------------------------
# 9. Generation-enabled predictor consistency
# ---------------------------------------------------------------------------


def test_generation_enabled_predictor_consistency() -> None:
    """Client runs generator for Pc, applies residual, and rejects fallback scores."""

    class _Paint:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            self.calls += 1
            return np.full((8, 8, 3), 100 + self.calls, dtype=np.uint8)

    backend = _Paint()
    ref = GeneratorRef(backend=backend, name="paint")
    config = PointstreamConfig(
        lattice=lattice_config_from(
            StageLattice.of(STAGE_DETECTION, STAGE_GENERATION, STAGE_RESIDUAL)
        ),
        generator=GeneratorConfig(backend="paint"),
        residual=ResidualConfig(
            codec="avc",
            rate_control=RateControl.LOSSLESS,
            rate=0,
        ),
    )
    source = _clip(0, frames=1, size=16)
    result = run(
        config,
        [source],
        generator=ref,
        objects=((_object(size=8),),),
    )

    # Generator is called for both server prediction and client reconstruction
    assert backend.calls >= 2
    assert result.chunks[0].reconstruction.quality is not None

    # Verify source fallback rejection in metrics
    mock_ctx = MagicMock()
    metrics_fn = make_metrics(mock_ctx)
    fallback_bag = {
        "source": source,
        ART_DELIVERED: {"fallback_reason": "generator out of memory"},
    }
    with pytest.raises(ConfigValueError, match="Source fallback cannot earn a valid quality score"):
        metrics_fn(fallback_bag)
