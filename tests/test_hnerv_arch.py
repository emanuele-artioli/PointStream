"""Fast, GPU-free unit tests for src/shared/hnerv_arch.py (no real video/weights)."""

from __future__ import annotations

import pytest
import torch

from src.shared.hnerv_arch import (
    HNeRVConfig,
    HNeRVDecoder,
    HNeRVModel,
    count_decoder_parameters,
    dequantize_tensor_int8,
    load_hnerv_checkpoint,
    quantize_tensor_int8,
    save_hnerv_checkpoint,
)

TINY_CONFIG = HNeRVConfig(
    height=8,
    width=8,
    embed_height=2,
    embed_width=2,
    embed_channels=6,
    strides=(2, 2),
    channels=(8, 4),
)


def test_config_rejects_mismatched_channels_and_strides_length() -> None:
    with pytest.raises(ValueError, match="channels"):
        HNeRVConfig(
            height=8,
            width=8,
            embed_height=2,
            embed_width=2,
            embed_channels=4,
            strides=(2, 2),
            channels=(8,),
        )


def test_config_rejects_stride_product_mismatch() -> None:
    with pytest.raises(ValueError, match="height"):
        HNeRVConfig(
            height=9,  # 2 * (2*2) == 8 != 9
            width=8,
            embed_height=2,
            embed_width=2,
            embed_channels=4,
            strides=(2, 2),
            channels=(8, 4),
        )


def test_config_round_trips_through_dict() -> None:
    payload = TINY_CONFIG.as_dict()
    restored = HNeRVConfig.from_dict(payload)
    assert restored == TINY_CONFIG


def test_hnerv_model_forward_shapes() -> None:
    model = HNeRVModel(TINY_CONFIG)
    frames = torch.rand(3, 3, TINY_CONFIG.height, TINY_CONFIG.width)
    reconstruction, embedding = model(frames)
    assert reconstruction.shape == (3, 3, TINY_CONFIG.height, TINY_CONFIG.width)
    assert embedding.shape == (3, TINY_CONFIG.embed_channels, TINY_CONFIG.embed_height, TINY_CONFIG.embed_width)
    assert torch.all(reconstruction >= 0.0) and torch.all(reconstruction <= 1.0)


def test_decoder_alone_matches_model_decoder_output() -> None:
    model = HNeRVModel(TINY_CONFIG)
    embedding = torch.rand(2, TINY_CONFIG.embed_channels, TINY_CONFIG.embed_height, TINY_CONFIG.embed_width)
    model.eval()
    with torch.no_grad():
        direct = model.decoder(embedding)
    standalone_decoder = HNeRVDecoder(TINY_CONFIG)
    standalone_decoder.load_state_dict(model.decoder.state_dict())
    standalone_decoder.eval()
    with torch.no_grad():
        via_standalone = standalone_decoder(embedding)
    assert torch.allclose(direct, via_standalone)


def test_count_decoder_parameters_positive() -> None:
    decoder = HNeRVDecoder(TINY_CONFIG)
    assert count_decoder_parameters(decoder) > 0


def test_quantize_dequantize_round_trip_within_tolerance() -> None:
    tensor = torch.randn(4, 6, 2, 2) * 3.0
    quantized, scale, zero_point = quantize_tensor_int8(tensor)
    assert quantized.dtype == torch.uint8
    dequantized = dequantize_tensor_int8(quantized, scale, zero_point)
    # int8 quantization error bound: at most half a quantization step.
    max_error = scale / 2.0 + 1e-5
    assert torch.max(torch.abs(dequantized - tensor)).item() <= max_error


def test_quantize_constant_tensor_does_not_divide_by_zero() -> None:
    tensor = torch.full((2, 2), 0.5)
    quantized, scale, zero_point = quantize_tensor_int8(tensor)
    dequantized = dequantize_tensor_int8(quantized, scale, zero_point)
    assert torch.allclose(dequantized, tensor)


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    decoder = HNeRVDecoder(TINY_CONFIG)
    decoder.eval()
    embeddings = torch.rand(3, TINY_CONFIG.embed_channels, TINY_CONFIG.embed_height, TINY_CONFIG.embed_width)

    checkpoint_path = tmp_path / "hnerv_checkpoint.pt.gz"
    byte_size = save_hnerv_checkpoint(checkpoint_path, decoder, embeddings)

    assert checkpoint_path.is_file()
    assert byte_size == checkpoint_path.stat().st_size
    assert byte_size > 0

    with torch.no_grad():
        original_output = decoder(embeddings)

    loaded_decoder, loaded_embeddings = load_hnerv_checkpoint(checkpoint_path)
    loaded_decoder.eval()
    with torch.no_grad():
        loaded_output = loaded_decoder(loaded_embeddings)

    # fp16 decoder weights + int8 embeddings introduce quantization error;
    # the round trip should still be close, not bit-exact.
    assert torch.max(torch.abs(loaded_output - original_output)).item() < 0.15
    assert loaded_embeddings.shape == embeddings.shape
