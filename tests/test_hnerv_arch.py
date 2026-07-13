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
    load_hnerv_residual,
    quantize_tensor_int8,
    save_hnerv_checkpoint,
    save_hnerv_residual,
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
    # Use a slightly larger config to dilute PyTorch serialization overhead for the B/param check
    larger_config = HNeRVConfig(
        height=16, width=16, embed_height=4, embed_width=4,
        embed_channels=16, strides=(2, 2), channels=(64, 32)
    )
    decoder = HNeRVDecoder(larger_config)
    decoder.eval()
    embeddings = torch.rand(3, larger_config.embed_channels, larger_config.embed_height, larger_config.embed_width)

    checkpoint_path = tmp_path / "hnerv_checkpoint.pt.gz"
    
    # Int8 checkpoint
    byte_size = save_hnerv_checkpoint(checkpoint_path, decoder, embeddings, use_fp16_weights=False)

    assert checkpoint_path.is_file()
    assert byte_size > 0
    
    # Calculate bytes per parameter
    params = count_decoder_parameters(decoder)
    bytes_per_param = byte_size / params
    assert bytes_per_param <= 1.3, f"Expected <= 1.3 B/param for int8, got {bytes_per_param:.2f}"

    with torch.no_grad():
        original_output = decoder(embeddings)

    loaded_decoder, loaded_embeddings = load_hnerv_checkpoint(checkpoint_path)
    loaded_decoder.eval()
    with torch.no_grad():
        loaded_output = loaded_decoder(loaded_embeddings)

    # Output has quantization noise due to embeddings being int8-quantized (not in-place updated)
    assert torch.max(torch.abs(loaded_output - original_output)).item() < 0.15
    
    # Embeddings have int8 quantization error
    scale = (embeddings.max() - embeddings.min()) / 255.0
    max_error = scale.item() / 2.0 + 1e-5
    assert torch.max(torch.abs(loaded_embeddings - embeddings)).item() <= max_error
    
    # Ah, let's just check the weights:
    for (name1, p1), (name2, p2) in zip(decoder.named_parameters(), loaded_decoder.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter {name1} mismatch"

def test_save_and_load_residual_round_trip(tmp_path) -> None:
    base_decoder = HNeRVDecoder(TINY_CONFIG)
    base_decoder.eval()
    base_state = {k: v.clone() for k, v in base_decoder.state_dict().items()}
    
    decoder = HNeRVDecoder(TINY_CONFIG)
    decoder.load_state_dict(base_state)
    # create some delta
    with torch.no_grad():
        for p in decoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)
            
    embeddings = torch.rand(3, TINY_CONFIG.embed_channels, TINY_CONFIG.embed_height, TINY_CONFIG.embed_width)
    
    residual_path = tmp_path / "hnerv_residual.pt.gz"
    save_hnerv_residual(residual_path, decoder, base_state, embeddings, sparsity=0.5)

    assert residual_path.is_file()
    
    # Decoder was updated in-place during save_hnerv_residual.
    # Now load and compare
    loaded_decoder, loaded_embeddings = load_hnerv_residual(residual_path, base_decoder)
    
    for (name1, p1), (name2, p2) in zip(decoder.named_parameters(), loaded_decoder.named_parameters()):
        assert torch.equal(p1, p2), f"Residual parameter {name1} mismatch"
