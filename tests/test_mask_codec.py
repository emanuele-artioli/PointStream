from __future__ import annotations

import numpy as np
import pytest

from src.shared.mask_codec import (
    decode_binary_mask,
    encode_binary_mask,
    encode_polygon_mask,
    normalize_mask_codec,
)


def _sample_mask(height: int = 48, width: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:40, 16:48] = 1
    mask[18:28, 26:38] = 0
    return mask


def test_mask_codec_roundtrip_rle_v1() -> None:
    source = _sample_mask()
    codec, payload, h, w = encode_binary_mask(source, codec="rle-v1")

    decoded = decode_binary_mask(codec=codec, payload=payload, height=h, width=w)

    assert codec == "rle-v1"
    assert decoded.shape == source.shape
    assert np.array_equal((decoded > 0).astype(np.uint8), source)


def test_mask_codec_roundtrip_png() -> None:
    source = _sample_mask()
    codec, payload, h, w = encode_binary_mask(source, codec="png")

    decoded = decode_binary_mask(codec=codec, payload=payload, height=h, width=w)

    assert codec == "png"
    assert decoded.shape == source.shape
    assert np.array_equal((decoded > 0).astype(np.uint8), source)


def test_mask_codec_roundtrip_bitpack_z1() -> None:
    source = _sample_mask()
    codec, payload, h, w = encode_binary_mask(source, codec="bitpack-z1")

    decoded = decode_binary_mask(codec=codec, payload=payload, height=h, width=w)

    assert codec == "bitpack-z1"
    assert decoded.shape == source.shape
    assert np.array_equal((decoded > 0).astype(np.uint8), source)


def test_mask_codec_auto_selects_smaller_payload() -> None:
    source = _sample_mask()
    auto_codec, auto_payload, h, w = encode_binary_mask(source, codec="auto")
    _ = (h, w)
    _rle_codec, rle_payload, _h, _w = encode_binary_mask(source, codec="rle-v1")
    _bit_codec, bit_payload, _h2, _w2 = encode_binary_mask(source, codec="bitpack-z1")

    assert auto_codec in {"rle-v1", "bitpack-z1"}
    assert len(auto_payload) == min(len(rle_payload), len(bit_payload))


def test_mask_codec_rejects_invalid_rle_payload() -> None:
    with pytest.raises(ValueError):
        decode_binary_mask(codec="rle-v1", payload=b"\x01\x00\x01", height=8, width=8)


def test_mask_codec_roundtrip_poly_v1() -> None:
    height, width = 48, 64
    polygons = [
        np.asarray([[10.0, 8.0], [40.0, 10.0], [36.0, 30.0], [12.0, 26.0]], dtype=np.float32),
    ]

    codec, payload, h, w = encode_polygon_mask(polygons, height=height, width=width, codec="poly-v1")
    decoded = decode_binary_mask(codec=codec, payload=payload, height=h, width=w)

    assert codec == "poly-v1"
    assert decoded.shape == (height, width)
    assert int(np.count_nonzero(decoded)) > 0


def test_normalize_mask_codec_aliases() -> None:
    assert normalize_mask_codec("rle") == "rle-v1"
    assert normalize_mask_codec("bitpack") == "bitpack-z1"
    assert normalize_mask_codec("poly") == "poly-v1"
    assert normalize_mask_codec("png") == "png"
    assert normalize_mask_codec("unknown") == "auto"
