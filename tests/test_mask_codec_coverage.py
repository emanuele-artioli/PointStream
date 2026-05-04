from __future__ import annotations

import zlib

import cv2
import numpy as np
import pytest

from src.shared.mask_codec import decode_binary_mask, encode_binary_mask, encode_polygon_mask, normalize_mask_codec


def test_mask_codec_edge_cases_and_errors() -> None:
    assert normalize_mask_codec(None) == "auto"
    assert normalize_mask_codec(" polygon ") == "poly-v1"

    with pytest.raises(ValueError, match="Expected a 2D mask"):
        encode_binary_mask(np.zeros((2, 2, 2, 2), dtype=np.uint8), codec="png")

    with pytest.raises(ValueError, match="Mask dimensions must be positive"):
        encode_polygon_mask([], height=0, width=1)

    with pytest.raises(ValueError, match="Polygon masks only support"):
        encode_polygon_mask([np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)], height=4, width=4, codec="png")

    with pytest.raises(ValueError, match="RLE payload is empty"):
        decode_binary_mask(codec="rle-v1", payload=b"", height=1, width=1)

    with pytest.raises(ValueError, match="not aligned to uint32 runs"):
        decode_binary_mask(codec="rle-v1", payload=b"\x00\x01\x02", height=1, width=1)

    with pytest.raises(ValueError, match="truncated"):
        decode_binary_mask(codec="bitpack-z1", payload=zlib.compress(b"\x00"), height=8, width=8)

    bad_png = cv2.imencode(".png", np.zeros((1, 1), dtype=np.uint8))[1].tobytes()[:4]
    with pytest.raises(ValueError, match="Failed to decode PNG mask payload"):
        decode_binary_mask(codec="png", payload=bad_png, height=1, width=1)
