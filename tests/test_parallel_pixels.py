"""Decoder support and byte-accounting checks for independently delivered tiles."""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from experiments.tennis.parallel_pixels import composite, crop_yuv, header_decode, header_encode, rectangles


def test_rectangles_even_cover_moving_regions() -> None:
    r = np.zeros((3, 2, 1080, 1920), bool)
    r[0, 0, 11:120, 17:58] = True
    r[1, 0, 7:100, 1800:1901] = True
    r[2, 0, 0:111, 0:53] = True
    r[:, 1, 1000:1080, 1870:1920] = True
    sizes, positions = rectangles(r)
    assert not np.any(positions % 2)
    for t in range(3):
        for p, (w, h) in enumerate(sizes):
            x, y = positions[t, p]
            covered = np.zeros((1080, 1920), bool)
            covered[y:y+h, x:x+w] = True
            assert not np.any(r[t, p] & ~covered)


def test_header_roundtrip_and_invalid_position_rejected() -> None:
    packet = header_encode([[64, 66]], np.array([[[2, 4]], [[6, 8]]]), 25)
    h = header_decode(packet)
    assert h['sizes'] == [[64, 66]]
    assert h['positions'] == [[[2, 4]], [[6, 8]]]
    h['positions'][0][0][0] = 3
    raw = json.dumps(h).encode()
    bad = b'PXT1' + struct.pack('<I', len(raw)) + zlib.compress(raw)
    with pytest.raises(ValueError, match='contract'):
        header_decode(bad)
    with pytest.raises(ValueError):
        header_decode(packet[:6])


def test_native_yuv_crop_has_correct_chroma_and_bounds() -> None:
    source = np.zeros((1620, 1920), np.uint8)
    source[:1080] = 17
    source[1080:].reshape(2, 540, 960)[0] = 23
    source[1080:].reshape(2, 540, 960)[1] = 29
    crop = crop_yuv(source, 2, 4, 64, 66)
    assert crop.shape == (99, 64)
    assert np.all(crop[:66] == 17)
    c = crop[66:].reshape(2, 33, 32)
    assert np.all(c[0] == 23) and np.all(c[1] == 29)
    with pytest.raises(ValueError, match='outside'):
        crop_yuv(source, 1900, 0, 64, 66)


def test_decoder_only_requires_rectangles_not_source_region() -> None:
    header = header_decode(header_encode([[64, 64]], np.array([[[12, 14]]]), 25))
    base = np.full((1080, 1920, 3), 3, np.uint8)
    tile = np.full((64, 64, 3), 99, np.uint8)
    result = composite(base, [tile], header, 0)
    assert np.all(result[14:78, 12:76] == 99)
    outside = np.ones((1080, 1920), bool)
    outside[14:78, 12:76] = False
    assert np.array_equal(result[outside], base[outside])
    assert np.all(base == 3)
