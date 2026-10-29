import json
import zlib

import numpy as np
import pytest

from experiments.tennis.persistent_memory import (
    H, W, choose_blocks, decode_blocks, decode_header, encode_header,
    object_history, predictions,
)
from experiments.tennis.parallel_background import pack_residual, reconstruct


def test_wire_selection_reconstructs_luma_and_chroma_without_source():
    black = np.full((H*3//2, W), 16, np.uint8)
    white = np.full_like(black, 200)
    source = black.copy()
    source[:32, :32] = 200
    labels = choose_blocks(source, [black, white])
    packet = encode_header(np.zeros((1, 2, 4), np.int32), labels[None], False)
    _, received, mode = decode_header(packet, 1)
    assert not mode
    result = decode_blocks([black, white], received[0])
    assert np.array_equal(result[:H], source[:H])
    assert np.all(result[H:].reshape(2, H//2, W//2)[:, :16, :16] == 200)
    assert np.all(result[H:].reshape(2, H//2, W//2)[:, 16:, :] == 16)


def test_object_prediction_uses_decoded_previous_patch_and_delivered_geometry():
    previous = np.full((H*3//2, W), 128, np.uint8)
    previous[10:30, 20:40] = 77
    old = np.array([[20, 10, 20, 20], [0, 0, 0, 0]])
    new = np.array([[100, 100, 40, 60], [0, 0, 0, 0]])
    result = object_history(previous, old, new)
    assert np.all(result[100:160, 100:140] == 77)
    assert np.array_equal(previous[100:160, 100:140], np.full((60, 40), 128))
    boxes = np.stack([old, new])
    base = np.stack([previous, np.full_like(previous, 99)])
    matrix = np.repeat(np.eye(3)[None], 2, axis=0)
    rows = predictions(base, previous, matrix, boxes, True)
    assert np.all(rows[1][3][100:160, 100:140] == 77)
    assert np.all(rows[1][0] == 99)


def test_corrupt_selector_and_outside_box_are_rejected():
    packet = encode_header(np.zeros((1, 2, 4), np.int32), np.zeros((1, 17, 30), np.uint8), True)
    header = json.loads(zlib.decompress(packet))
    header['labels'][0][0][0] = 4
    with pytest.raises(ValueError):
        decode_header(zlib.compress(json.dumps(header).encode()), 1)
    header['labels'][0][0][0] = 0
    header['boxes'][0][0] = [960, 0, 32, 32]
    with pytest.raises(ValueError):
        decode_header(zlib.compress(json.dumps(header).encode()), 1)


def test_finite_residual_relaxation_is_not_a_lossless_oracle():
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    reference = np.full_like(source, 123)
    restored = reconstruct(reference, pack_residual(source, reference))
    assert np.max(np.abs(restored.astype(int)-source.astype(int))) <= 1
    assert not np.array_equal(restored, source)
