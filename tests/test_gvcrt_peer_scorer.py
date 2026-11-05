"""CPU checks for the independent GVC scoring boundary."""
import numpy as np
import pytest

from experiments.generative.gvcrt_peer_scorer import decoded_rgb, sha256


def test_decode_round_clip_and_layout():
    values = np.zeros((4, 3, 720, 1280), dtype=np.float32)
    values[0, :, 0, 0] = [-0.5, 0.5, 1.5]
    result = decoded_rgb(values, 4)
    assert result.shape == (4, 720, 1280, 3)
    assert result.dtype == np.uint8
    assert result[0, 0, 0].tolist() == [0, 128, 255]


def test_decode_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="NCHW"):
        decoded_rgb(np.zeros((1, 3, 720, 1280)), 4)
    values = np.zeros((4, 3, 720, 1280), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        decoded_rgb(values, 4)


def test_sha256(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

