import numpy as np
import pytest
from experiments.tennis.k6b_sampled_residual import encode, decode


def test_sampled_residual_exact_sequence_and_packet_integrity() -> None:
    target = np.zeros((2, 7, 9), bool)
    target[0, 1:4, 2:6] = True
    target[1, 3:6, 0:2] = True
    prediction = np.roll(target, 1, axis=2)
    residuals = [target ^ prediction, np.roll(target, 1, axis=1) ^ prediction]
    blob = encode({'units': ['a', 'b'], 'ticks': [0, 3]}, residuals)
    meta, decoded = decode(blob)
    assert meta['ticks'] == [0, 3]
    for expected, actual in zip(residuals, decoded):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(prediction ^ decoded[0], target)
    with pytest.raises((ValueError, RuntimeError)):
        decode(blob[:-1])
