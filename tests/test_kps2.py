"""KPS2: the range-coded keypoint packet decodes to exactly what KPS1 decodes to."""

import numpy as np
import pytest

from experiments.tennis.k6b_rate_streams import (
    decode_keypoints,
    encode_keypoints,
    encode_keypoints_kps2,
    transcode_kps1_to_kps2,
)


def _track(rng: np.random.Generator, times: np.ndarray) -> np.ndarray:
    """Every-4th-frame q4 samples with linear interpolation between them, as KPS1 sends them."""
    kp = np.zeros((len(times), 17, 3))
    base = rng.integers(100, 400, size=(17, 2)) * 4.0
    walk = np.cumsum(rng.integers(-2, 3, size=(len(times), 17, 2)) * 4.0, axis=0)
    kp[:, :, :2] = base + walk
    kp[:, :, 2] = rng.choice([0.3, 0.9], size=(len(times), 17), p=[0.1, 0.9])
    return kp


def _kps1_units(rng: np.random.Generator) -> list[dict]:
    # Runs that do not end on a multiple of 4, a one-frame run, and two players.
    t1 = np.asarray([*range(0, 37), *range(50, 51), *range(60, 130)], np.int64)
    t2 = np.arange(5, 240, dtype=np.int64)
    units = [{"name": "near", "t": t1, "kp": _track(rng, t1)}, {"name": "far", "t": t2, "kp": _track(rng, t2)}]
    _, decoded = decode_keypoints(encode_keypoints({"native_frames": 240}, units))
    return decoded  # KPS1's sent geometry


def test_kps2_decodes_exactly_like_kps1_and_is_smaller():
    rng = np.random.default_rng(0)
    units = _kps1_units(rng)
    kps1 = encode_keypoints({"native_frames": 240, "window": "w"}, units)
    kps2 = transcode_kps1_to_kps2(kps1)
    meta1, a = decode_keypoints(kps1)
    meta2, b = decode_keypoints(kps2)
    assert kps2[:4] == b"KPS2" and len(kps2) < len(kps1)
    assert meta2["window"] == meta1["window"] and meta2["native_frames"] == 240
    for x, y in zip(a, b):
        assert x["name"] == y["name"]
        assert np.array_equal(x["t"], y["t"])
        assert np.array_equal(x["kp"], y["kp"])


def test_kps2_rejects_keypoints_off_the_sent_geometry():
    rng = np.random.default_rng(1)
    units = _kps1_units(rng)
    units[0]["kp"] = units[0]["kp"].copy()
    units[0]["kp"][2, 3, 0] += 1.0  # a between-sample frame that linear interpolation cannot give
    with pytest.raises(ValueError):
        encode_keypoints_kps2({"native_frames": 240}, units)


def test_kps2_integrity_and_profile_checks():
    rng = np.random.default_rng(2)
    kps2 = transcode_kps1_to_kps2(encode_keypoints({"native_frames": 240}, _kps1_units(rng)))
    with pytest.raises(ValueError):
        decode_keypoints(kps2 + b"x")
    corrupt = bytearray(kps2)
    corrupt[-3] ^= 1
    with pytest.raises(Exception):
        decode_keypoints(bytes(corrupt))
