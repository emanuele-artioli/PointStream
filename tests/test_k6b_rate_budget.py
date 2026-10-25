from __future__ import annotations

import pytest
import numpy as np

from experiments.tennis.k6b_rate_budget import comparison, headroom
from experiments.tennis.k6b_rate_streams import (
    decode_keypoints,
    decode_masks,
    decode_reference,
    encode_keypoints,
    encode_masks,
    encode_reference,
)


def row(error: float, rate: float = 2.0):
    return {
        "anchor_points": {"low": {"kbps": 2.0, "lpips": 0.5}, "high": {"kbps": 6.0, "lpips": 0.25}},
        "method": {"lpips": error},
        "billed_kbps": rate,
    }


def test_inverse_crossing_keeps_strict_equality_and_losing_cell():
    r = row(0.375)
    h = headroom(r, "lpips")
    assert h["threshold_kbps"] == 4.0
    assert h["headroom_kbps"] == 2.0
    assert comparison(r, "lpips", 3.999)["won"] is True
    assert comparison(r, "lpips", 4.0)["won"] is False
    assert headroom(row(0.375, 5.0), "lpips")["headroom_kbps"] == -1.0


def test_inverse_censors_beyond_either_measured_end():
    hi = headroom(row(0.1), "lpips")
    assert hi["threshold_kbps"] is None and hi["headroom_lower_bound_kbps"] == 4.0
    lo = headroom(row(0.75), "lpips")
    assert lo["threshold_kbps"] is None and lo["headroom_upper_bound_kbps"] == 0.0
    assert comparison(row(0.1), "lpips", 7.0)["won"] is None
    assert comparison(row(0.75), "lpips", 1.0)["won"] is None


def test_equality_at_each_anchor_endpoint_is_not_a_win():
    assert headroom(row(0.5), "lpips")["threshold_kbps"] == 2.0
    assert headroom(row(0.25), "lpips")["threshold_kbps"] == 6.0
    assert comparison(row(0.5), "lpips", 2.0)["won"] is False
    assert comparison(row(0.25), "lpips", 6.0)["won"] is False


def test_inverse_uses_achievable_hull_and_rejects_nonfinite():
    r = row(0.375)
    r["anchor_points"]["dominated"] = {"kbps": 4.0, "lpips": 0.49}
    assert headroom(r, "lpips")["threshold_kbps"] == 4.0
    r["anchor_points"]["bad"] = {"kbps": float("nan"), "lpips": 0.1}
    with pytest.raises(ValueError):
        headroom(r, "lpips")


def test_keypoint_stream_reconstructs_run_ends_and_every_frame_visibility():
    # Two runs, including one frame; samples every4, last held, visibility changes between sends.
    t = np.asarray([0, 1, 2, 3, 4, 5, 9], np.int64)
    xy = np.zeros((7, 17, 3), np.float64)
    xy[:4, :, 0] = np.arange(4)[:, None]
    xy[4:6, :, 0] = 4
    xy[6, :, 0] = 12
    xy[:, :, 1] = 8
    xy[:, :, 2] = 1
    xy[1, :, 2] = 0.49
    xy[2, :, 2] = 0.5
    payload = encode_keypoints({"native_frames": 10}, [{"name": "player", "t": t, "kp": xy}])
    meta, restored = decode_keypoints(payload)
    assert meta["units"][0]["runs"] == [[0, 6], [9, 1]]
    assert np.array_equal(restored[0]["t"], t)
    assert np.array_equal(restored[0]["kp"][:, :, :2], xy[:, :, :2])
    assert np.array_equal(restored[0]["kp"][:, :, 2] >= 0.5, xy[:, :, 2] >= 0.5)
    with pytest.raises(ValueError):
        decode_keypoints(payload + b"x")
    corrupt = bytearray(payload)
    corrupt[-4] ^= 1
    with pytest.raises(Exception):
        decode_keypoints(bytes(corrupt))


@pytest.mark.parametrize("mode", ["rle", "packed", "xor"])
def test_real_matte_coders_preserve_overlap_empty_and_motion(mode):
    masks = np.zeros((5, 2, 12, 16), bool)
    for t in range(1, 5):
        masks[t, 0, 2:8, t : t + 5] = True
        masks[t, 1, 4:10, 5:12] = True
    payload = encode_masks(masks, mode)
    _, restored = decode_masks(payload)
    assert np.array_equal(restored, masks)
    with pytest.raises(Exception):
        decode_masks(payload[:-1])


def test_reference_png_is_color_and_pixel_exact_and_lossy_is_not_relabelled():
    image = np.random.default_rng(31).integers(0, 256, (32, 24, 3), dtype=np.uint8)
    payload = encode_reference(image, {"unit": "u"})
    meta, restored = decode_reference(payload)
    assert meta["quality"] is None and np.array_equal(restored, image)
    _, lossy = decode_reference(encode_reference(image, {"unit": "u"}, 70))
    assert not np.array_equal(lossy, image)
