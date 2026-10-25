from __future__ import annotations

import pytest
import numpy as np
import json

from experiments.tennis.k6b_rate_budget import accounting, comparison, headroom
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
    assert comparison(row(0.1), "lpips", 7.0)["observed_dominator"] is None
    high_rate_loss = comparison(row(0.375), "lpips", 7.0)
    assert high_rate_loss["won"] is None and high_rate_loss["observed_dominator"]["kbps"] == 6.0
    assert comparison(row(0.75), "lpips", 1.0)["won"] is None


def test_equality_at_each_anchor_endpoint_is_not_a_win():
    assert headroom(row(0.5), "lpips")["threshold_kbps"] == 2.0
    assert headroom(row(0.25), "lpips")["threshold_kbps"] == 6.0
    assert comparison(row(0.5), "lpips", 2.0)["won"] is False
    assert comparison(row(0.25), "lpips", 6.0)["won"] is False


def test_inverse_uses_achievable_hull_and_rejects_nonfinite():
    r = row(0.375)
    r["anchor_points"]["dominated"] = {"kbps": 4.0, "lpips": 0.49}
    r["anchor_points"]["high"]["unused_diagnostic"] = float("nan")
    json.dumps(comparison(r, "lpips", 7.0), allow_nan=False)
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


def accounting_fixture():
    rows = []
    costs = []
    for i in range(12):
        match = "alcaraz_hurkacz" if i < 6 else "medvedev_struff"
        window = f"{match}/{i}"
        r = row(0.375)
        for a in r["anchor_points"].values():
            a["dists"] = a["lpips"]
        r.update(
            window=window,
            match=match,
            method={"lpips": 0.375, "dists": 0.375},
            keypoint_kbps=1.0,
            registered_seconds=10.0,
            whole_window_seconds=12.0,
            billed={"lpips": {"won": True}, "dists": {"won": True}},
        )
        rows.append(r)
        costs.append(
            {
                "window": window,
                "match": match,
                "keypoints": {"bytes": 1500, "equivalent": True},
                "references": {x + "_bytes": 1000 for x in ("png", "jpeg70", "jpeg85", "jpeg95")},
                "mattes": {x: {"bytes": 100} for x in ("exact", "half_grid", "quarter_grid")},
            }
        )
    return (
        {"rows": rows, "complete": True, "limitations": {}, "provenance": {}},
        {"rows": costs, "provenance": {}},
    )


def test_accounting_replaces_entropy_once_and_pays_all_match_references():
    d, t = accounting_fixture()
    out = accounting(d, t)
    s = next(
        s
        for s in out["scenarios"]
        if s["reference"] == "png"
        and s["matte"] == "exact"
        and s["asset_amortization_seconds"] == 600
    )
    c = s["rows"][0]
    assert c["components_kbps"] == pytest.approx(
        {
            "keypoints_and_metadata": 1.2,
            "adapter_guider_7200s": 1.0,
            "reference": 0.08,
            "matte": 0.08,
        }
    )
    assert c["total_kbps"] == pytest.approx(2.36)
    assert c["whole_window_total_kbps"] == pytest.approx(1600 * 8 / 12 / 1000 + 1 + 0.08)
    assert out["png_bank_bytes_by_match"]["alcaraz_hurkacz"] == 6000
    lossy = next(
        s for s in out["scenarios"] if s["reference"] == "jpeg70" and s["matte"] == "half_grid"
    )
    assert "counterfactual" in lossy["quality_status"]


def test_accounting_rejects_duplicate_cohort_and_unverified_conditioning():
    d, t = accounting_fixture()
    t["rows"].append(t["rows"][0])
    with pytest.raises(ValueError):
        accounting(d, t)
    t["rows"].pop()
    t["rows"][0]["keypoints"]["equivalent"] = False
    with pytest.raises(ValueError):
        accounting(d, t)
