"""W4: the receiver mask in H5p 4a's composites (record, billing, specs, rule)."""
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.visor import h5, w4_receiver, w4_report, w4_specs
from src.pointstream import composite as cp
from src.pointstream import mask as mk
from src.pointstream import shape

SCALE = 8
SIZE = (96, 64)  # (width, height)


def classes_of(frames: int = 3) -> np.ndarray:
    out = np.zeros((frames, SIZE[1], SIZE[0]), np.uint8)
    for t in range(frames):
        out[t, 16:48, 16 + 4 * t:56 + 4 * t] = cp.OBJECT
        out[t, 16:32, 16 + 4 * t:40 + 4 * t] = cp.HAND
    return out


def saved(tmp_path: Path, classes: np.ndarray, predicted: np.ndarray | None = None) -> tuple[Path, Path]:
    reduced = (mk.reduce(classes, SCALE) > 0).astype(np.uint8)
    seed = tmp_path / "seed.shape"
    seed.write_bytes(shape.encode(reduced[:1], 0))
    masks = tmp_path / "masks.npz"
    np.savez_compressed(masks, predicted=reduced if predicted is None else predicted)
    return masks, seed


def test_payload_round_trip_and_rejects_others() -> None:
    payload = w4_receiver.pack_payload(b"seed", 8)
    assert payload[:2] == bytes([w4_receiver.VERSION, 8])
    assert w4_receiver.unpack_payload(payload) == (8, b"seed")
    with pytest.raises(ValueError):
        w4_receiver.unpack_payload(mk.encode(mk.reduce(classes_of(), SCALE), SCALE))


def test_exact_prediction_equals_f_hat_and_bills_only_the_seed(tmp_path: Path) -> None:
    classes = classes_of()
    masks, seed = saved(tmp_path, classes)
    shown, payload, record = w4_receiver.receiver_classes(classes, SCALE, masks, seed, 30.0)
    expected = mk.upsample(mk.reduce(classes, SCALE), SIZE)
    assert np.array_equal(shown > 0, expected > 0)
    assert len(payload) == len(seed.read_bytes()) + 2 == record["bytes"]
    assert record["receiver"] and record["seed_is_first_union_mask"]
    assert record["iou"] > 0.85 and record["iou_frame_min"] > 0.8
    assert record["kbps"] == pytest.approx(len(payload) * 8 / (3 / 30.0) / 1000)


def test_missed_hand_lowers_iou_and_is_counted(tmp_path: Path) -> None:
    classes = classes_of()
    reduced = (mk.reduce(classes, SCALE) > 0).astype(np.uint8)
    wrong = reduced.copy()
    wrong[:, :, : wrong.shape[2] // 2] = 0  # the tracker lost the left half
    (tmp_path / "ok").mkdir()
    _, _, exact = w4_receiver.receiver_classes(classes, SCALE, *saved(tmp_path / "ok", classes), 30.0)
    masks, seed = saved(tmp_path, classes, wrong)
    _, _, record = w4_receiver.receiver_classes(classes, SCALE, masks, seed, 30.0)
    assert record["iou"] < exact["iou"] and record["missed_share"] > 0.2 and record["extra_share"] < 0.1


def test_seed_that_is_not_the_first_frame_is_flagged(tmp_path: Path) -> None:
    classes = classes_of()
    masks, seed = saved(tmp_path, classes)
    seed.write_bytes(shape.encode(np.ones((1, *mk.reduce(classes[:1], SCALE).shape[1:]), np.uint8), 0))
    assert not w4_receiver.receiver_classes(classes, SCALE, masks, seed, 30.0)[2]["seed_is_first_union_mask"]


def test_too_few_predicted_frames_is_an_error(tmp_path: Path) -> None:
    classes = classes_of()
    masks, seed = saved(tmp_path, classes, (mk.reduce(classes[:2], SCALE) > 0).astype(np.uint8))
    with pytest.raises(ValueError, match="predicted frames"):
        w4_receiver.receiver_classes(classes, SCALE, masks, seed, 30.0)


def test_mask_billed_needs_the_seed_to_be_the_first_mask() -> None:
    rows = [{"breakdown": {"mask": 120}}]
    receiver = {"receiver": True, "bytes": 114, "seed_is_first_union_mask": True}
    assert h5.mask_billed(receiver, rows)
    assert not h5.mask_billed({**receiver, "seed_is_first_union_mask": False}, rows)
    assert not h5.mask_billed(receiver, [{"breakdown": {"mask": 114}}])
    assert not h5.mask_billed(receiver, [{"breakdown": {}}])
    assert h5.mask_billed(None, rows)
    assert h5.mask_billed({"bytes": 100, "lossless_on_reduced_grid": True}, rows)
    assert not h5.mask_billed({"bytes": 100, "lossless_on_reduced_grid": False}, rows)


def test_spec_reads_the_diagnostic_and_keeps_4a_pairs() -> None:
    base: dict[str, Any] = {"arguments": ["score", "--pairs", "a+b", "--tag", "h5p4-gho", "--mask-scale", "8"],
            "staged_inputs": [{"name": "video", "path": "/x", "sha256": "0" * 64}],
            "hosts": ["gpu2", "gpu3", "gpu5", "gpu6"], "gpu_models": ["RTX 6000 Ada", "RTX A6000"],
            "scale": {"limit": {"smoke": 24, "full": 0}, "combos": {"smoke": 3, "full": 0}},
            "smoke": {"seconds": 600, "representative_basis": "old"}, "full": {"seconds": 2400},
            "budget_seconds": 3600, "deadline": "old"}
    s = w4_specs.make_spec(base, "P03_10", "gho", "2026-10-30T00:00:00+02:00", 600)
    a = s["arguments"]
    assert a[a.index("--tag") + 1] == "w4-gho" and a[a.index("--pairs") + 1] == "a+b"
    assert a[-4:] == ["--receiver-masks", "{staged:receiver_masks}", "--receiver-seed", "{staged:receiver_seed}"]
    staged = {x["name"]: x for x in s["staged_inputs"]}
    assert staged["receiver_masks"]["path"].endswith("20261024T084607Z-e00640d5/full/arm0-dcvc-0_4.npz")
    assert staged["receiver_seed"]["sha256"] == w4_specs.DIAGNOSTICS["P03_10"]["seed"]
    assert s["hosts"] == ["gpu6"] and s["budget_seconds"] == 600 + 300 + 120
    assert base["arguments"][4] == "h5p4-gho" and len(base["staged_inputs"]) == 1  # the 4a spec is untouched
    with pytest.raises(ValueError):
        w4_specs.make_spec(base, "P03_10", "venue", "d", 600)


def curve(rates: list[float], dists: list[float]) -> list[dict[str, Any]]:
    return [{"kbps": r, "scores": {"visor_dense_sam_fill": {"wdists": d, "wlpips": d, "wpsnr": 30 - 100 * d},
                                   "frame": {"dists": d, "lpips": d, "psnr": 30 - 100 * d}}} for r, d in zip(rates, dists)]


def test_rule_needs_ten_percent_and_three_items() -> None:
    def summary(bds: list[float | None]) -> dict[str, Any]:
        items = {str(i): {"bd": {"receiver": {m: {"bd": b, "where": "overlap" if b is not None else "worse at every rate"}
                                              for m in w4_report.METRICS}}} for i, b in enumerate(bds)}
        return w4_report.summarize(items)

    assert w4_report.verdict(summary([-0.3, -0.2, -0.1, 0.1]))["passes"]
    assert not w4_report.verdict(summary([-0.2, -0.15, 0.05, 0.3]))["passes"]  # two items below zero
    assert not w4_report.verdict(summary([-0.05, -0.05, -0.05, -0.05]))["passes"]  # mean above -10%
    assert not w4_report.verdict(summary([-0.5, -0.5, -0.5, None]))["passes"]  # an item without a BD-rate


def test_without_mask_takes_the_mask_bytes_off_the_rate() -> None:
    row = {"kbps": 100.0, "bytes": 1000, "breakdown": {"mask": 250, "background": 750}}
    assert w4_report.without_mask(row)["kbps"] == pytest.approx(75.0)


def test_compare_variants_ranks_a_cheaper_curve_below_zero() -> None:
    anchors = curve([20, 40, 80, 160], [0.30, 0.25, 0.20, 0.16])
    better = curve([10, 20, 40, 80], [0.30, 0.25, 0.20, 0.16])
    out = w4_report.compare_variants({"receiver": better}, anchors)
    assert out["receiver"]["wdists_fill"]["bd"] < -0.4


def test_thin_picks_four_pairs_by_rate_rank_only() -> None:
    chosen = {(f"p{i}", "gho") for i in range(10)}
    rate = {(f"p{i}", "gho"): float(100 - i) for i in range(10)}  # p9 is the cheapest
    # ranks 0, 3, 6, 9 of the 10 sorted by rate: p9, p6, p3, p0
    assert w4_report.thin(chosen, rate) == {("p9", "gho"), ("p6", "gho"), ("p3", "gho"), ("p0", "gho")}
    few = {("a", "venue"), ("b", "gho")}
    assert w4_report.thin(few, {("a", "venue"): 1.0, ("b", "gho"): 2.0}) == few
    many = {(f"p{i}", "gho") for i in range(15)}
    assert len(w4_report.thin(many, {(f"p{i}", "gho"): float(i) for i in range(15)})) == 4


def test_spec_caps_follow_the_pair_count() -> None:
    assert w4_specs.FULL_BASE_SECONDS + w4_specs.FULL_PAIR_SECONDS * 7 == 1700


def test_rule_fails_on_fewer_items_but_never_passes_on_them() -> None:
    items = {str(i): {"bd": {"receiver": {m: {"bd": b, "where": "overlap"} for m in w4_report.METRICS}}}
             for i, b in enumerate([11.7, 0.4, 0.2])}
    v = w4_report.verdict(w4_report.summarize(items))
    assert not v["passes"] and v["fails_whatever_is_missing"] and v["planned"] == 4
    good = {str(i): {"bd": {"receiver": {m: {"bd": b, "where": "overlap"} for m in w4_report.METRICS}}}
            for i, b in enumerate([-0.3, -0.3, -0.3])}
    v = w4_report.verdict(w4_report.summarize(good))
    assert not v["passes"] and not v["fails_whatever_is_missing"]
