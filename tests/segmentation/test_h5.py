import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from experiments.visor import h5
from src.pointstream import bitstream as bs


def test_points_parse_and_name() -> None:
    assert h5.parse_point("dcvc@21/4") == (21, 4)
    assert h5.point_key("dcvc@0/2") == "dcvc-0_2"
    assert h5.coded_size(4) == (480, 272) and h5.coded_size(1) == (1920, 1080)
    with pytest.raises(SystemExit):
        h5.parse_point("svt@21/3")


def av1_ivf(frames: int) -> bytes:
    td = bytes([bs.OBU_TEMPORAL_DELIMITER << 3 | 0x02, 0])
    frame = bytes([6 << 3 | 0x02]) + bs.leb128(40) + bytes(40)
    return bs.obus_to_ivf((td + frame) * frames, 1920, 1080, Fraction(50))


def published(tmp: Path, name: str, data: bytes) -> tuple[str, str]:
    (tmp / name).write_bytes(data)
    return name, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("route", ["enh", "scene"])
def test_container_carries_the_published_streams(tmp_path: Path, route: str) -> None:
    if route == "enh":
        bg_file, bg_sha = published(tmp_path, "bg.ivf", av1_ivf(6))
        bg = {"key": "enh-1080-48", "route": "enh", "codec": "enh", "coded_size": [1920, 1080],
              "model_sha256": "11" * 32}
    else:
        bg_file, bg_sha = published(tmp_path, "bg.psdc", b"PSDC" + bytes(300))
        bg = {"key": "scene-540-13", "route": "scene", "codec": "scene", "coded_size": [960, 544],
              "model_sha256": "22" * 32}
    bg.update(root=str(tmp_path), bitstream=bg_file, bitstream_sha256=bg_sha)
    fg_file, fg_sha = published(tmp_path, "fg.psdc", b"PSDC" + bytes(range(200)))
    fg = {"key": "dcvc-21_4", "scale": 4, "coded_size": [480, 272], "bitstream": fg_file, "bitstream_sha256": fg_sha,
          "slots": {"hand": {"sha256": "33" * 32}, "object": None}}
    w = {"fps": Fraction(50), "frames": 6}
    c = h5.container_for(w, bg, fg, tmp_path)
    data = c.pack()
    checks = h5.bitstream_check(c, data, bg, fg, tmp_path)
    assert all(checks.values()), checks
    parts = c.breakdown()
    assert sum(parts.values()) == len(data)
    if route == "enh":  # AV1 is billed without IVF's framing, as B2 counts SVT-AV1
        assert parts["background"] < (tmp_path / bg_file).stat().st_size
    assert [m.kind for m in c.models()] == ["background", "foreground"]


def foreground_result(**change: object) -> dict:
    rows = []
    for scale in (4, 2, 1):
        for qp, kbps in ((0, 8.0), (21, 20.0), (42, 50.0)):
            rows.append({"name": f"dcvc@{qp}/{scale}", "scale": scale, "qp": qp, "kbps": kbps * (5 - scale), "deterministic": True,
                         "decoder_matches_encoder_intra": True, "psnr_f": 30.0 + (42 - qp) / 10,
                         "slots": {"hand": {"sha256": "x"} if scale in (2, 4) else None},
                         "changed_by_slots": 0.05 if scale in (2, 4) else 0.0, "decoded_frame_sha256": ["a"] * 240})
    result = {"kind": "foreground", "points": rows, "frames": 240, "limit_frames": 0, "jpeg_gate_holds": True,
              "mask_cost": {"bytes": 1000, "share": 0.3}}
    result.update(change)
    return result


def test_foreground_validator() -> None:
    assert all(h5.validate_foreground(foreground_result()).values())
    bad = foreground_result()
    bad["points"][1]["kbps"] = 999.0  # QP 21 dearer than QP 42
    assert not h5.validate_foreground(bad)["rate_rises_with_qp"]
    bad = foreground_result()
    bad["points"][0]["deterministic"] = False
    assert not h5.validate_foreground(bad)["every_point_coded"]


def scores(wpsnr: float = 32.0) -> dict:
    m: dict = {"wpsnr": wpsnr, "wlpips": 0.2, "wdists": 0.15, "psnr_fg": 30.0}
    return {h5.DENSE: dict(m), h5.FILL: dict(m), "frame": {"psnr": 33.0}}


def test_score_validator_checks_anchor_reproduction() -> None:
    row = {"bytes": 30, "breakdown": {"header": 20, "background": 10}, "kbps": 0.06, "scores": scores()}
    anchor: dict = {"deterministic": True, "stream_sha256": "s", "b2_stream_sha256": "s", "reencoded": False,
                    "kbps": 100.0, "scores": scores(32.0),
              "b2": {"stream_sha256": "s", "kbps": 100.0, "wpsnr": 32.01}}
    result = {"rows": [row], "anchors": [anchor], "pairs_planned": 1, "jpeg_gate_holds": True, "limit_frames": 0,
              "routes": ["scene"], "bitstream_checks": {"scene": {"pair": "p", "segments_round_trip": True},
                                                        "gpu_compositor": {"max_abs": 0}},
              "background_f_is_foreground_f": True}
    assert all(h5.validate_score(result).values())
    anchor["b2"]["wpsnr"] = 32.5
    assert not h5.validate_score(result)["anchors_reproduce_b2"]


def scored(kbps: float, lpips: float, route: str = "scene", codec: str | None = None) -> dict:
    s: dict = {m: {"wlpips": lpips, "wdists": lpips / 2, "wpsnr": 40 - 30 * lpips, "lpips_fg": lpips, "lpips_bg": lpips,
             "psnr_fg": 30.0, "psnr_bg": 30.0, "dists_fg": lpips, "dists_bg": lpips} for m in h5.MASK_SETS}
    s["frame"] = {"psnr": 40 - 30 * lpips, "lpips": lpips, "dists": lpips / 2}
    row = {"kbps": kbps, "scores": s}
    if codec:
        return {**row, "codec": codec, "point": kbps}
    return {**row, "route": route, "pair": f"{route}-{kbps}", "kbps_background": kbps / 2}


def test_compare_halves_rate_and_sides() -> None:
    anchor = [(100.0, -0.3), (200.0, -0.2), (400.0, -0.1)]
    test = [(50.0, -0.3), (100.0, -0.2), (200.0, -0.1)]
    c = h5.compare(test, anchor, 0.005)
    assert c["where"] == "overlap" and abs(c["bd"] + 0.5) < 1e-9
    above = h5.compare([(150.0, -0.05), (300.0, -0.04)], anchor, 0.005)
    assert above["where"] == "better at every rate" and h5.below_zero(above)
    assert h5.compare([(10.0, -0.5)], anchor, 0.005)["where"] == "no shared rate"


def test_report_decision_reads_the_rule() -> None:
    anchors = [scored(k, q, codec=c) for c in ("svtav1", "dcvc") for k, q in ((100, 0.3), (200, 0.2), (400, 0.1))]
    good = [scored(k, q) for k, q in ((50, 0.3), (100, 0.2), (200, 0.1))]
    items = {f"I{i}": h5.item_report(good, anchors, 1.0) for i in range(4)}
    out = h5.summary(items)
    assert out["decision"]["verdict"] == "passes"
    head = out["comparisons"]["pointstream|pooled|wlpips_fill"]
    assert abs(head["bd"]["mean"] + 0.5) < 1e-6 and head["items_below_zero"] == 4
    bad = [scored(k, q) for k, q in ((200, 0.3), (400, 0.2), (800, 0.1))]
    items = {f"I{i}": h5.item_report(bad, anchors, 1.0) for i in range(4)}
    assert h5.summary(items)["decision"]["verdict"] == "fails"


def test_container_bills_the_mask(tmp_path: Path) -> None:
    from src.pointstream import mask as mk

    bg_file, bg_sha = published(tmp_path, "bg.psdc", b"PSDC" + bytes(64))
    fg_file, fg_sha = published(tmp_path, "fg.psdc", b"PSDC" + bytes(32))
    bg = {"key": "scene-540-0", "route": "scene", "codec": "scene", "coded_size": [960, 544], "model_sha256": None,
          "root": str(tmp_path), "bitstream": bg_file, "bitstream_sha256": bg_sha}
    fg = {"key": "dcvc-0_4", "scale": 4, "coded_size": [480, 272], "bitstream": fg_file, "bitstream_sha256": fg_sha,
          "slots": {"hand": None, "object": None}}
    classes = np.zeros((4, 32, 48), np.uint8)
    classes[:, 8:20, 10:30] = 2
    payload = mk.encode(mk.reduce(classes, 4), 4)
    c = h5.container_for({"fps": Fraction(50), "frames": 4}, bg, fg, tmp_path, payload)
    parts = c.breakdown()
    assert parts["mask"] == 1 + len(bs.leb128(len(payload))) + len(payload)
    back = bs.Container.unpack(c.pack())
    assert [p for t, p in back.segments if t == bs.Tag.MASK] == [payload]


def test_variants_split_granted_and_coded_masks() -> None:
    rows = [{**scored(100, 0.2), "mask_scale": 0}, {**scored(120, 0.2), "mask_scale": 8},
            {**scored(110, 0.25), "mask_scale": 16}, {**scored(90, 0.3, route="enh"), "mask_scale": 0}]
    v = h5.variant_rows(rows)
    assert len(v["pointstream"]) == 2 and len(v["scene"]) == 1 and len(v["enh"]) == 1
    assert [r["kbps"] for r in v["mask8"]] == [120] and len(v["mask-hull"]) == 2


def test_tagged_variant_replaces_only_its_foregrounds() -> None:
    base = [{**scored(100, 0.2), "foreground": "dcvc-21_4"}, {**scored(300, 0.1), "foreground": "dcvc-63_1"}]
    gho = [{**scored(100, 0.18), "foreground": "dcvc-21_4", "tag": "gho"}]
    v = h5.variant_rows(base + gho)
    assert len(v["pointstream"]) == 2
    assert sorted((r["foreground"], r.get("tag", "")) for r in v["gho"]) == [("dcvc-21_4", "gho"), ("dcvc-63_1", "")]


def test_venue_foreground_falls_back_to_gho_and_is_matched() -> None:
    base = [{**scored(100, 0.2), "foreground": "dcvc-21_4"}, {**scored(300, 0.1), "foreground": "dcvc-63_1"}]
    gho = [{**scored(100, 0.18), "foreground": "dcvc-21_4", "tag": "gho"}]
    venue = [{**scored(300, 0.05), "foreground": "dcvc-63_1", "tag": "venuefg"}]
    v = h5.variant_rows(base + gho + venue)
    assert sorted((r["foreground"], r.get("tag", "")) for r in v["venuefg"]) == [("dcvc-21_4", "gho"),
                                                                              ("dcvc-63_1", "venuefg")]
    anchors = [scored(k, q, codec="dcvc-venue") for k, q in ((100, 0.3), (400, 0.1))]
    report = h5.item_report(base + gho + venue, anchors, 1.0)
    assert set(report["matched_by_variant"]) == {"gho", "venuefg"}
    assert report["matched_by_variant"]["venuefg"][-1]["fg_lpips"]["pointstream"] == 0.05


def test_foreground_validator_checks_the_venue_model() -> None:
    venue = {"fill": "none", "converged": {"converged": True}, "model_sha256": "m", "id": "g6-window/P01_107"}
    good = foreground_result(item="P01_107_0000003049", video_model={"video_ckpt_sha256": "m", "venue": venue})
    assert all(h5.validate_foreground(good).values())
    for change in ({"fill": "exact"}, {"converged": {"converged": False}}, {"model_sha256": "other"},
                   {"id": "g6-window/P02_02"}):
        bad = foreground_result(item="P01_107_0000003049",
                                video_model={"video_ckpt_sha256": "m", "venue": {**venue, **change}})
        assert not h5.validate_foreground(bad)["venue_model_is_the_windows"]


def test_billing_adds_the_routes_model_and_payback_repays_it() -> None:
    anchors = [scored(k, q, codec="dcvc") for k, q in ((100, 0.3), (200, 0.2), (400, 0.1))]
    rows = [scored(50, 0.3), scored(100, 0.2), scored(150, 0.15, route="enh"), scored(200, 0.1)]
    model = {"scene": 1_000_000.0, "enh": 100_000.0}  # bytes
    out = h5.billed(rows, model, 100.0)  # 80 kbps for the scene model, 8 kbps for the enhancer
    assert [round(r["kbps"] - b["kbps"], 6) for r, b in zip(out, rows)] == [80.0, 80.0, 8.0, 80.0]
    pay = h5.payback(rows, anchors, model)
    first = next(p for p in pay if p["kbps"] == 50)
    assert abs(first["anchor_kbps"] - 100) < 1e-6 and abs(first["payback_seconds"] - 8e6 / 50e3) < 1e-6
    report = h5.item_report(rows, anchors, 0.0, {"bytes": model, "seconds": {"video": 10.0, "kitchen": 1e6}})
    free = report["comparisons"]["pointstream|pooled|wlpips_fill"]["bd"]
    assert report["comparisons"]["pointstream@kitchen|pooled|wlpips_fill"]["bd"] == pytest.approx(free, abs=1e-3)
    assert not h5.below_zero(report["comparisons"]["pointstream@video|pooled|wlpips_fill"])


def test_model_size_validator() -> None:
    def size(b: int) -> dict:
        return {"bytes": b}
    row = {"checkpoint_sha256": "a", "checkpoint_bytes": 480, "parameters": 120, "sizes": [size(50), size(110)]}
    result: dict = {"rows": [row, {**row, "checkpoint_sha256": "b"}], "limit_tensors": 0}
    assert all(h5.validate_model_size(result).values())
    result["rows"][1]["sizes"] = [size(300), size(110)]
    assert not h5.validate_model_size(result)["fewer_bits_smaller"]


def test_ladder_parses_heights() -> None:
    assert h5.parse_ladder("1080=41,48;360=63") == [(1080, 41), (1080, 48), (360, 63)]
    assert h5.parse_ladder(None) == []
    with pytest.raises(SystemExit):
        h5.parse_ladder("480=40")


def test_anchors_validator_reads_each_knob_its_way() -> None:
    def row(codec: str, height: int, point: int, kbps: float) -> dict:
        return {"codec": codec, "height": height, "point": point, "kbps": kbps, "decoded_frames": 240,
                "deterministic": True, "scores": scores()}
    rows = [row("svtav1-p1", 360, 48, 90.0), row("svtav1-p1", 360, 63, 30.0),
            row("dcvc-ladder", 540, 0, 20.0), row("dcvc-ladder", 540, 21, 60.0)]
    result: dict = {"rows": rows, "planned": 4, "frames": 240, "jpeg_gate_holds": True}
    assert all(h5.validate_anchors(result).values())
    rows[1]["kbps"] = 120.0  # CRF 63 dearer than CRF 48
    assert not h5.validate_anchors(result)["rate_moves_with_the_knob"]


def test_venue_anchor_is_billed_like_pointstream() -> None:
    anchors = [scored(k, q, codec="dcvc") for k, q in ((100, 0.3), (200, 0.2), (400, 0.1))]
    venue = [{**scored(k, q, codec="dcvc-venue")} for k, q in ((40, 0.3), (80, 0.2), (160, 0.1))]
    rows = [scored(50, 0.3), scored(100, 0.2), scored(200, 0.1)]
    model = {"scene": 1_000_000.0, "venue": 1_000_000.0}
    report = h5.item_report(rows, anchors + venue, 0.0, {"bytes": model, "seconds": {"kitchen": 1e6}})
    free = report["comparisons"]["pointstream|venue|wlpips_fill"]
    paid = report["comparisons"]["pointstream@kitchen|venue|wlpips_fill"]
    assert free["bd"] > 0 and paid["bd"] == pytest.approx(free["bd"], abs=0.02)  # same bits, same seconds
    assert "pointstream@kitchen|pooled|wlpips_fill" in report["comparisons"]


def test_coded_classes_is_the_clients_f_hat() -> None:
    from src.pointstream import mask as mk

    classes = np.zeros((3, 64, 96), np.uint8)
    classes[:, 10:40, 20:70] = 2
    classes[:, 12:30, 22:40] = 1
    classes[1, :, :] = np.roll(classes[1], 3, axis=1)
    shown, payload, record = h5.coded_classes(classes, 8, Fraction(50))
    decoded, scale = mk.decode(payload)
    assert scale == 8 and np.array_equal(shown, mk.upsample(decoded, (96, 64)))
    assert record["lossless_on_reduced_grid"] and record["bytes"] == len(payload)
    assert 0.8 < record["iou"] <= 1 and abs(record["kbps"] - len(payload) * 8 / (3 / 50) / 1000) < 1e-9


def test_foreground_validator_checks_f_hat() -> None:
    mask = {"scale": 8, "lossless_on_reduced_grid": True, "iou": 0.97}
    assert all(h5.validate_foreground(foreground_result(mask_scale=8, mask=mask)).values())
    for change in ({"scale": 16}, {"lossless_on_reduced_grid": False}, {"iou": 0.5}):
        bad = foreground_result(mask_scale=8, mask={**mask, **change})
        assert not h5.validate_foreground(bad)["f_hat_as_the_client_decodes"]


def test_coded_rows_with_a_tag_stay_out_of_the_tag_variants() -> None:
    base = [{**scored(100, 0.2), "foreground": "dcvc-21_4"}]
    coded = [{**scored(150, 0.2), "foreground": "dcvc-21_4", "tag": "h5p4-venue", "mask_scale": 8}]
    v = h5.variant_rows(base + coded)
    assert "h5p4-venue" not in v and [r["kbps"] for r in v["mask8"]] == [150]


def test_match_and_crop_box() -> None:
    m = h5.parse_match("scene-540-0+dcvc-0_4|scene-540-13|540=62")
    assert m == {"pair": "scene-540-0+dcvc-0_4", "foreground": 0, "venue": "scene-540-13", "svt": (540, 62)}
    assert h5.parse_match("enh-1080-63+dcvc-42_1@1|scene-1080-50|1080=55")["foreground"] == 1
    with pytest.raises(SystemExit):
        h5.parse_match("scene-540-0|scene-540-13|500=62")
    f = np.zeros((1080, 1920), bool)
    f[100:150, 1880:1910] = True  # small F near the right edge: widened to CROP_MIN, kept inside the frame
    x0, y0, x1, y1 = h5.crop_box(f)
    assert x1 == 1920 and x1 - x0 == h5.CROP_MIN and y1 - y0 == h5.CROP_MIN and y0 >= 0
    assert x0 <= 1880 and y0 <= 100 and y1 >= 150
    assert h5.crop_box(np.zeros((1080, 1920), bool)) == (768, 348, 1152, 732)


def test_panel_grid_and_crops_validator() -> None:
    rng = np.random.default_rng(0)
    frames = {n: rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for n in ("source", "a", "b", "c")}
    grid = h5.panel_grid(frames, (0, 0, 640, 480))
    assert grid.shape[1] == 4 * h5.PANEL_FULL[0] + 3 * 6 and grid.shape[2] == 3
    row = {"pointstream_kbps": 100.0, "venue_kbps": 90.0, "svt": {"kbps": 110.0, "decoded_frames": 240},
           "pngs": [{"std": 40.0, "shape": list(grid.shape)}] * 2}
    result: dict = {"jpeg_gate_holds": True, "planned": 1, "rows": [row], "shown_frames": [60, 180], "frames": 240}
    assert all(h5.validate_crops(result).values())
    result["rows"][0]["pngs"] = [{"std": 0.0, "shape": list(grid.shape)}] * 2
    assert not h5.validate_crops(result)["panels_are_pictures"]
