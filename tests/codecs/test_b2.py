from __future__ import annotations

import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.visor import b2
from src.codecs import quality, svtav1


def ivf(payloads: list[bytes]) -> bytes:
    header = b"DKIF" + struct.pack("<HHIHHIIII", 0, 32, 0x31305641, 64, 48, 50, 1, len(payloads), 0)
    body = b"".join(struct.pack("<IQ", len(p), n) + p for n, p in enumerate(payloads))
    return header + body


def test_ivf_payload_excludes_framing_and_rejects_truncation() -> None:
    data = ivf([b"a" * 10, b"b" * 3])
    assert svtav1.ivf_frames(data) == [10, 3]
    with pytest.raises(ValueError):
        svtav1.ivf_frames(data[:-1])
    with pytest.raises(ValueError):
        svtav1.ivf_frames(b"RIFF" + data[4:])


def test_encode_command_is_single_keyframe_crf_on_raw_planes(tmp_path: Path) -> None:
    command = svtav1.encode_command("SvtAv1EncApp", tmp_path / "s.yuv", tmp_path / "o.ivf", width=1920, height=1080,
                                    fps=Fraction(60000, 1001), frames=240, crf=35, preset=4, full_range=True)
    joined = " ".join(command)
    for part in ("--keyint -1", "--crf 35", "--preset 4", "--fps-num 60000", "--fps-denom 1001", "-n 240",
                 "--color-range 1", "--input-depth 8", "--rc 0", "--lp 6"):
        assert part in joined


def test_psnr_cap_and_value() -> None:
    assert quality.psnr(0.0) == quality.PSNR_CAP
    assert quality.psnr(255.0**2) == pytest.approx(0.0)
    assert quality.psnr(1.0) == pytest.approx(48.1308, abs=1e-3)


def yuv(frames: int, width: int, height: int, y: int = 128, u: int = 128, v: int = 128) -> np.ndarray:
    out = np.empty((frames, height * 3 // 2, width), np.uint8)
    out[:, :height] = y
    chroma = out[:, height:].reshape(frames, 2, height // 2, width // 2)
    chroma[:, 0], chroma[:, 1] = u, v
    return out


def test_rgb_view_of_neutral_and_coloured_planes() -> None:
    torch = pytest.importorskip("torch")
    grey = quality.yuv420_to_rgb(torch.from_numpy(yuv(1, 8, 4)), 8, 4, full_range=True)
    assert grey.shape == (1, 3, 4, 8) and bool((grey == 128).all())
    limited_black = quality.yuv420_to_rgb(torch.from_numpy(yuv(1, 8, 4, y=16)), 8, 4, full_range=False)
    assert bool((limited_black == 0).all())
    red = quality.yuv420_to_rgb(torch.from_numpy(yuv(1, 8, 4, y=54, u=99, v=255)), 8, 4, full_range=True)
    assert red[0, 0, 0, 0] > 240 and red[0, 1, 0, 0] < 20 and red[0, 2, 0, 0] < 20


def random_backbone(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("torchmetrics")
    torch.manual_seed(0)
    path = tmp_path / "alexnet.pth"
    torch.save(torchvision.models.alexnet(weights=None).state_dict(), path)
    return path


def test_scores_split_by_region_and_sum_exactly(tmp_path: Path) -> None:
    width, height, frames = 192, 176, 3
    rng = np.random.default_rng(0)
    ref = rng.integers(40, 200, size=(frames, height * 3 // 2, width), dtype=np.uint8)
    dec = ref.copy()
    dec[:, : height // 2, :] += 2  # luma error in the top half only
    ref_path, dec_path = tmp_path / "ref.yuv", tmp_path / "dec.yuv"
    ref.tofile(ref_path)
    dec.tofile(dec_path)
    fg = np.zeros((height, width), bool)
    fg[: height // 2] = True
    regions = [{"s/fg": fg, "s/bg": ~fg} for _ in range(frames)]
    net = quality.load_lpips(random_backbone(tmp_path), "cpu")
    rows = quality.score(ref_path, dec_path, regions, width=width, height=height, frames=frames, full_range=True,
                         device="cpu", lpips_net=net, batch=2)
    assert len(rows) == frames
    for row in rows:
        assert row["s/bg"]["sse"] == 0 and row["s/bg"]["psnr"] == quality.PSNR_CAP
        assert row["s/fg"]["sse"] > 0 and row["s/fg"]["sse"] == row["frame"]["sse"]
        assert row["frame"]["pixels"] == row["s/fg"]["pixels"] + row["s/bg"]["pixels"]
        assert row["planes"]["psnr_y"] == pytest.approx(quality.psnr(4 * 0.5), abs=1e-9)
        assert row["planes"]["psnr_u"] == quality.PSNR_CAP
        assert row["s/fg"]["lpips"] > row["s/bg"]["lpips"] >= 0.0
        assert 0.0 < row["ms_ssim"] <= 1.0
    same = quality.score(ref_path, ref_path, regions, width=width, height=height, frames=frames, full_range=True,
                         device="cpu", lpips_net=net)
    assert all(r["frame"]["lpips"] == pytest.approx(0.0, abs=1e-6) and r["frame"]["sse"] == 0 for r in same)


def test_decode_window_writes_native_planes_and_checks_jpegs(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    from PIL import Image

    video = tmp_path / "P99_101.mp4"
    width, height = 64, 48
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=50)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"qscale": "1", "g": "4"}
        for n in range(30):
            image = np.full((height, width, 3), 20 + 7 * n, np.uint8)
            image[:, : n + 1] = 250
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    archive = tmp_path / "archive"
    (archive / "rgb_frames").mkdir(parents=True)
    from src.segmentation import visor

    decoded = dict(visor.decode_frames(video, [11, 13]))
    # On a 50 fps video EPIC rgb frame k is decoded frame k - 1.
    Image.fromarray(decoded[11]).save(archive / "rgb_frames" / "P99_101_frame_0000000012.jpg", quality=98)
    Image.fromarray(decoded[13]).save(archive / "rgb_frames" / "P99_101_frame_0000000099.jpg", quality=98)
    (archive / "frame_mapping.json").write_text(json.dumps({"P99_101": {
        "P99_101_frame_0000000012.jpg": "frame_0000000012.jpg",
        "P99_101_frame_0000000099.jpg": "frame_0000000013.jpg",  # names frame 12: the wrong image
    }}))
    item = {"id": "P99_101_x", "video": "P99_101", "fps": 50.0, "first_video_index": 8,
            "sparse_jpegs": ["rgb_frames/P99_101_frame_0000000012.jpg", "rgb_frames/P99_101_frame_0000000099.jpg"]}
    out = tmp_path / "source.yuv"
    record = b2.decode_window(video, item, 10, archive, out, threads=1)
    assert out.stat().st_size == 10 * width * height * 3 // 2
    assert record["pix_fmt"] == ["yuv420p"] and record["width"] == width and record["height"] == height
    planes = np.fromfile(out, np.uint8).reshape(10, height * 3 // 2, width)
    with av.open(str(video)) as container:
        frames = [f.to_ndarray() for f in container.decode(video=0)]
    assert np.array_equal(planes, np.stack(frames[8:18]))
    gate = {g["jpeg"]: g for g in record["jpeg_gate"]}
    assert gate["rgb_frames/P99_101_frame_0000000012.jpg"]["holds"]
    wrong = gate["rgb_frames/P99_101_frame_0000000099.jpg"]
    assert not wrong["holds"] and wrong["best_index"] == 13 and wrong["index"] == 12


def frame_row(fg_psnr: float, bg_psnr: float, fg_px: int = 10, bg_px: int = 30) -> dict[str, Any]:
    def region(p: float, n: int) -> dict[str, Any]:
        mse = 255.0**2 / 10 ** (p / 10)
        return {"sse": int(round(mse * 3 * n)), "pixels": n, "psnr": p, "lpips": 0.1}
    return {"frame": region((fg_psnr + bg_psnr) / 2, fg_px + bg_px), "v/fg": region(fg_psnr, fg_px),
            "v/bg": region(bg_psnr, bg_px), "planes": {"psnr_y": 40.0, "psnr_u": 45.0, "psnr_v": 45.0, "psnr_yuv": 41.25},
            "ms_ssim": 0.98}


def test_weighted_psnr_is_per_frame_db_weighting() -> None:
    summary = b2.summarize([frame_row(30, 40), frame_row(34, 40)], [90.0, 92.0], ["v"])
    ms = summary["mask_sets"]["v"]
    assert ms["wpsnr"] == pytest.approx(0.7 * 32 + 0.3 * 40)
    assert ms["psnr_fg"] == pytest.approx(32) and ms["frames_scored"] == 2
    assert ms["foreground_fraction"] == pytest.approx(0.25)
    assert summary["frame"]["vmaf"] == pytest.approx(91.0)
    # A frame without foreground is not weighted.
    empty = frame_row(30, 40, fg_px=0)
    empty["v/fg"]["psnr"] = None
    assert b2.summarize([empty, frame_row(30, 40)], [90, 90], ["v"])["mask_sets"]["v"]["frames_scored"] == 1


def test_bd_rate_of_identical_and_halved_curves() -> None:
    rate = [100.0, 200.0, 400.0, 800.0]
    psnr = [30.0, 33.0, 36.0, 39.0]
    assert b2.bd_rate(rate, psnr, rate, psnr) == pytest.approx(0.0, abs=1e-9)
    assert b2.bd_rate(rate, psnr, [r / 2 for r in rate], psnr) == pytest.approx(-50.0, abs=1e-6)
    assert b2.bd_rate(rate, psnr, rate, [q + 20 for q in psnr]) is None  # no common quality range


def test_curves_average_items_per_video_type() -> None:
    def item(item_id: str, kind: str, kbps: float) -> dict[str, Any]:
        summary = b2.summarize([frame_row(30, 40)], [90.0], ["v"])
        return {"id": item_id, "video_type": kind, "mask_sets": {"v": {"missing_objects": {"mean_share": 0.5}}},
                "points": [{"point": 35.0, "kbps": kbps, "bpp": 0.01, "summary": summary}]}
    out = b2.curves([item("a", "EK-100", 100), item("b", "EK-55", 300)], ["v"])
    assert out["all"]["v"][0]["kbps"] == 200 and out["all"]["v"][0]["items"] == 2
    assert out["EK-55"]["v"][0]["kbps"] == 300 and out["EK-100"]["v"][0]["missing_object_share"] == 0.5


def test_rate_points_cover_the_common_range_below_the_cap() -> None:
    av1 = [(62, 400, 34.0), (55, 900, 36.5), (48, 1400, 37.5), (41, 2200, 38.5), (34, 4200, 39.8), (27, 8600, 41.2)]
    dcvc = [(9, 200, 31.0), (18, 350, 33.0), (27, 600, 35.0), (36, 1000, 36.8), (45, 1700, 38.0), (54, 3000, 39.6), (63, 6000, 41.0)]
    out = b2.choose_points({"svtav1": [(float(a), float(b), c) for a, b, c in av1],
                            "dcvc": [(float(a), float(b), c) for a, b, c in dcvc]})
    assert out["range_db"] == [34.0, 38.5]
    # SVT-AV1 has nothing below the range; DCVC-UF adds its point just below it (QP 18).
    assert out["points"]["svtav1"] == [41.0, 48.0, 55.0, 62.0]
    assert 18.0 in out["points"]["dcvc"] and 54.0 not in out["points"]["dcvc"]
    assert out["refine_first"] is False


def test_variants_label_only_what_differs_from_b2() -> None:
    import argparse

    args = argparse.Namespace(codec="svtav1", preset="4", roi_offset="0,-16", enable_tf=None)
    assert [v["label"] for v in b2.variants(args)] == ["", "roi-16"]
    args = argparse.Namespace(codec="svtav1", preset="2,4", roi_offset="0", enable_tf=0)
    assert [v["label"] for v in b2.variants(args)] == ["p2-tf0", "p4-tf0"]
    args = argparse.Namespace(codec="dcvc", dcvc_range="full,limited")
    assert [v["label"] for v in b2.variants(args)] == ["", "range-limited"]


def test_roi_map_marks_blocks_touching_the_foreground(tmp_path: Path) -> None:
    width, height = 130, 70  # 3 x 2 blocks of 64, the last ones partial
    fg = np.zeros((height, width), bool)
    fg[65, 129] = True  # bottom-right block only
    record = b2.write_roi_map([{"m/fg": fg}, {"m/fg": np.zeros_like(fg)}], "m", 2, width, height, -16, tmp_path / "roi.txt")
    lines = (tmp_path / "roi.txt").read_text().splitlines()
    assert lines == ["0 0 0 0 0 0 -16", "1 0 0 0 0 0 0"]
    assert record["grid"] == [2, 3] and record["block_share_mean"] == pytest.approx(1 / 12, abs=1e-4)


def test_range_conversion_round_trips_within_one_level(tmp_path: Path) -> None:
    width, height, frames = 16, 8, 3
    data = np.random.default_rng(1).integers(0, 256, size=(frames, height * 3 // 2, width), dtype=np.uint8)
    full, limited, back = tmp_path / "f.yuv", tmp_path / "l.yuv", tmp_path / "b.yuv"
    data.tofile(full)
    b2.convert_range(full, limited, frames, width, height, to_limited=True)
    narrow = np.fromfile(limited, np.uint8).reshape(data.shape)
    assert narrow[:, :height].min() >= 16 and narrow[:, :height].max() <= 235
    assert narrow[:, height:].min() >= 16 and narrow[:, height:].max() <= 240
    b2.convert_range(limited, back, frames, width, height, to_limited=False)
    assert np.abs(np.fromfile(back, np.uint8).reshape(data.shape).astype(int) - data).max() <= 1


def test_spearman_ranks() -> None:
    assert b2.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert b2.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert b2.spearman([1, 2], [1, 2]) is None


def test_positive_roi_offset_raises_the_background_blocks(tmp_path: Path) -> None:
    width, height = 130, 70
    fg = np.zeros((height, width), bool)
    fg[65, 129] = True
    b2.write_roi_map([{"m/fg": fg}], "m", 1, width, height, 32, tmp_path / "roi.txt")
    assert (tmp_path / "roi.txt").read_text().splitlines() == ["0 32 32 32 32 32 0"]


def test_stream_root_accepts_an_extracted_published_tar(tmp_path: Path) -> None:
    from experiments.visor.b2 import stream_root

    assert stream_root(tmp_path) == tmp_path
    (tmp_path / "publish" / "streams" / "dcvc").mkdir(parents=True)
    assert stream_root(tmp_path) == tmp_path / "publish" / "streams"


def test_stream_record_finds_the_encoding_jobs_hash(tmp_path: Path) -> None:
    import argparse

    record = tmp_path / "b2.json"
    record.write_text(json.dumps({"items": [{"id": "I", "points": [
        {"variant": "", "point": 41, "stream_sha256": "aa"}, {"point": 48, "stream_sha256": "bb"}]}]}))
    args = argparse.Namespace(stream_record=str(record))
    assert b2.stream_record(args, "I", "", 41.0) == "aa"
    assert b2.stream_record(args, "I", "", 48) == "bb"
    assert b2.stream_record(args, "J", "", 41) is None
    assert b2.stream_record(argparse.Namespace(stream_record=None), "I", "", 41) is None
