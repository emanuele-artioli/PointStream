"""G2 protocol pieces on synthetic data: colour, regions, plate, shadows, rate attribution, acceptable loss."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.background import g1, g2
from src.codecs import quality

ANALYSIS_W, ANALYSIS_H = g2.ANALYSIS_SIZE


def test_rgb_to_yuv420_round_trips_through_the_scoring_view() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    base = rng.integers(30, 220, size=(1, 1, 3))
    yy, xx = np.mgrid[0:64, 0:96]
    rgb = np.clip(base + np.stack([xx, yy, xx + yy], axis=-1) // 4, 0, 255).astype(np.uint8)
    planes = g2.rgb_to_yuv420(rgb, full=True)
    assert planes.shape == (96, 96)
    back = quality.yuv420_to_rgb(torch.from_numpy(planes[None]), 96, 64, full_range=True)[0]
    error = np.abs(back.permute(1, 2, 0).numpy() - rgb.astype(np.float32))
    assert error.mean() < 1.5 and error.max() <= 6


def test_analysis_indices_match_g1() -> None:
    clip = {"fps": 120.0, "first_index": 322, "last_index": 14721, "analysis": {"mode": "rate", "fps": 10.0}}
    assert g2.analysis_indices(clip) == g1.plan(clip, 0)["indices"]


def write_regions(directory: Path, inputs: Path, clip: dict, indices: list[int], scored: list[int]) -> None:
    fill = np.zeros((len(indices), ANALYSIS_H, ANALYSIS_W), bool)
    for k in range(len(indices)):
        fill[k, 200:260, 100 + 40 * k:140 + 40 * k] = True
    shadow = np.zeros((len(scored), ANALYSIS_H, ANALYSIS_W), bool)
    shadow[:, 262:280, 100:160] = True
    shadow[:, 120:140, 140:160] = True  # inside the umpire's zone: must not count as shadow
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "regions.npz", indices=np.array(indices), scored=np.array(scored),
                        fill=np.packbits(fill.reshape(len(indices), -1), axis=1),
                        shadow=np.packbits(shadow.reshape(len(scored), -1), axis=1))
    labels = inputs / "labels" / g2.safe(clip["id"])
    labels.mkdir(parents=True)
    (labels / "ball.json").write_text(json.dumps({"16": [640.0, 600.0]}))


def test_regions_partition_the_visible_background(tmp_path: Path) -> None:
    clip = {"id": "tracknet/game10/Clip1"}
    write_regions(tmp_path / "r", tmp_path / "in", clip, [12, 15, 18], [15])
    regions = g2.Regions(clip, tmp_path / "r", tmp_path / "in")
    masks = regions.masks(15)
    visible = masks["V"]
    parts = [masks[k] for k in ("P", "C", "G", "S")]
    assert np.array_equal(np.logical_or.reduce(parts), visible)
    assert np.sum([p.astype(int) for p in parts], axis=0).max() == 1
    assert masks["S"].any() and not masks["S"][165:185, 190:210].any() and masks["C"][165:185, 190:210].all()
    assert masks["G"][90, 200]  # the score box
    assert not visible[230 * 4 // 3, 220]  # the analysis-frame foreground, upscaled
    # Between analysis frames the fill is the union of both neighbours, plus the ball when labelled.
    between = regions.fill_mask(16)
    for near in (15, 18):
        assert between[g2.upscale(regions.fill_analysis(near), 1280, 720)].all()
    assert between[600, 640] and not regions.foreground(15)[600, 640]


def test_median_plate_ignores_the_foreground_and_inpaints_the_unseen() -> None:
    width, height, n = 32, 16, 5
    frames = np.full((n, height * 3 // 2, width), 100, np.uint8)
    masks = np.zeros((n, height, width), bool)
    for k in range(n):
        frames[k, 4:8, 4 * k:4 * k + 4] = 250
        masks[k, 4:8, 4 * k:4 * k + 4] = True
    masks[:, 0:2, 28:32] = True  # never seen
    frames[:, 0:2, 28:32] = 0
    plate, unseen = g2.median_plate(frames, masks, width, height, True)
    assert unseen == pytest.approx(8 / (width * height))
    seen = ~masks.all(axis=0)
    assert np.all(plate[:height][seen] == 100)
    assert np.abs(plate[:height][~seen].astype(int) - 100).max() <= 3  # inpainted
    assert np.all(np.abs(plate[height:].astype(int) - 100) <= 3)


def test_shadow_rule() -> None:
    plate = (np.full((ANALYSIS_H, ANALYSIS_W), 150, np.float32), np.full((ANALYSIS_H, ANALYSIS_W), 128, np.float32),
             np.full((ANALYSIS_H, ANALYSIS_W), 128, np.float32))
    frame = tuple(p.copy() for p in plate)
    player = np.zeros((ANALYSIS_H, ANALYSIS_W), bool)
    player[100:200, 300:340] = True
    frame[0][200:220, 300:360] = 90  # darker, same chroma, at the feet: shadow
    frame[0][400:420, 700:760] = 90  # far from any player: not a shadow
    frame[0][220:230, 300:360] = 20  # too dark to be a shadow
    shadow = g2.shadow_mask(frame, plate, [{"mask": player, "box": (300, 100, 340, 200)}], player, True)
    assert shadow[205:215, 305:355].all()
    assert not shadow[400:420, 700:760].any()
    assert not shadow[222:228, 300:360].any()


def test_dcvc_bytes_are_spread_over_the_frames_of_each_nal() -> None:
    report = {"nals": [{"nal_bytes": 1000, "sps_bytes": 10, "display_frames": 1},
                       {"nal_bytes": 400, "sps_bytes": 0, "display_frames": 4}]}
    assert g2.frame_bytes_dcvc(report, 5) == [1010, 100, 100, 100, 100]
    with pytest.raises(RuntimeError):
        g2.frame_bytes_dcvc(report, 6)


def region(sse: float, pixels: int, lpips: float) -> dict:
    return {"sse": sse, "pixels": pixels, "psnr": quality.psnr(sse / (3 * pixels)) if pixels else None,
            "lpips": lpips if pixels else None}


def test_excess_is_zero_when_crowd_errs_like_plain_and_counts_the_extra() -> None:
    plain, crowd = region(3000.0, 1000, 0.1), region(300.0, 100, 0.1)
    frame = {"P": plain, "C": crowd, "G": region(0, 0, 0), "S": region(0, 0, 0), "V": region(3300.0, 1100, 0.1),
             "frame": region(4000.0, 1200, 0.1)}
    assert g2.excess([frame])["crowd_graphics"]["psnr_v_db"] == pytest.approx(0.0, abs=1e-9)
    loud = dict(frame, C=region(3300.0, 100, 0.4), V=region(6300.0, 1100, 0.127))
    terms = g2.excess([loud])["crowd_graphics"]
    assert terms["psnr_v_db"] == pytest.approx(10 * math.log10(6300 / 3300))
    assert terms["weighted_psnr_db"] == pytest.approx(0.3 * terms["psnr_v_db"])
    assert terms["lpips_frame"] == pytest.approx((0.4 - 0.1) * 100 / 1200)
    assert not terms["acceptable"]
    assert g2.excess([frame])["shadows"] is None


def test_choose_points_follows_the_rule() -> None:
    svt = [(13.0, 3000.0, 46.0), (20.0, 1500.0, 44.0), (27.0, 800.0, 42.0), (34.0, 400.0, 40.0),
           (41.0, 200.0, 38.0), (48.0, 100.0, 36.0), (55.0, 50.0, 34.0), (62.0, 25.0, 32.0)]
    dcvc = [(float(q), 20.0 * 2 ** (k / 1.5), 33.0 + 1.6 * k) for k, q in enumerate(range(0, 72, 9))]
    out = g2.choose_points({"svtav1": svt, "dcvc": dcvc})
    low, high = out["range_db"]
    assert low == pytest.approx(33.0) and high == pytest.approx(44.2)
    assert len(out["points"]["svtav1"]) >= 6 and not out["refine"]
    assert 62.0 in out["points"]["svtav1"]  # the next lower-rate point below the range


def test_scoring_composite_changes_no_visible_pixel() -> None:
    width, height = 8, 4
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 256, size=(height * 3 // 2, width), dtype=np.uint8)
    output = rng.integers(0, 256, size=(height * 3 // 2, width), dtype=np.uint8)
    visible = np.ones((height, width), bool)
    visible[0:2, 0:3] = False  # one whole chroma block (0:2, 0:2) and half of the next
    out = g2.with_source_foreground(reference, output, visible)
    assert np.array_equal(out[:height][visible], output[:height][visible])
    assert np.array_equal(out[:height][~visible], reference[:height][~visible])
    u_out, u_ref, u_dec = (a[height:height + height // 4].reshape(2, 4) for a in (out, reference, output))
    assert u_out[0, 0] == u_ref[0, 0] and u_out[0, 1] == u_dec[0, 1]
