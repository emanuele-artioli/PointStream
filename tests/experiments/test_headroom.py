"""Headroom measurement: player cost and panorama cost, with bounds written first."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.components.metrics.bd_rate import RDCurve
from experiments.headroom.measure import (
    FG_MODEST,
    FG_STRONG,
    bg_headroom,
    declared_bounds,
    fg_headroom,
    fg_verdict,
)
from experiments.headroom.real import bbox_slices, opaque_mae, pair_track, paste_crop
from experiments.headroom.remove import (
    court_median_fill,
    flat_fill,
    plate_fill,
    player_fraction,
)
from experiments.headroom.synthetic import handheld_clip, tennis_clip


def test_empty_mask_plate_fill_keeps_the_pixels() -> None:
    frames, _masks = tennis_clip(n_frames=4, seed=2)
    empty = np.zeros(frames.shape[:3], dtype=bool)
    filled, _plate, _h = plate_fill(frames, empty)
    assert filled.shape == frames.shape
    assert np.mean(np.abs(filled.astype(int) - frames.astype(int))) < 8


def test_flat_fill_erases_the_player_region() -> None:
    frames, masks = tennis_clip(n_frames=4)
    filled = flat_fill(frames, masks, value=128)
    assert np.all(filled[masks] == 128)
    assert not np.all(filled[~masks] == 128)


def test_plate_fill_removes_player_stripes() -> None:
    frames, masks = tennis_clip(n_frames=8, seed=0)
    filled, _plate, _h = plate_fill(frames, masks)
    original_std = float(frames[masks].std())
    filled_std = float(filled[masks].std())
    assert original_std > filled_std
    assert player_fraction(masks) > 0.04
    assert player_fraction(masks) < 0.25


def test_bounds_are_the_prewritten_bars() -> None:
    bounds = declared_bounds()
    assert bounds["written_before_measurement"] is True
    assert bounds["fg_strong_saving"] == FG_STRONG == 0.25
    assert bounds["fg_modest_saving"] == FG_MODEST == 0.10
    assert fg_verdict(0.30) == "strong"
    assert fg_verdict(0.15) == "modest"
    assert fg_verdict(0.05) == "weak"


def _stub_encode(frames, *, work_dir, qps, masks=None, label=""):
    """Rate follows spatial gradient energy — what a codec spends bits on."""
    del work_dir, masks
    luma = frames.astype(np.float64).mean(axis=-1)
    grad_y, grad_x = np.gradient(luma, axis=(1, 2))
    energy = float(np.abs(grad_x).mean() + np.abs(grad_y).mean())
    rates = tuple(max(10.0, 5000.0 * energy / qp) for qp in qps)
    qualities = tuple(45.0 - 0.15 * qp for qp in qps)
    return {"curve": RDCurve(rates=rates, qualities=qualities, label=label), "qps": qps}


def test_fg_plate_saving_is_below_flat_saving(tmp_path: Path) -> None:
    """A flat hole overstates the prize; plate inpaint is the honest arm."""
    frames, masks = tennis_clip(n_frames=8, seed=4)
    result = fg_headroom(
        frames, masks, work_dir=tmp_path, encode_curve=_stub_encode, qps=(32, 40)
    )
    plate = result["plate_vs_original"]["saving"]
    flat = result["flat_vs_original"]["saving"]
    assert plate is not None and flat is not None
    assert flat >= plate
    # A gradient stub can give a slightly negative plate saving when warp
    # seams add edges; the real encoder run is the measurement. The stub
    # only has to keep flat from understating relative to plate.
    assert plate <= 1.0
    assert plate > -0.05


def test_fg_null_empty_mask_saves_almost_nothing(tmp_path: Path) -> None:
    frames, _ = tennis_clip(n_frames=6, seed=5)
    empty = np.zeros(frames.shape[:3], dtype=bool)
    result = fg_headroom(
        frames, empty, work_dir=tmp_path, encode_curve=_stub_encode, qps=(32, 40)
    )
    saving = result["plate_vs_original"]["saving"]
    assert saving is not None
    assert abs(saving) < 0.05


def test_bg_panorama_is_orders_cheaper_on_a_static_court() -> None:
    frames, masks = tennis_clip(n_frames=16, seed=0)
    result = bg_headroom(frames, masks, panorama_valid=True)
    assert result["panorama_valid"] is True
    assert result["conventional_over_panorama"] is not None
    assert result["conventional_over_panorama"] >= 10.0
    assert result["orders_of_magnitude"] is True
    assert result["plate_bytes"] > 0
    assert result["homography_bytes"] > 0
    assert result["jpeg_quality_moves_size"] is True


def test_bg_panorama_is_marked_invalid_on_handheld() -> None:
    frames, masks = handheld_clip()
    result = bg_headroom(frames, masks, panorama_valid=False)
    assert result["panorama_valid"] is False
    assert result["orders_of_magnitude"] is False
    assert result["note"]


def test_court_median_fill_paints_masked_pixels_with_unmasked_median() -> None:
    frames = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    frames[:] = (10, 80, 20)
    frames[:, 2:6, 3:7] = (200, 30, 30)
    masks = np.zeros((2, 8, 10), dtype=bool)
    masks[:, 2:6, 3:7] = True
    filled = court_median_fill(frames, masks)
    assert np.array_equal(filled[0, 0, 0], (10, 80, 20))
    assert np.array_equal(filled[1, 0, 0], (10, 80, 20))
    assert np.array_equal(filled[0, 3, 4], (10, 80, 20))
    assert np.array_equal(filled[1, 5, 6], (10, 80, 20))


def test_constructed_crop_pastes_back_with_zero_mae() -> None:
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    frame[:] = (20, 90, 30)
    frame[8:20, 10:18] = (180, 40, 40)
    crop = np.zeros((12, 8, 4), dtype=np.uint8)
    crop[..., :3] = frame[8:20, 10:18]
    crop[..., 3] = 255
    rows, cols = bbox_slices((10, 8, 18, 20), 12, 8, 32, 40)
    assert opaque_mae(frame, crop, rows, cols) == 0.0
    pasted = paste_crop(frame, crop, rows, cols)
    assert np.array_equal(pasted, frame)


def test_shifted_bbox_mae_is_high() -> None:
    frame = np.zeros((32, 40, 3), dtype=np.uint8)
    frame[:] = (20, 90, 30)
    frame[8:20, 10:18] = (180, 40, 40)
    crop = np.zeros((12, 8, 4), dtype=np.uint8)
    crop[..., :3] = frame[8:20, 10:18]
    crop[..., 3] = 255
    rows, cols = bbox_slices((18, 8, 26, 20), 12, 8, 32, 40)
    assert opaque_mae(frame, crop, rows, cols) > 10.0


def test_pair_track_refuses_crop_metadata_length_mismatch(tmp_path: Path) -> None:
    import json

    import cv2

    track = tmp_path / "track_0001"
    track.mkdir()
    cv2.imwrite(str(track / "frame_000000.png"), np.zeros((4, 4, 4), dtype=np.uint8))
    (tmp_path / "track_0001_metadata.json").write_text(
        json.dumps([{"frame_id": 0, "bbox": [0, 0, 4, 4]}, {"frame_id": 1, "bbox": [0, 0, 4, 4]}])
    )
    try:
        pair_track(tmp_path, track)
    except ValueError as exc:
        assert "crops" in str(exc)
        assert "metadata" in str(exc)
    else:
        raise AssertionError("length mismatch must raise")


def test_declared_bounds_include_codec_ranking() -> None:
    bounds = declared_bounds()
    assert bounds["written_before_measurement"] is True
    assert bounds["fg_codec_ranking_prediction"].startswith("AV1")
    assert bounds["player_area_band"] == [0.004, 0.020]
    assert bounds["fg_saving_band_avc"] == [0.184, 0.304]
    assert bounds["fg_saving_band_vvc"] == [0.107, 0.227]
    assert bounds["vvc_gap_expect_survive"] is True
    assert bounds["flat_fill_is_not_an_upper_bound"] is True
    assert bounds["paste_back_mae_opaque_max"] == 2.0


def test_fg_headroom_reports_median_arm(tmp_path: Path) -> None:
    frames, masks = tennis_clip(n_frames=8, seed=4)
    result = fg_headroom(
        frames, masks, work_dir=tmp_path, encode_curve=_stub_encode, qps=(32, 40)
    )
    assert "median_vs_original" in result
    assert result["median_vs_original"]["saving"] is not None


def test_resume_keeps_finished_codec_clip_arms(tmp_path: Path) -> None:
    """Wiping fg on restart is how AVC/HEVC/AV1 got encoded twice."""
    from experiments.headroom.real_ladder import _resume_state

    previous = tmp_path / "report.json"
    previous.write_text(
        json.dumps(
            {
                "fg": {"avc": {"alcaraz_highlights/scene_000": {"ok": True}}},
                "bg_intercoded": {"avc": {"alcaraz_highlights/scene_000": {"ratio": 1.4}}},
                "alarms": ["already raised"],
                "started_at": "before",
            }
        )
    )
    merged = _resume_state(
        previous,
        {
            "fg": {},
            "bg_intercoded": {},
            "nulls": {},
            "tools": {},
            "alarms": [],
            "started_at": "now",
        },
    )
    assert "alcaraz_highlights/scene_000" in merged["fg"]["avc"]
    assert merged["alarms"] == ["already raised"]
    assert merged["started_at"] == "before"


def test_vvc_qp48_is_remapped_not_pretended_to_run() -> None:
    from experiments.headroom.ladder import qps_for_codec

    assert qps_for_codec("avc", (32, 40, 48)) == (32, 40, 48)
    assert qps_for_codec("vvc", (32, 40, 48)) == (32, 40, 46)
    assert qps_for_codec("vvc", (32, 40, 46)) == (32, 40, 46)


def test_vvc_empty_bitstream_steps_qp_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.headroom.ladder import encode_qp_with_vvc_fallback

    source = tmp_path / "source.y4m"
    source.write_bytes(b"y4m")
    notes: list[str] = []
    calls: list[int] = []

    class _Record:
        size_bytes = 12

    def fake_encode(_src, dest, request):
        qp = int(request.rate)
        calls.append(qp)
        if qp >= 47:
            dest.write_bytes(b"")
            raise RuntimeError("command failed (0): empty")
        dest.write_bytes(b"bitstream")
        return _Record()

    monkeypatch.setattr("experiments.headroom.ladder.encode", fake_encode)
    used, dest, record = encode_qp_with_vvc_fallback(
        source, tmp_path, "vvc", 47, notes, "plate"
    )
    assert used == 46
    assert dest.read_bytes() == b"bitstream"
    assert record is not None
    assert calls == [47, 46]
    assert "encoded at QP 46" in notes[0]


def _fake_scenes(video: str) -> list[dict]:
    n = {"a": 3, "b": 3, "c": 2, "d": 2}[video]
    return [
        {"video": video, "scene": f"scene_{i:03d}", "cluster": "cluster_point"}
        for i in range(n)
    ]


def test_chooser_returns_eight_point_scenes_from_four_matches() -> None:
    from experiments.headroom.real import choose_point_scenes

    chosen = choose_point_scenes(
        n=8,
        min_matches=4,
        videos=("a", "b", "c", "d"),
        scene_lister=_fake_scenes,
    )
    assert len(chosen) >= 8
    matches = {row["video"] for row in chosen}
    assert len(matches) >= 4
    # Round-robin: first four are one from each match, not four from `a`.
    assert [row["video"] for row in chosen[:4]] == ["a", "b", "c", "d"]


def test_paste_back_failure_is_recorded_and_the_clip_is_dropped(tmp_path: Path) -> None:
    from experiments.headroom.real import PasteBackError, SceneClip, load_clips_until

    scenes = [
        {"video": "bad", "scene": "scene_000", "cluster": "cluster_point"},
        {"video": "ok_a", "scene": "scene_000", "cluster": "cluster_point"},
        {"video": "ok_b", "scene": "scene_000", "cluster": "cluster_point"},
        {"video": "ok_c", "scene": "scene_000", "cluster": "cluster_point"},
        {"video": "ok_d", "scene": "scene_000", "cluster": "cluster_point"},
        {"video": "ok_a", "scene": "scene_001", "cluster": "cluster_point"},
        {"video": "ok_b", "scene": "scene_001", "cluster": "cluster_point"},
        {"video": "ok_c", "scene": "scene_001", "cluster": "cluster_point"},
        {"video": "ok_d", "scene": "scene_001", "cluster": "cluster_point"},
    ]

    def fake_load(scene, work_dir, n_frames=48):
        del work_dir, n_frames
        if scene["video"] == "bad":
            raise PasteBackError("mae 28.4 exceeds 2.0")
        return SceneClip(
            video=scene["video"],
            scene=scene["scene"],
            video_path=tmp_path / f"{scene['video']}.mp4",
            t_start=0.0,
            t_end=4.0,
            cluster="cluster_point",
            convention="extract_24_frame_id",
            window_start=0,
            n_frames=48,
            frames=np.zeros((2, 4, 4, 3), dtype=np.uint8),
            masks=np.zeros((2, 4, 4), dtype=bool),
            player_area=0.01,
            paste_back={"winner": "extract_24_frame_id", "winner_mae": 0.0},
            ffmpeg={"path": "/opt/local/bin/ffmpeg", "version": "test"},
        )

    survivors, dropped = load_clips_until(
        scenes,
        n=8,
        min_matches=4,
        work_dir=tmp_path,
        n_frames=48,
        load_clip=fake_load,
    )
    assert all(clip.video != "bad" for clip in survivors)
    assert len(survivors) == 8
    assert {clip.video for clip in survivors} >= {"ok_a", "ok_b", "ok_c", "ok_d"}
    assert dropped == [
        {"video": "bad", "scene": "scene_000", "reason": "mae 28.4 exceeds 2.0"}
    ]


def test_common_interval_bd_rate_integrates_only_on_the_overlap() -> None:
    """Two-point curves shifted in quality: slice then BD-rate, hand-computable.

    Anchor: q 20→40, r 1000→100. log10(rate) = 3.0 at 20, 2.0 at 40 (slope −0.05).
    Candidate: q 30→50, r 200→20. log10(rate) = 2.30103 at 30, 1.30103 at 50.
    Common interval [30, 40]:
      anchor at 30: 10**2.5 ≈ 316.227766, at 40: 100
      candidate at 30: 200, at 40: 10**1.80103 ≈ 63.241
    Constant log10 gap ≈ −0.19897 → bd_rate = 10**(gap)−1 ≈ −0.3675, saving ≈ 0.3675.
    """
    from experiments.headroom.measure import (
        common_quality_interval,
        saving_on_interval,
        slice_rd_curve,
    )

    anchor = RDCurve(rates=(1000.0, 100.0), qualities=(20.0, 40.0), label="anchor")
    candidate = RDCurve(rates=(200.0, 20.0), qualities=(30.0, 50.0), label="cand")
    interval = common_quality_interval(anchor, candidate)
    assert interval == pytest.approx((30.0, 40.0))
    sliced_a = slice_rd_curve(anchor, *interval)
    sliced_b = slice_rd_curve(candidate, *interval)
    assert sliced_a.qualities[0] == pytest.approx(30.0)
    assert sliced_a.qualities[-1] == pytest.approx(40.0)
    assert sliced_b.qualities[0] == pytest.approx(30.0)
    assert sliced_b.qualities[-1] == pytest.approx(40.0)
    assert sliced_a.rates[0] == pytest.approx(10**2.5)
    assert sliced_b.rates[0] == pytest.approx(200.0)
    result = saving_on_interval(anchor, candidate, interval)
    assert result["sliced"] is True
    assert result["saving"] == pytest.approx(1.0 - 10 ** (np.log10(200.0) - 2.5), rel=1e-6)
    assert result["interval"] == pytest.approx([30.0, 40.0])
