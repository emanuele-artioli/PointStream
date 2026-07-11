"""Fast unit tests for scripts/eval_checkpoint.py's manifest/scoring logic.

No real checkpoints, GPU, or ffmpeg calls here — those are exercised by the
end-to-end smoke run (see report 10's Phase 5.4 findings entry). These tests
cover: probe-clip path resolution, reference-frame selection, tensor/frame
conversion, metric aggregation, and JSONL log append.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.eval_checkpoint import (
    aggregate_clip_metrics,
    append_jsonl_log,
    clip_color_dir,
    clip_condition_dir,
    clip_frame_paths,
    load_manifest,
    resolve_reference_frame_path,
    rgb01_to_bgr_uint8_frames,
)


def _write_track(root: Path, video: str, scene: str, track: str, frame_ids: list[int]) -> None:
    seg = root / video / "segmentations" / scene
    color_dir = seg / track
    color_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        (color_dir / f"frame_{fid:06d}.png").write_bytes(b"fake")
    skel_dir = seg / f"{track}_skeleton"
    skel_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        (skel_dir / f"frame_{fid:06d}.png").write_bytes(b"fake-skel")


def test_clip_color_dir_and_condition_dir(tmp_path: Path) -> None:
    clip = {"video": "alcaraz_ruud", "scene": "scene_002", "track": "track_0021"}
    assert clip_color_dir(tmp_path, clip) == tmp_path / "alcaraz_ruud" / "segmentations" / "scene_002" / "track_0021"
    assert clip_condition_dir(tmp_path, clip, "skeleton") == (
        tmp_path / "alcaraz_ruud" / "segmentations" / "scene_002" / "track_0021_skeleton"
    )
    assert clip_condition_dir(tmp_path, clip, "canny") == (
        tmp_path / "alcaraz_ruud" / "segmentations" / "scene_002" / "track_0021_canny"
    )


def test_clip_frame_paths_formats_frame_ids(tmp_path: Path) -> None:
    paths = clip_frame_paths(tmp_path, [1, 22, 333])
    assert [p.name for p in paths] == ["frame_000001.png", "frame_000022.png", "frame_000333.png"]


def test_resolve_reference_frame_prefers_frame_outside_window(tmp_path: Path) -> None:
    _write_track(tmp_path, "alcaraz_ruud", "scene_002", "track_0021", list(range(0, 40)))
    clip = {"video": "alcaraz_ruud", "scene": "scene_002", "track": "track_0021", "frame_ids": [10, 11, 12]}
    ref = resolve_reference_frame_path(tmp_path, clip)
    ref_id = int(ref.stem.split("_")[1])
    assert ref_id not in clip["frame_ids"]
    assert ref_id == 0  # earliest frame outside the window


def test_resolve_reference_frame_falls_back_to_window_when_track_equals_window(tmp_path: Path) -> None:
    _write_track(tmp_path, "alcaraz_ruud", "scene_002", "track_0021", [5, 6, 7])
    clip = {"video": "alcaraz_ruud", "scene": "scene_002", "track": "track_0021", "frame_ids": [5, 6, 7]}
    ref = resolve_reference_frame_path(tmp_path, clip)
    ref_id = int(ref.stem.split("_")[1])
    assert ref_id == 5  # falls back to the window's own first frame


def test_rgb01_to_bgr_uint8_frames_shape_and_channel_order() -> None:
    frames = torch.zeros(2, 3, 4, 4)
    frames[:, 0, :, :] = 1.0  # pure red in RGB order
    bgr_frames = rgb01_to_bgr_uint8_frames(frames)
    assert len(bgr_frames) == 2
    assert bgr_frames[0].shape == (4, 4, 3)
    # Red channel (index 0 in RGB) should land at BGR index 2.
    assert bgr_frames[0][0, 0, 2] == 255
    assert bgr_frames[0][0, 0, 0] == 0


def test_aggregate_clip_metrics_averages_and_skips_none() -> None:
    per_clip = [
        {"psnr_mean": 30.0, "ssim_mean": 0.9, "vmaf_mean": 80.0, "fvd": 1.0, "lpips_vgg_uncalibrated": 0.1},
        {"psnr_mean": 32.0, "ssim_mean": 0.92, "vmaf_mean": None, "fvd": 1.5, "lpips_vgg_uncalibrated": None},
    ]
    agg = aggregate_clip_metrics(per_clip)
    assert agg["psnr_mean"] == 31.0
    assert agg["ssim_mean"] == 0.91
    assert agg["vmaf_mean"] == 80.0  # only one non-None value
    assert agg["fvd"] == 1.25
    assert agg["lpips_vgg_uncalibrated"] == 0.1
    assert agg["num_clips_total"] == 2
    assert agg["num_clips_scored"] == 2


def test_aggregate_clip_metrics_all_none_returns_none() -> None:
    per_clip = [{"psnr_mean": None, "ssim_mean": None, "vmaf_mean": None, "fvd": None, "lpips_vgg_uncalibrated": None}]
    agg = aggregate_clip_metrics(per_clip)
    assert all(agg[k] is None for k in ("psnr_mean", "ssim_mean", "vmaf_mean", "fvd", "lpips_vgg_uncalibrated"))
    assert agg["num_clips_scored"] == 0
    assert agg["num_clips_total"] == 1


def test_aggregate_clip_metrics_empty_list() -> None:
    agg = aggregate_clip_metrics([])
    assert agg["num_clips_total"] == 0
    assert agg["num_clips_scored"] == 0


def test_append_jsonl_log_appends_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    append_jsonl_log(log_path, {"step": 1, "psnr_mean": 30.0})
    append_jsonl_log(log_path, {"step": 2, "psnr_mean": 31.0})

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2


def test_load_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = {"schema": "pointstream.probe_set.v1", "probe_clips": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path) == manifest
