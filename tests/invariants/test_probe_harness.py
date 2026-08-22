"""Probe harness behaviour and plausible misuse. No GPU, no real weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from experiments.probe.bounds import (
    OBJECT_PSNR_ALARM_HIGH_DB,
    OBJECT_PSNR_ALARM_LOW_DB,
    judge_object_psnr,
)
from experiments.probe.clips import list_clips, load_frame
from experiments.probe.engines import plan_for
from experiments.probe.run import drive_engine
from experiments.probe.score import score_generation
from experiments.probe_set.schema import SCHEMA_ID, TRAINING_SPLIT_VIDEOS
from src.contracts.conditioning import ConditioningBundle, GenerationParams


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _tiny_probe_set(tmp_path: Path, *, n_frames: int = 3) -> Path:
    root = tmp_path / "probe_set"
    video = TRAINING_SPLIT_VIDEOS[0]
    scene = "scene_001"
    track = "track_0001"
    crop_dir = root / "clips" / video / scene / track
    skel_dir = root / "clips" / video / scene / f"{track}_skeleton"
    canny_dir = root / "clips" / video / scene / f"{track}_canny"
    h, w = 32, 16
    for index in range(n_frames):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[4:28, 4:12, :3] = (40 + index * 10, 80, 120)
        rgba[4:28, 4:12, 3] = 255
        pose = np.zeros((h, w, 3), dtype=np.uint8)
        pose[6:26, 7:9] = 255
        canny = np.zeros((h, w), dtype=np.uint8)
        canny[6:26, 7:9] = 255
        name = f"frame_{index:06d}.png"
        _write_png(crop_dir / name, rgba)
        _write_png(skel_dir / name, pose)
        _write_png(canny_dir / name, canny)
    manifest = {
        "schema": SCHEMA_ID,
        "probe_clips": [
            {
                "video": video,
                "scene": scene,
                "track": track,
                "key": f"{video}/{scene}/{track}",
                "path": f"clips/{video}/{scene}/{track}",
                "num_frames": n_frames,
                "frame_ids": list(range(n_frames)),
            }
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_bounds_were_written_before_any_probe_number() -> None:
    """The alarm gates are the ones the brief named, not fitted after the run."""
    assert OBJECT_PSNR_ALARM_LOW_DB == 10.0
    assert OBJECT_PSNR_ALARM_HIGH_DB == 35.0
    low = judge_object_psnr("pose-controlnet", 4.0)
    high = judge_object_psnr("pose-controlnet", 40.0)
    floor = judge_object_psnr("ip-adapter-controlnet", 11.0)
    assert low.status == "alarm-low"
    assert high.status == "alarm-high"
    assert floor.status == "known-floor"


def test_unknown_engine_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="unknown probe engine"):
        plan_for("definitely-not-wired")


def test_clip_loader_pairs_channels_by_index(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path)
    clips = list_clips(root)
    assert len(clips) == 1
    frame = load_frame(clips[0], 1)
    assert frame.frame_index == 1
    assert frame.appearance_rgb.shape == (32, 16, 3)
    assert frame.pose_rgb.shape[:2] == (32, 16)
    assert frame.object_mask.shape == (32, 16)
    assert int(frame.object_mask.sum()) > 0
    assert frame.motion_field.shape[0] == 2


def test_frame_index_out_of_range_is_an_index_error(tmp_path: Path) -> None:
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=2))
    with pytest.raises(IndexError, match="out of range"):
        load_frame(clips[0], 2)


def test_optical_flow_accepts_a_neighbor_with_a_different_crop_size(tmp_path: Path) -> None:
    """Player crops change size frame to frame; Farneback cannot see that."""
    root = _tiny_probe_set(tmp_path, n_frames=2)
    crop_dir = root / "clips" / TRAINING_SPLIT_VIDEOS[0] / "scene_001" / "track_0001"
    other = np.zeros((40, 20, 4), dtype=np.uint8)
    other[5:35, 5:15, :3] = 90
    other[5:35, 5:15, 3] = 255
    _write_png(crop_dir / "frame_000001.png", other)
    frame = load_frame(list_clips(root)[0], 0)
    assert frame.motion_field.shape == (2, 32, 16)


def test_mismatched_channel_counts_are_refused(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=2)
    extra = root / "clips" / TRAINING_SPLIT_VIDEOS[0] / "scene_001" / "track_0001_skeleton"
    _write_png(extra / "frame_000002.png", np.zeros((32, 16, 3), dtype=np.uint8))
    clips = list_clips(root)
    with pytest.raises(ValueError, match="paired by position"):
        load_frame(clips[0], 0)


def test_object_psnr_is_worse_than_frame_when_only_the_player_moves() -> None:
    """The §6.4 shape on a canvas: pad matches, object does not."""
    from src.components.generation._numpy import as_hwc, prepare_letterboxed

    appearance = np.zeros((32, 16, 3), dtype=np.uint8)
    appearance[4:28, 4:12] = 200
    mask = np.zeros((32, 16), dtype=bool)
    mask[4:28, 4:12] = True
    predicted_crop = appearance.copy()
    predicted_crop[4:28, 4:12] = 40
    predicted = as_hwc(prepare_letterboxed(predicted_crop, None, 32, 32)["appearance"])
    score = score_generation(
        appearance, predicted, object_mask=mask, canvas_width=32, canvas_height=32
    )
    assert score.differs_from_input
    assert score.frame_psnr_db > score.object_psnr_db
    assert np.isfinite(score.object_psnr_db)


def test_identical_output_is_flagged() -> None:
    appearance = np.full((16, 16, 3), 80, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    score = score_generation(
        appearance, appearance, object_mask=mask, canvas_width=16, canvas_height=16
    )
    assert score.differs_from_input is False
    assert np.isinf(score.object_psnr_db)


def test_drive_engine_records_a_construct_refusal(tmp_path: Path) -> None:
    from experiments.probe.engines import EnginePlan

    plan = EnginePlan(
        name="mofa-video",
        kind="refuse-construct",
        frame_indices=(),
        refuse_at="construct",
        notes="test",
    )
    result = drive_engine(plan, (), device="cpu", seed=0, out_dir=tmp_path / "out")
    assert result.refused
    assert result.refuse_reason is not None
    assert "Stability" in result.refuse_reason or "licence" in result.refuse_reason.lower()


class _CopyPipe:
    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> np.ndarray:
        del seed, device
        from src.components.generation._numpy import as_chw, prepare_letterboxed

        width = params.width if params.width is not None else 512
        height = params.height if params.height is not None else 512
        prepared = prepare_letterboxed(conditioning.appearance, None, width, height)
        return as_chw(prepared["appearance"])


class _ShiftPipe:
    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> np.ndarray:
        del seed, device
        from src.components.generation._numpy import as_chw, prepare_letterboxed

        width = params.width if params.width is not None else 512
        height = params.height if params.height is not None else 512
        prepared = prepare_letterboxed(conditioning.appearance, None, width, height)
        appearance = np.asarray(prepared["appearance"]).copy()
        appearance = np.clip(appearance.astype(np.int16) + 25, 0, 255).astype(np.uint8)
        return as_chw(appearance)


def test_drive_engine_records_identity_instead_of_silently_passing(tmp_path: Path) -> None:
    from experiments.probe.engines import EnginePlan

    root = _tiny_probe_set(tmp_path, n_frames=3)
    clips = list_clips(root)
    plan = EnginePlan(
        name="upscale-refine",
        kind="one-pass",
        frame_indices=(1,),
        notes="identity misuse",
    )
    result = drive_engine(
        plan,
        clips,
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        generator=_CopyPipe(),
    )
    assert result.clips[0].differs_from_input is False
    assert result.clips[0].error is not None
    assert "identical" in result.clips[0].error


def test_drive_engine_scores_a_fake_generator(tmp_path: Path) -> None:
    from experiments.probe.engines import EnginePlan

    root = _tiny_probe_set(tmp_path, n_frames=3)
    clips = list_clips(root)
    plan = EnginePlan(
        name="upscale-refine",
        kind="one-pass",
        frame_indices=(1,),
        notes="fake",
    )
    result = drive_engine(
        plan,
        clips,
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        generator=_ShiftPipe(),
    )
    assert result.refused is False
    assert result.clips[0].differs_from_input is True
    assert result.clips[0].object_psnr_db is not None
    assert result.clips[0].error is None
    dumped = json.loads((tmp_path / "out" / "upscale-refine.json").read_text())
    assert dumped["headline"]["n"] == 1
