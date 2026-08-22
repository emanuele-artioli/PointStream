"""Probe harness: coding-task behaviour and plausible misuse. No GPU, no real weights.

Behaviour the caller relies on: appearance and reference are different frames,
static copy always runs, at-or-below that floor is labelled "not using
appearance", self-reconstruction is recorded and never used for ranking,
the keyframe/target offset is in the output, and more than one offset is
scored per clip.

Plausible misuse is the silent kind — pairing channels by reconstructing a
filename, scoring the keyframe against itself, ranking on the diagnostic
PSNR, treating a pasted keyframe as a low-ranking engine.

Deliberately not tested: live probe-set PNG contents, argparse help, real
generator weights, VMAF/LPIPS, Wave-2 self-reconstruction numbers as a
regression target, construct.py's SAM3 leftover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from experiments.probe.bounds import (
    NOT_USING_APPEARANCE,
    STATIC_COPY_EXPECTED_HIGH_DB,
    STATIC_COPY_EXPECTED_LOW_DB,
    appearance_use_label,
    judge_static_copy_object_psnr,
)
from experiments.probe.clips import list_clips, load_coding_sample, load_frame
from experiments.probe.engines import STATIC_COPY, plan_for
from experiments.probe.run import drive_all, drive_engine, rank_engines
from experiments.probe.score import score_generation
from experiments.probe_set.schema import SCHEMA_ID, TRAINING_SPLIT_VIDEOS
from src.contracts.conditioning import ConditioningBundle, GenerationParams


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _tiny_probe_set(
    tmp_path: Path,
    *,
    n_frames: int = 5,
    crop_ids: list[int] | None = None,
    skeleton_ids: list[int] | None = None,
) -> Path:
    root = tmp_path / "probe_set"
    video = TRAINING_SPLIT_VIDEOS[0]
    scene = "scene_001"
    track = "track_0001"
    crop_dir = root / "clips" / video / scene / track
    skel_dir = root / "clips" / video / scene / f"{track}_skeleton"
    canny_dir = root / "clips" / video / scene / f"{track}_canny"
    h, w = 32, 16
    crop_ids = list(range(n_frames)) if crop_ids is None else crop_ids
    skeleton_ids = list(range(n_frames)) if skeleton_ids is None else skeleton_ids
    if len(crop_ids) != n_frames or len(skeleton_ids) != n_frames:
        raise AssertionError("fixture ids must match n_frames")
    for index in range(n_frames):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[4:28, 4:12, :3] = (40 + index * 10, 80, 120)
        rgba[4:28, 4:12, 3] = 255
        pose = np.zeros((h, w, 3), dtype=np.uint8)
        pose[6:26, 7:9] = 255
        pose[0, 0] = index + 1
        canny = np.zeros((h, w), dtype=np.uint8)
        canny[6:26, 7:9] = 255
        _write_png(crop_dir / f"frame_{crop_ids[index]:06d}.png", rgba)
        _write_png(skel_dir / f"frame_{skeleton_ids[index]:06d}.png", pose)
        _write_png(canny_dir / f"frame_{crop_ids[index]:06d}.png", canny)
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
                "frame_ids": crop_ids,
            }
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_bounds_are_anchored_on_the_static_copy_floor() -> None:
    assert STATIC_COPY_EXPECTED_LOW_DB == 8.0
    assert STATIC_COPY_EXPECTED_HIGH_DB == 16.0
    assert judge_static_copy_object_psnr(11.82).status == "expected"
    assert appearance_use_label(11.01, 11.82) == NOT_USING_APPEARANCE
    assert appearance_use_label(11.82, 11.82) == NOT_USING_APPEARANCE
    assert appearance_use_label(13.0, 11.82) == "beats floor"


def test_unknown_engine_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="unknown probe engine"):
        plan_for("definitely-not-wired")


def test_appearance_and_reference_frames_differ(tmp_path: Path) -> None:
    clips = list_clips(_tiny_probe_set(tmp_path))
    sample = load_coding_sample(clips[0], 0, 2)
    assert sample.appearance_frame_index == 0
    assert sample.target_frame_index == 2
    assert sample.offset == 2
    assert not np.array_equal(sample.appearance_rgb, sample.reference_rgb)
    assert int(sample.pose_rgb[0, 0, 0]) == 3


def test_zero_offset_is_refused(tmp_path: Path) -> None:
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=3))
    with pytest.raises(ValueError, match="later frame than appearance"):
        load_coding_sample(clips[0], 0, 0)


def test_channels_are_paired_by_position_not_filename(tmp_path: Path) -> None:
    """Crop uses global source ids; skeleton is track-local from zero."""
    root = _tiny_probe_set(
        tmp_path,
        n_frames=2,
        crop_ids=[641, 642],
        skeleton_ids=[0, 1],
    )
    sample = load_coding_sample(list_clips(root)[0], 0, 1)
    assert int(sample.pose_rgb[0, 0, 0]) == 2
    assert sample.appearance_rgb[4, 4, 0] == 40
    assert sample.reference_rgb[4, 4, 0] == 50


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
        load_coding_sample(clips[0], 0, 1)


def test_object_psnr_is_worse_than_frame_when_only_the_player_moves() -> None:
    from src.components.generation._numpy import as_hwc, prepare_letterboxed

    reference = np.zeros((32, 16, 3), dtype=np.uint8)
    reference[4:28, 4:12] = 200
    mask = np.zeros((32, 16), dtype=bool)
    mask[4:28, 4:12] = True
    predicted_crop = reference.copy()
    predicted_crop[4:28, 4:12] = 40
    predicted = as_hwc(prepare_letterboxed(predicted_crop, None, 32, 32)["appearance"])
    appearance = np.zeros((32, 16, 3), dtype=np.uint8)
    appearance[4:28, 4:12] = 40
    score = score_generation(
        reference,
        predicted,
        object_mask=mask,
        canvas_width=32,
        canvas_height=32,
        appearance=appearance,
    )
    assert score.differs_from_reference
    assert score.frame_psnr_db > score.object_psnr_db
    assert np.isfinite(score.object_psnr_db)


def test_identical_output_is_flagged_as_self_reconstruction() -> None:
    appearance = np.full((16, 16, 3), 80, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    score = score_generation(
        appearance,
        appearance,
        object_mask=mask,
        canvas_width=16,
        canvas_height=16,
        appearance=appearance,
    )
    assert score.differs_from_input is False
    assert score.differs_from_reference is False
    assert np.isinf(score.object_psnr_db)


def test_drive_engine_records_a_construct_refusal(tmp_path: Path) -> None:
    from experiments.probe.engines import EnginePlan

    plan = EnginePlan(
        name="mofa-video",
        kind="refuse-construct",
        offsets=(),
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


class _PaintPipe:
    """A generator that paints a constant colour, independent of appearance."""

    def __init__(self, rgb: tuple[int, int, int]) -> None:
        self.rgb = rgb

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> np.ndarray:
        del conditioning, seed, device
        from src.components.generation._numpy import as_chw

        width = params.width if params.width is not None else 512
        height = params.height if params.height is not None else 512
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = self.rgb
        return as_chw(canvas)


def test_static_copy_always_runs_and_records_offset(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=5)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("upscale-refine",),
        generators={"upscale-refine": _CopyPipe()},
        keyframe_index=0,
        offsets=(1, 2),
        self_recon=True,
        progress=lambda *_args, **_kwargs: None,
    )
    assert STATIC_COPY in summary["engines"]
    assert "upscale-refine" in summary["engines"]
    assert summary["keyframe_index"] == 0
    assert summary["offsets"] == [1, 2]
    dumped = json.loads((tmp_path / "out" / "static-copy.json").read_text())
    offsets = {row["offset"] for row in dumped["clips"]}
    assert offsets == {1, 2}
    assert all(row["appearance_frame_index"] == 0 for row in dumped["clips"])
    assert all(row["target_frame_index"] == 0 + row["offset"] for row in dumped["clips"])
    assert len(dumped["clips"]) == 2


def test_at_or_below_floor_is_labelled_not_using_appearance(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("upscale-refine",),
        generators={"upscale-refine": _CopyPipe()},
        keyframe_index=0,
        offsets=(2,),
        self_recon=False,
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["upscale-refine"]["headline"]
    assert headline["appearance_use"] == NOT_USING_APPEARANCE
    assert "not using appearance" in (
        headline.get("object_bound_note") or headline["appearance_use"]
    )
    rows = json.loads((tmp_path / "out" / "upscale-refine.json").read_text())["clips"]
    assert rows[0]["appearance_use"] == NOT_USING_APPEARANCE
    assert rows[0]["error"] is None


def test_engine_that_beats_the_floor_is_not_labelled_unused(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("pix2pix",),
        generators={"pix2pix": _PaintPipe((60, 80, 120))},
        keyframe_index=0,
        offsets=(2,),
        self_recon=False,
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["pix2pix"]["headline"]
    assert headline["appearance_use"] == "beats floor"
    assert headline["vs_static_copy_db"] > 0


def test_self_reconstruction_is_recorded_and_not_used_for_ranking() -> None:
    summaries = {
        STATIC_COPY: {
            "headline": {"object_psnr_db": 12.0, "self_reconstruction_psnr": 99.0}
        },
        "weak": {"headline": {"object_psnr_db": 10.0, "self_reconstruction_psnr": 40.0}},
        "strong": {"headline": {"object_psnr_db": 14.0, "self_reconstruction_psnr": 8.0}},
    }
    assert rank_engines(summaries) == ["strong", "weak"]


def test_driven_self_reconstruction_is_on_the_record_not_the_rank_key(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("upscale-refine",),
        generators={"upscale-refine": _CopyPipe()},
        keyframe_index=0,
        offsets=(2,),
        self_recon=True,
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["upscale-refine"]["headline"]
    assert headline["self_reconstruction_psnr"] is not None
    assert headline["ranking_uses"] == "object_psnr_db"
    assert "self_reconstruction_psnr" in headline["ranking_ignores"]
    assert summary["ranking_uses"] == "object_psnr_db"
    rows = json.loads((tmp_path / "out" / "upscale-refine.json").read_text())["clips"]
    assert rows[0]["self_reconstruction_psnr"] is not None
    assert rows[0]["object_psnr_db"] != rows[0]["self_reconstruction_psnr"]
