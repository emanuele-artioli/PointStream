"""Probe harness: coding-task behaviour and plausible misuse. No GPU, no real weights.

Behaviour the caller relies on: appearance and reference are different frames,
static copy always runs, at-or-below that floor is labelled "not using
appearance", self-reconstruction is recorded and never used for ranking,
the keyframe/target offset is in the output, and more than one offset is
scored per clip.

Plausible misuse is the silent kind — pairing channels by reconstructing a
filename, scoring the keyframe against itself, ranking on the diagnostic
PSNR, treating a pasted keyframe as a low-ranking engine.

Clip mode, the LPIPS ranking key and the null control arrived with BP12. The
faults they exist to catch are all of the silent kind: a temporal model driven
one frame at a time, a ranking taken on a metric with no dynamic range, and a
"control" that quietly pastes the right player.

Deliberately not tested: live probe-set PNG contents, argparse help, real
generator weights, VMAF, real LPIPS *values* (anchored against published
figures in tests/invariants/test_metric_calibration.py — duplicating them here
would only be slower), _pad_box arithmetic beyond the one growing case,
Animate-Anyone's own pipeline internals, Wave-2 self-reconstruction numbers as
a regression target, construct.py's SAM3 leftover.
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
    judge_engine_lpips,
    judge_null_separation,
    judge_static_copy_clip,
    judge_static_copy_lpips,
    judge_static_copy_object_psnr,
    judge_unrelated_lpips,
)
from experiments.probe.clips import (
    list_clips,
    load_coding_sample,
    load_coding_sequence,
    load_frame,
)
from experiments.probe.engines import STATIC_COPY, UNRELATED_IMAGE, plan_for
from experiments.probe.run import (
    donor_for,
    drive_all,
    drive_engine,
    rank_engines,
)
from experiments.probe.score import score_generation
from experiments.probe_set.schema import SCHEMA_ID, TRAINING_SPLIT_VIDEOS
from src.contracts.conditioning import ConditioningBundle, GenerationParams


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _clip_colour(index: int) -> tuple[int, int, int]:
    """A distinct player colour per clip, on all three channels.

    Separating the clips on red alone is not enough: a distance averaged over
    channels divides a one-channel difference by three, and the fixture's null
    control then fails to separate for a reason that has nothing to do with the
    code under test. Red also carries the +10-per-frame ramp, so its base stays
    low enough that a clip's last frame still fits in a byte.
    """
    return (
        40 + (index * 61) % 120,
        30 + (index * 83) % 180,
        20 + (index * 97) % 200,
    )


def _write_clip(
    root: Path,
    *,
    video: str,
    scene: str,
    track: str,
    n_frames: int,
    crop_ids: list[int],
    skeleton_ids: list[int],
    colour: tuple[int, int, int],
) -> dict[str, Any]:
    crop_dir = root / "clips" / video / scene / track
    skel_dir = root / "clips" / video / scene / f"{track}_skeleton"
    canny_dir = root / "clips" / video / scene / f"{track}_canny"
    h, w = 32, 16
    for index in range(n_frames):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[4:28, 4:12, :3] = (colour[0] + index * 10, colour[1], colour[2])
        rgba[4:28, 4:12, 3] = 255
        pose = np.zeros((h, w, 3), dtype=np.uint8)
        pose[6:26, 7:9] = 255
        pose[0, 0] = index + 1
        canny = np.zeros((h, w), dtype=np.uint8)
        canny[6:26, 7:9] = 255
        _write_png(crop_dir / f"frame_{crop_ids[index]:06d}.png", rgba)
        _write_png(skel_dir / f"frame_{skeleton_ids[index]:06d}.png", pose)
        _write_png(canny_dir / f"frame_{crop_ids[index]:06d}.png", canny)
    return {
        "video": video,
        "scene": scene,
        "track": track,
        "key": f"{video}/{scene}/{track}",
        "path": f"clips/{video}/{scene}/{track}",
        "num_frames": n_frames,
        "frame_ids": crop_ids,
    }


def _tiny_probe_set(
    tmp_path: Path,
    *,
    n_frames: int = 5,
    crop_ids: list[int] | None = None,
    skeleton_ids: list[int] | None = None,
    n_clips: int = 1,
) -> Path:
    """Clips cycle through the training-split videos, one per clip.

    The null control borrows a keyframe from another *video*, so anything that
    drives the full arm list needs at least two clips, and a paired comparison
    wants eight before it will call a direction. One clip is enough for the
    loader tests and keeps their row counts readable.
    """
    root = tmp_path / "probe_set"
    crop_ids = list(range(n_frames)) if crop_ids is None else crop_ids
    skeleton_ids = list(range(n_frames)) if skeleton_ids is None else skeleton_ids
    if len(crop_ids) != n_frames or len(skeleton_ids) != n_frames:
        raise AssertionError("fixture ids must match n_frames")
    records = [
        _write_clip(
            root,
            video=TRAINING_SPLIT_VIDEOS[index % len(TRAINING_SPLIT_VIDEOS)],
            scene=f"scene_{index + 1:03d}",
            track=f"track_{index + 1:04d}",
            n_frames=n_frames,
            crop_ids=crop_ids,
            skeleton_ids=skeleton_ids,
            colour=_clip_colour(index),
        )
        for index in range(n_clips)
    ]
    manifest = {"schema": SCHEMA_ID, "probe_clips": records}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_bounds_are_anchored_on_the_static_copy_floor() -> None:
    assert STATIC_COPY_EXPECTED_LOW_DB == 8.0
    assert STATIC_COPY_EXPECTED_HIGH_DB == 16.0
    assert judge_static_copy_object_psnr(11.82).status == "expected"
    assert judge_static_copy_clip(5.91).status == "ok"
    assert judge_static_copy_clip(3.0).status == "alarm-low"
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


class _ImprovePipe:
    """Nudges the keyframe toward the target: the fixture's ramp is +10 red per
    frame, so this lands on the reference whatever colour a clip starts from.

    A constant-colour pipe cannot beat the floor on two clips at once, and the
    thing under test is the label, not the pipe.
    """

    def __init__(self, red_shift: int) -> None:
        self.red_shift = red_shift

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> np.ndarray:
        del seed, device
        from src.components.generation._numpy import as_chw, as_hwc, prepare_letterboxed

        width = params.width if params.width is not None else 512
        height = params.height if params.height is not None else 512
        appearance = as_hwc(conditioning.appearance).astype(np.int16)
        appearance[..., 0] = np.clip(appearance[..., 0] + self.red_shift, 0, 255)
        prepared = prepare_letterboxed(appearance.astype(np.uint8), None, width, height)
        return as_chw(prepared["appearance"])


def test_static_copy_always_runs_and_records_offset(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=5, n_clips=2)
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
        lpips_metric=_FakeLpips(),
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
    assert len(dumped["clips"]) == 4  # two clips x two offsets


def test_at_or_below_floor_is_labelled_not_using_appearance(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
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
        lpips_metric=_FakeLpips(),
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
    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("pix2pix",),
        generators={"pix2pix": _ImprovePipe(20)},
        keyframe_index=0,
        offsets=(2,),
        self_recon=False,
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["pix2pix"]["headline"]
    assert headline["appearance_use"] == "beats floor"
    assert headline["vs_static_copy_db"] > 0


def test_self_reconstruction_is_recorded_and_not_used_for_ranking() -> None:
    summaries = {
        STATIC_COPY: {"headline": {"object_lpips": 0.40, "self_reconstruction_psnr": 99.0}},
        UNRELATED_IMAGE: {"headline": {"object_lpips": 0.65}},
        "weak": {"headline": {"object_lpips": 0.60, "self_reconstruction_psnr": 40.0}},
        "strong": {"headline": {"object_lpips": 0.30, "self_reconstruction_psnr": 8.0}},
    }
    assert rank_engines(summaries) == ["strong", "weak"]


def test_driven_self_reconstruction_is_on_the_record_not_the_rank_key(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
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
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["upscale-refine"]["headline"]
    assert headline["self_reconstruction_psnr"] is not None
    assert headline["ranking_uses"] == "object_lpips"
    assert "self_reconstruction_psnr" in headline["ranking_ignores"]
    assert summary["ranking_uses"] == "object_lpips"
    rows = json.loads((tmp_path / "out" / "upscale-refine.json").read_text())["clips"]
    assert rows[0]["self_reconstruction_psnr"] is not None
    assert rows[0]["object_psnr_db"] != rows[0]["self_reconstruction_psnr"]


# ---------------------------------------------------------------------------
# BP12: clip mode, the LPIPS ranking key, and the null control.


class _FakeLpips:
    """A deterministic stand-in with real dynamic range.

    The published anchors are asserted in the metric calibration invariants.
    What is worth checking *here* is the plumbing: that a distance reaches the
    rows, the headline and the ranking. A fake keeps these tests CPU-fast and
    lets a caller construct the cases a real metric will not produce on demand.
    """

    name = "lpips"

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref = np.asarray(reference, dtype=np.float64)
        pred = np.asarray(predicted, dtype=np.float64)
        return float(np.abs(ref - pred).mean() / 255.0)


class _RecordingSequencePipe:
    """A temporal generator that records exactly how it was driven."""

    def __init__(self) -> None:
        self.sequence_calls: list[list[int]] = []
        self.frame_calls = 0

    def generate_sequence(
        self,
        conditioning: Any,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> list[np.ndarray]:
        del seed, device
        bundles = list(conditioning)
        self.sequence_calls.append([int(bundle.frame_index) for bundle in bundles])
        from src.components.generation._numpy import as_chw, prepare_letterboxed

        width = params.width if params.width is not None else 512
        height = params.height if params.height is not None else 512
        return [
            as_chw(prepare_letterboxed(bundle.appearance, None, width, height)["appearance"])
            for bundle in bundles
        ]

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> np.ndarray:
        del conditioning, seed, device, params
        self.frame_calls += 1
        raise AssertionError("a temporal engine must not reach the single-frame path")


class _ShortSequencePipe(_RecordingSequencePipe):
    """Returns one frame fewer than it was asked for."""

    def generate_sequence(
        self,
        conditioning: Any,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> list[np.ndarray]:
        frames = super().generate_sequence(
            conditioning, seed=seed, device=device, params=params
        )
        return frames[:-1]


def _temporal_plan(offsets: tuple[int, ...]) -> Any:
    from experiments.probe.engines import EnginePlan

    return EnginePlan(
        name="animate-anyone",
        kind="temporal",
        offsets=offsets,
        steps=2,
        sequence=True,
        notes="test",
    )


def test_a_temporal_engine_is_driven_once_per_clip_not_once_per_offset(
    tmp_path: Path,
) -> None:
    """The T=1 fault. AA carries a motion module and was called frame by frame."""
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=6))
    pipe = _RecordingSequencePipe()
    result = drive_engine(
        _temporal_plan((1, 2, 3)),
        clips,
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        generator=pipe,
        keyframe_index=0,
        offsets=(1, 2, 3),
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    assert pipe.frame_calls == 0
    assert pipe.sequence_calls == [[1, 2, 3]], "one call per clip, in time order"
    assert result.drive_mode == "clip"
    assert [row.offset for row in result.clips] == [1, 2, 3]
    assert all(row.drive_mode == "clip" for row in result.clips)


def test_a_sequence_plan_without_a_sequence_path_refuses_to_fall_back(
    tmp_path: Path,
) -> None:
    """Falling back to generate() is worse than crashing: it produces a number."""
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=4))
    with pytest.raises(RuntimeError, match="no generate_sequence"):
        drive_engine(
            _temporal_plan((1, 2)),
            clips,
            device="cpu",
            seed=0,
            out_dir=tmp_path / "out",
            generator=_CopyPipe(),
            offsets=(1, 2),
            lpips_metric=_FakeLpips(),
            progress=lambda *_args, **_kwargs: None,
        )


def test_a_short_sequence_is_an_error_not_a_truncated_zip(tmp_path: Path) -> None:
    """zip() would silently pair frame 1's output with frame 1's target and drop
    the last — every row plausible, the clip quietly one frame short."""
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=6))
    result = drive_engine(
        _temporal_plan((1, 2, 3)),
        clips,
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        generator=_ShortSequencePipe(),
        offsets=(1, 2, 3),
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    assert all(row.error is not None for row in result.clips)
    assert "returned 2 frames for 3 bundles" in (result.clips[0].error or "")


def test_a_clip_sequence_shares_one_keyframe_and_is_time_ordered(tmp_path: Path) -> None:
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=6))
    samples = load_coding_sequence(clips[0], 0, (3, 1, 2, 1))
    assert [sample.offset for sample in samples] == [1, 2, 3]
    assert all(sample.appearance_frame_index == 0 for sample in samples)
    assert [sample.target_frame_index for sample in samples] == [1, 2, 3]
    first = samples[0].appearance_rgb
    assert all(np.array_equal(sample.appearance_rgb, first) for sample in samples)


def test_lpips_reaches_the_rows_the_headline_and_the_ranking(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("pix2pix",),
        generators={"pix2pix": _PaintPipe((60, 80, 120))},
        keyframe_index=0,
        offsets=(1, 2),
        self_recon=False,
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    headline = summary["engines"]["pix2pix"]["headline"]
    assert headline["object_lpips"] is not None
    assert headline["object_psnr_db"] is not None, "PSNR is reported beside, not instead"
    assert headline["ranking_lower_is_better"] is True
    assert "object_psnr_db" in headline["reported_beside"]
    assert headline["anchors"]["static_copy_lpips"] is not None
    assert headline["anchors"]["unrelated_image_lpips"] is not None
    rows = json.loads((tmp_path / "out" / "pix2pix.json").read_text())["clips"]
    assert all(row["object_lpips"] is not None for row in rows)


def test_ranking_follows_lpips_even_when_psnr_disagrees() -> None:
    """The two orders are not the same order. Mixing them is how a table lies."""
    summaries = {
        STATIC_COPY: {"headline": {"object_lpips": 0.40, "object_psnr_db": 12.0}},
        UNRELATED_IMAGE: {"headline": {"object_lpips": 0.65, "object_psnr_db": 9.0}},
        "sharp": {"headline": {"object_lpips": 0.30, "object_psnr_db": 10.0}},
        "blurry": {"headline": {"object_lpips": 0.55, "object_psnr_db": 14.0}},
    }
    assert rank_engines(summaries) == ["sharp", "blurry"]


def test_an_engine_without_lpips_is_left_out_rather_than_ranked_on_psnr() -> None:
    summaries: dict[str, dict[str, Any]] = {
        "measured": {"headline": {"object_lpips": 0.50, "object_psnr_db": 11.0}},
        "unmeasured": {"headline": {"object_lpips": None, "object_psnr_db": 20.0}},
    }
    assert rank_engines(summaries) == ["measured"]


def test_the_null_control_borrows_a_keyframe_from_another_video(tmp_path: Path) -> None:
    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
    clips = list_clips(root)
    assert donor_for(clips, 0).video != clips[0].video
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=(),
        keyframe_index=0,
        offsets=(1, 2),
        self_recon=False,
        lpips_metric=_FakeLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    assert summary["donors"][clips[0].key] == clips[1].key
    rows = json.loads((tmp_path / "out" / "unrelated-image.json").read_text())["clips"]
    assert all(row["appearance_source"].startswith("donor:") for row in rows)
    assert summary["control"]["readable"] is True
    assert summary["control"]["separation"] > 0


def test_one_clip_cannot_supply_a_null_control(tmp_path: Path) -> None:
    """A control that pastes the right player is not a control."""
    clips = list_clips(_tiny_probe_set(tmp_path, n_frames=4))
    with pytest.raises(ValueError, match="needs a second clip"):
        donor_for(clips, 0)


def test_a_run_whose_control_does_not_separate_ranks_nothing(tmp_path: Path) -> None:
    """The 2026-08-23 fault, made structural: a metric that cannot tell the
    right player from the wrong one used to still produce a ranking."""

    class _BlindLpips:
        name = "lpips"

        def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
            del reference, predicted
            return 0.5

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=2)
    summary = drive_all(
        device="cpu",
        seed=0,
        out_dir=tmp_path / "out",
        probe_root=root,
        engines=("pix2pix",),
        generators={"pix2pix": _PaintPipe((60, 80, 120))},
        keyframe_index=0,
        offsets=(1, 2),
        self_recon=False,
        lpips_metric=_BlindLpips(),
        progress=lambda *_args, **_kwargs: None,
    )
    assert summary["control"]["readable"] is False
    assert summary["control"]["status"] == "alarm-low"
    assert summary["rank"] == []
    assert summary["engines"]["pix2pix"]["headline"]["object_lpips"] == 0.5


def test_lpips_bounds_fire_where_they_were_written() -> None:
    assert judge_static_copy_lpips(0.42).status == "expected"
    assert judge_static_copy_lpips(0.02).status == "alarm-low"
    assert judge_unrelated_lpips(0.65).status == "expected"
    assert judge_unrelated_lpips(0.30).status == "alarm-low"
    assert judge_null_separation(0.42, 0.47).status == "alarm-low"
    assert judge_null_separation(0.42, 0.65).status == "ok"
    # near-identity from a 20-step generation is a scoring fault, not a win
    assert judge_engine_lpips(0.02, 0.42, 0.65).status == "alarm-low"
    assert judge_engine_lpips(0.30, 0.42, 0.65).status == "beats-floor"
    assert judge_engine_lpips(0.70, 0.42, 0.65).status == "at-or-worse-than-null"
    between = judge_engine_lpips(0.57, 0.42, 0.65)
    assert between.status == "between-floor-and-null"
    assert "0.420" in between.note and "0.650" in between.note


def test_absence_of_an_appearance_effect_is_a_claim_about_the_interval() -> None:
    """A near-zero point estimate with a wide interval rules nothing out, and
    a 1.5-sigma effect is not a direction. Both were reported here as findings."""
    from experiments.probe.bounds import judge_cross_appearance

    tight = judge_cross_appearance(0.001, sigmas=0.1, standard_error=0.004)
    assert tight.status == "no appearance pathway"
    loose = judge_cross_appearance(0.001, sigmas=0.02, standard_error=0.050)
    assert loose.status == "inside-noise", "wide interval cannot rule an effect out"
    weak = judge_cross_appearance(0.09, sigmas=1.5, standard_error=0.060)
    assert weak.status == "inside-noise"
    leak = judge_cross_appearance(0.05, sigmas=3.0, standard_error=0.017)
    assert leak.status == "init leakage only"
    works = judge_cross_appearance(0.16, sigmas=4.0, standard_error=0.040)
    assert works.status == "uses appearance"


# ---------------------------------------------------------------------------
# BP12: the cross-appearance control.


def _cross_kwargs(tmp_path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        "device": "cpu",
        "seed": 0,
        "out_dir": tmp_path / "cross",
        "probe_root": root,
        "keyframe_index": 0,
        "offsets": (1, 2),
        "lpips_metric": _FakeLpips(),
        "paste_separation_lpips": 0.285,
        "progress": lambda *_args, **_kwargs: None,
        **extra,
    }


class _IgnoresAppearancePipe:
    """Paints the same thing whatever it is shown. The wiring-fault case."""

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
        canvas[:] = (90, 90, 90)
        return as_chw(canvas)


def test_an_engine_that_ignores_appearance_shows_no_cross_appearance_delta(
    tmp_path: Path,
) -> None:
    from experiments.probe.cross_appearance import run_cross_appearance

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=10)
    result = run_cross_appearance(
        "pix2pix",
        generator=_IgnoresAppearancePipe(),
        **_cross_kwargs(tmp_path, root),
    )
    verdict = result.verdict
    assert verdict["readable"] is True
    assert verdict["n"] == 10
    assert verdict["status"] == "no appearance pathway"
    assert abs(verdict["lpips"]["delta"]) < 1e-9
    assert "Check the wiring before the architecture" in verdict["note"]


def test_an_engine_that_uses_appearance_shows_a_clear_delta(tmp_path: Path) -> None:
    from experiments.probe.cross_appearance import run_cross_appearance

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=10)
    result = run_cross_appearance(
        "upscale-refine",
        generator=_CopyPipe(),
        **_cross_kwargs(tmp_path, root),
    )
    verdict = result.verdict
    assert verdict["status"] == "uses appearance"
    assert verdict["lpips"]["delta"] > 0.10
    assert verdict["lpips"]["sigmas"] >= 2.0
    assert "of the 0.285 a paste is worth" in verdict["note"]
    assert verdict["psnr_db"]["delta"] > 0, "PSNR agrees, and is reported beside"


def test_too_few_clips_claims_no_direction_however_large_the_effect(
    tmp_path: Path,
) -> None:
    """A large sigma on three clips is not a result; compare_paired says so and
    the bound must not overrule it."""
    from experiments.probe.cross_appearance import run_cross_appearance

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=3)
    result = run_cross_appearance(
        "upscale-refine",
        generator=_CopyPipe(),
        **_cross_kwargs(tmp_path, root),
    )
    assert result.verdict["status"] == "underpowered"
    assert result.verdict["lpips"]["delta"] > 0.10


def test_both_arms_are_driven_through_the_same_path(tmp_path: Path) -> None:
    """If the correct and the wrong appearance took different code paths, the
    delta would compare invocations rather than appearances."""
    from experiments.probe.cross_appearance import run_cross_appearance

    root = _tiny_probe_set(tmp_path, n_frames=5, n_clips=2)
    pipe = _RecordingSequencePipe()
    result = run_cross_appearance(
        "animate-anyone",
        generator=pipe,
        **_cross_kwargs(tmp_path, root, offsets=(1, 2, 3)),
    )
    assert result.drive_mode == "clip"
    assert pipe.frame_calls == 0
    # two clips x (own, wrong), one sequence call each, all in time order
    assert pipe.sequence_calls == [[1, 2, 3]] * 4
    assert [pair.donor_key for pair in result.pairs] == [
        result.pairs[1].clip_key,
        result.pairs[0].clip_key,
    ]


def test_a_single_usable_clip_is_not_a_comparison(tmp_path: Path) -> None:
    from experiments.probe.cross_appearance import CrossAppearanceResult, summarise

    empty = CrossAppearanceResult(
        engine="pix2pix",
        drive_mode="frame",
        seed=0,
        device="cpu",
        offsets=[1],
        keyframe_index=0,
        paste_separation_lpips=0.285,
    )
    verdict = summarise(empty)
    assert verdict["readable"] is False
    assert "at least" in verdict["note"]


def test_donor_modes_pick_different_kinds_of_wrong_appearance(tmp_path: Path) -> None:
    """Same-video isolates the player from the court; different-video does not."""
    from experiments.probe.run import donor_for

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=6)
    clips = list_clips(root)
    # the fixture cycles five videos over six clips, so clip 0 and clip 5 share one
    assert clips[0].video == clips[5].video
    assert donor_for(clips, 0, mode="different-video").video != clips[0].video
    assert donor_for(clips, 0, mode="same-video").key == clips[5].key
    with pytest.raises(ValueError, match="donor mode must be one of"):
        donor_for(clips, 0, mode="whatever")


def test_donor_modes_are_recorded_separately_and_never_overwrite(tmp_path: Path) -> None:
    from experiments.probe.cross_appearance import run_cross_appearance

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=10)
    for mode in ("different-video", "same-video"):
        result = run_cross_appearance(
            "pix2pix",
            generator=_IgnoresAppearancePipe(),
            donor_mode=mode,
            **_cross_kwargs(tmp_path, root),
        )
        assert result.donor_mode == mode
    written = sorted(path.name for path in (tmp_path / "cross").glob("cross-appearance-*"))
    assert written == [
        "cross-appearance-pix2pix.json",
        "cross-appearance-pix2pix.same-video.json",
    ]


def test_cross_appearance_arms_are_compared_on_their_deltas(tmp_path: Path) -> None:
    """"This engine uses appearance more than that one" is a difference of two
    differences, on shared clips, and needs a standard error like anything else."""
    from experiments.probe.cross_appearance import run_cross_appearance
    from experiments.probe.report import cross_report

    root = _tiny_probe_set(tmp_path, n_frames=4, n_clips=10)
    out = tmp_path / "cross"
    run_cross_appearance(
        "upscale-refine", generator=_CopyPipe(), **_cross_kwargs(tmp_path, root)
    )
    run_cross_appearance(
        "pix2pix", generator=_IgnoresAppearancePipe(), **_cross_kwargs(tmp_path, root)
    )
    report = cross_report(out)
    arms = {arm["engine"]: arm for arm in report["arms"]}
    assert arms["upscale-refine"]["delta_lpips"] > arms["pix2pix"]["delta_lpips"]
    assert len(report["between"]) == 1
    between = report["between"][0]
    assert between["verdict"] == "clear"
    assert between["winner"] == "upscale-refine[different-video]"
