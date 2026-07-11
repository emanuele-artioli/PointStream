"""Unit tests for scripts/select_probe_set.py.

All tests build a tiny synthetic dataset tree under tmp_path mirroring
`assets/dataset/<video>/segmentations/scene_NNN/track_MMMM(_skeleton|_canny)`
— fast and mockable, no real dataset or GPU/model dependency, per this
workstream's testing ground rules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.select_probe_set import (
    HELD_OUT_VIDEOS,
    TRAINING_SPLIT_VIDEOS,
    build_manifest,
    discover_candidate_tracks,
    materialize_training_view,
    select_probe_clips,
)


def _make_track(
    dataset_root: Path,
    video: str,
    scene: str,
    track: str,
    frame_ids: list[int],
    with_skeleton: bool = True,
    with_canny: bool = True,
) -> None:
    seg_root = dataset_root / video / "segmentations" / scene
    track_dir = seg_root / track
    track_dir.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        (track_dir / f"frame_{fid:06d}.png").write_bytes(b"fake-png")

    if with_skeleton:
        skel_dir = seg_root / f"{track}_skeleton"
        skel_dir.mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            (skel_dir / f"frame_{fid:06d}.png").write_bytes(b"fake-skeleton")

    if with_canny:
        canny_dir = seg_root / f"{track}_canny"
        canny_dir.mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            (canny_dir / f"frame_{fid:06d}.png").write_bytes(b"fake-canny")

    (seg_root / f"{track}_caption.json").write_text('{"caption": "test"}')
    (seg_root / f"{track}_metadata.json").write_text("[]")
    (seg_root / f"{track}_keypoints.json").write_text("[]")


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _make_track(root, "alcaraz_ruud", "scene_002", "track_0021", list(range(0, 40)))
    _make_track(root, "alcaraz_ruud", "scene_004", "track_0100", list(range(100, 130)))
    _make_track(root, "sinner_alcaraz", "scene_001", "track_0005", list(range(0, 60)))
    _make_track(root, "sinner_alcaraz", "scene_002", "track_0006", list(range(0, 5)))  # too short (min_frames)
    _make_track(root, "djokovic_federer", "scene_003", "track_0010", list(range(0, 50)))
    # No skeleton sibling -> ineligible even though frame count is fine.
    _make_track(root, "federer_djokovic", "scene_001", "track_0099", list(range(0, 50)), with_skeleton=False)
    return root


def test_discover_candidate_tracks_filters_by_min_frames_and_skeleton(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(
        fake_dataset, ("alcaraz_ruud", "sinner_alcaraz", "djokovic_federer", "federer_djokovic"), min_frames=8
    )
    keys = {c.key for c in candidates}

    assert "alcaraz_ruud/scene_002/track_0021" in keys
    assert "alcaraz_ruud/scene_004/track_0100" in keys
    assert "sinner_alcaraz/scene_001/track_0005" in keys
    assert "djokovic_federer/scene_003/track_0010" in keys
    # Excluded: too short.
    assert "sinner_alcaraz/scene_002/track_0006" not in keys
    # Excluded: no skeleton sibling.
    assert "federer_djokovic/scene_001/track_0099" not in keys


def test_discover_candidate_tracks_ignores_missing_video(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("alcaraz_perricard",), min_frames=8)
    assert candidates == []


def test_select_probe_clips_is_deterministic_for_same_seed(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(
        fake_dataset, ("alcaraz_ruud", "sinner_alcaraz", "djokovic_federer"), min_frames=8
    )
    clips_a = select_probe_clips(candidates, seed=42, num_clips=3, clip_len_frames=16)
    clips_b = select_probe_clips(candidates, seed=42, num_clips=3, clip_len_frames=16)
    assert clips_a == clips_b


def test_select_probe_clips_different_seeds_can_differ(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(
        fake_dataset, ("alcaraz_ruud", "sinner_alcaraz", "djokovic_federer"), min_frames=8
    )
    clips_a = select_probe_clips(candidates, seed=1, num_clips=3, clip_len_frames=16)
    clips_b = select_probe_clips(candidates, seed=2, num_clips=3, clip_len_frames=16)
    # Not a hard guarantee in general, but with these fixtures the starting
    # offsets should differ for at least one clip.
    assert clips_a != clips_b


def test_select_probe_clips_spreads_across_videos(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(
        fake_dataset, ("alcaraz_ruud", "sinner_alcaraz", "djokovic_federer"), min_frames=8
    )
    clips = select_probe_clips(candidates, seed=7, num_clips=3, clip_len_frames=16)
    videos = {c.video for c in clips}
    assert len(videos) == 3  # one clip drawn from each of the 3 videos present


def test_select_probe_clips_window_is_contiguous_and_within_bounds(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("sinner_alcaraz",), min_frames=8)
    clips = select_probe_clips(candidates, seed=3, num_clips=1, clip_len_frames=10)
    assert len(clips) == 1
    clip = clips[0]
    assert len(clip.frame_ids) == 10
    ids = list(clip.frame_ids)
    assert ids == sorted(ids)
    # contiguous means consecutive entries in the *candidate's sorted frame_ids*
    assert all(b - a == 1 for a, b in zip(ids, ids[1:]))


def test_select_probe_clips_handles_track_shorter_than_clip_len(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("alcaraz_ruud",), min_frames=8)
    # alcaraz_ruud/scene_004/track_0100 has 30 frames; ask for a longer window.
    clips = select_probe_clips(candidates, seed=1, num_clips=2, clip_len_frames=1000)
    for clip in clips:
        assert len(clip.frame_ids) <= 40  # capped at whichever track's own length


def test_select_probe_clips_num_clips_zero_returns_empty(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("alcaraz_ruud",), min_frames=8)
    assert select_probe_clips(candidates, seed=1, num_clips=0, clip_len_frames=10) == []


def test_build_manifest_records_excluded_keys_and_splits(fake_dataset: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("alcaraz_ruud", "sinner_alcaraz"), min_frames=8)
    clips = select_probe_clips(candidates, seed=1, num_clips=2, clip_len_frames=10)
    manifest = build_manifest(clips, seed=1, clip_len_frames=10, min_frames=8, training_videos=("alcaraz_ruud", "sinner_alcaraz"))

    assert manifest["schema"] == "pointstream.probe_set.v1"
    assert manifest["num_probe_clips"] == len(clips)
    assert set(manifest["held_out_videos"]) == set(HELD_OUT_VIDEOS)
    assert set(manifest["excluded_training_keys"]) == {c.key for c in clips}


def test_training_split_and_held_out_videos_are_disjoint() -> None:
    assert set(TRAINING_SPLIT_VIDEOS).isdisjoint(set(HELD_OUT_VIDEOS))


def test_materialize_training_view_excludes_probe_tracks_but_keeps_others(fake_dataset: Path, tmp_path: Path) -> None:
    candidates = discover_candidate_tracks(fake_dataset, ("alcaraz_ruud",), min_frames=8)
    clips = select_probe_clips(candidates, seed=1, num_clips=1, clip_len_frames=10)
    excluded_keys = {c.key for c in clips}
    assert excluded_keys  # sanity: something was actually excluded

    output_dir = tmp_path / "training_view"
    materialize_training_view(
        dataset_root=fake_dataset,
        output_dir=output_dir,
        training_videos=("alcaraz_ruud",),
        excluded_training_keys=excluded_keys,
    )

    excluded_key = next(iter(excluded_keys))
    _, scene, track = excluded_key.split("/")

    # The excluded track must not be reachable in the materialized view at all.
    assert not (output_dir / "alcaraz_ruud" / "segmentations" / scene / track).exists()
    assert not (output_dir / "alcaraz_ruud" / "segmentations" / scene / f"{track}_skeleton").exists()

    # Some other track in alcaraz_ruud must still be present (non-overlap does
    # not mean the whole video is thrown away).
    all_tracks = {c.track for c in candidates}
    remaining_tracks = all_tracks - {track}
    assert remaining_tracks, "fixture should have more than one track to prove partial exclusion"
    other_track = next(iter(remaining_tracks))
    other_scene = next(c.scene for c in candidates if c.track == other_track)
    assert (output_dir / "alcaraz_ruud" / "segmentations" / other_scene / other_track).exists()


def test_materialize_training_view_refuses_to_overwrite_existing_dir(fake_dataset: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "training_view"
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        materialize_training_view(
            dataset_root=fake_dataset,
            output_dir=output_dir,
            training_videos=("alcaraz_ruud",),
            excluded_training_keys=set(),
        )
