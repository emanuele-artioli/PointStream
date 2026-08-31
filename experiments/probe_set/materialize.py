"""Build the probe-set view, then derive the manifest from what was written.

The v1 fault was writing a clip list and a view independently. Here the view
is the source of truth: colour frames are materialised first (track-local
names, symlinked to the source files), and the manifest is walked off that
tree. A later verifier that compared the two would be tautological *right
after* this function; it exists to catch a subsequent edit of one side.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.probe_set.schema import (
    CLIPS_VIEW_NAME,
    CONDITION_DIR_SUFFIXES,
    COORDINATE_SYSTEM,
    DEFAULT_CLIP_LEN_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_NUM_CLIPS,
    DEFAULT_SEED,
    HELD_OUT_VIDEOS,
    SCHEMA_ID,
    SELECTION_RULE,
    SIDECAR_SUFFIXES,
    TRAINING_SPLIT_VIDEOS,
    TRAINING_VIEW_NAME,
    ProbeSetError,
)
from experiments.probe_set.select import (
    FRAME_ID_RE,
    TRACK_PRIMARY_RE,
    ProbeClip,
    discover_candidate_tracks,
    extract_track_id,
    list_source_frame_ids,
    reject_held_out,
    select_probe_clips,
)


def _replace_tree(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _symlink(src: Path, dst: Path) -> None:
    """Link ``dst`` at ``src``, **relatively** whenever both share a root.

    An absolute link records where the data sat when the view was built, and
    that is not a stable fact: the 2026-08-29 move of `assets/` and `outputs/`
    out of the checkout dangled every one of the 3,033 links in this view at
    once, because each recorded `<repo>/assets/dataset/...`. A relative link
    records the *relationship* between the view and the dataset, which the two
    keep when the pair is moved together.

    Falls back to an absolute link when the two are not under a common root —
    a link that works is better than one that is elegantly relative and wrong.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    target = src.resolve()
    try:
        target = Path(os.path.relpath(target, dst.parent.resolve()))
    except ValueError:
        pass
    dst.symlink_to(target, target_is_directory=src.is_dir())


def dataset_track_dir(dataset_root: Path, video: str, scene: str, track: str) -> Path:
    return dataset_root / video / "segmentations" / scene / track


def sorted_frame_files(directory: Path) -> list[Path]:
    """``frame_*.png`` paths in sorted filename order. Position is the track index.

    Crop, canny, ``_pose_body`` and ``_pose_racket`` name files by global source
    id; ``_skeleton`` names them track-local and zero-based. Pairing by this
    list's index, never by reconstructing a filename, is what keeps those
    conventions aligned.
    """
    frames = [
        path
        for path in directory.glob("frame_*.png")
        if FRAME_ID_RE.match(path.name) is not None
    ]
    frames.sort(key=lambda path: path.name)
    return frames


def window_positions(
    crop_frames: list[Path], source_frame_ids: tuple[int, ...]
) -> list[int]:
    """Map each selected source id to its index in the sorted crop list."""
    id_to_idx: dict[int, int] = {}
    for idx, path in enumerate(crop_frames):
        match = FRAME_ID_RE.match(path.name)
        if match is None:
            continue
        id_to_idx[int(match.group(1))] = idx
    missing = [sid for sid in source_frame_ids if sid not in id_to_idx]
    if missing:
        raise ProbeSetError(
            [f"source frames missing from crop listing: {missing[:8]}"]
        )
    return [id_to_idx[sid] for sid in source_frame_ids]


def materialize_clip(
    dataset_root: Path,
    clips_dir: Path,
    clip: ProbeClip,
) -> Path:
    """Write one clip directory of track-local frame symlinks. Returns that dir.

    Every channel is resolved by the frame's position in the track's sorted
    ``frame_*.png`` list. Reconstructing ``frame_{source_id:06d}.png`` under
    ``_skeleton`` silently skipped every file on tracks that do not start at
    source frame 0.
    """
    source_track = dataset_track_dir(dataset_root, clip.video, clip.scene, clip.track)
    dest_track = clips_dir / clip.video / clip.scene / clip.track
    dest_track.mkdir(parents=True, exist_ok=True)

    crop_frames = sorted_frame_files(source_track)
    if not crop_frames:
        raise ProbeSetError([f"{clip.key}: source track has no frames"])
    try:
        positions = window_positions(crop_frames, clip.source_frame_ids)
    except ProbeSetError as exc:
        raise ProbeSetError(
            [f"{clip.key}: {item}" for item in exc.violations]
        ) from exc

    for local_id, pos in enumerate(positions):
        _symlink(crop_frames[pos], dest_track / f"frame_{local_id:06d}.png")

    scene_dir = source_track.parent
    for suffix in CONDITION_DIR_SUFFIXES:
        source_cond = scene_dir / f"{clip.track}{suffix}"
        if not source_cond.is_dir():
            continue
        cond_frames = sorted_frame_files(source_cond)
        if len(cond_frames) != len(crop_frames):
            raise ProbeSetError(
                [
                    f"{clip.key}: {suffix} has {len(cond_frames)} frames, "
                    f"crop has {len(crop_frames)}"
                ]
            )
        dest_cond = clips_dir / clip.video / clip.scene / f"{clip.track}{suffix}"
        dest_cond.mkdir(parents=True, exist_ok=True)
        for local_id, pos in enumerate(positions):
            _symlink(cond_frames[pos], dest_cond / f"frame_{local_id:06d}.png")

    dest_scene = clips_dir / clip.video / clip.scene
    for suffix in SIDECAR_SUFFIXES:
        source_sidecar = scene_dir / f"{clip.track}{suffix}"
        if source_sidecar.is_file():
            _symlink(source_sidecar, dest_scene / f"{clip.track}{suffix}")

    return dest_track


def iter_clip_track_dirs(clips_dir: Path) -> list[Path]:
    """Primary ``track_NNNN`` directories under the clips view, sorted."""
    found: list[Path] = []
    if not clips_dir.is_dir():
        return found
    for video_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in video_dir.iterdir() if p.is_dir()):
            for item in sorted(scene_dir.iterdir()):
                if item.is_dir() and TRACK_PRIMARY_RE.match(item.name):
                    found.append(item)
    return found


def _global_id_from_symlink(frame_path: Path) -> int:
    """Recover the source-video frame number from the symlink target's name."""
    target = frame_path.resolve()
    match = FRAME_ID_RE.match(target.name)
    if match is None:
        raise ProbeSetError(
            [f"{frame_path}: symlink target {target.name!r} is not a frame_*.png"]
        )
    return int(match.group(1))


def manifest_from_view(
    clips_dir: Path,
    *,
    seed: int,
    clip_len_frames: int,
    min_frames: int,
    training_videos: tuple[str, ...],
    dataset_root: Path,
) -> dict[str, Any]:
    """Walk the materialised clips view and describe it. The view is the source."""
    probe_clips: list[dict[str, Any]] = []
    for track_dir in iter_clip_track_dirs(clips_dir):
        video = track_dir.parent.parent.name
        scene = track_dir.parent.name
        track = track_dir.name
        local_ids = list_source_frame_ids(track_dir)
        if not local_ids:
            raise ProbeSetError([f"{video}/{scene}/{track}: view track has no frames"])
        global_ids: list[int] = []
        for local_id in local_ids:
            frame_path = track_dir / f"frame_{local_id:06d}.png"
            if not frame_path.exists():
                raise ProbeSetError([f"{frame_path}: named in directory listing but missing"])
            global_ids.append(_global_id_from_symlink(frame_path))
        rel_path = track_dir.relative_to(clips_dir).as_posix()
        probe_clips.append(
            {
                "video": video,
                "scene": scene,
                "track": track,
                "key": f"{video}/{scene}/{track}",
                "path": f"{CLIPS_VIEW_NAME}/{rel_path}",
                "frame_ids": list(local_ids),
                "num_frames": len(local_ids),
                "global_offset": global_ids[0],
                "global_frame_ids": global_ids,
                "source_track": (
                    dataset_track_dir(dataset_root, video, scene, track).as_posix()
                ),
            }
        )

    keys = sorted(clip["key"] for clip in probe_clips)
    return {
        "schema": SCHEMA_ID,
        "coordinate_system": COORDINATE_SYSTEM,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "selection_rule": SELECTION_RULE,
        "clip_len_frames": clip_len_frames,
        "min_frames": min_frames,
        "training_videos": list(training_videos),
        "held_out_videos": list(HELD_OUT_VIDEOS),
        "num_probe_clips": len(probe_clips),
        "view": CLIPS_VIEW_NAME,
        "training_view": TRAINING_VIEW_NAME,
        "dataset_root": dataset_root.as_posix(),
        "probe_clips": probe_clips,
        "excluded_training_keys": keys,
    }


def materialize_training_view(
    dataset_root: Path,
    output_dir: Path,
    training_videos: tuple[str, ...],
    excluded_training_keys: set[str],
) -> None:
    """Symlink farm of the training-split videos with probe tracks omitted.

    Held-out videos are never walked. Whole scenes with nothing excluded are
    symlinked wholesale; scenes containing an excluded track are rebuilt
    entry-by-entry so the probe track and its siblings disappear.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_by_scene: dict[tuple[str, str], set[str]] = {}
    for key in excluded_training_keys:
        video, scene, track = key.split("/")
        excluded_by_scene.setdefault((video, scene), set()).add(track)

    for video in training_videos:
        src_video_dir = dataset_root / video
        if not src_video_dir.is_dir():
            continue
        dst_video_dir = output_dir / video
        dst_video_dir.mkdir(exist_ok=True)
        for entry in sorted(src_video_dir.iterdir()):
            if entry.name != "segmentations":
                _symlink(entry, dst_video_dir / entry.name)
                continue
            dst_seg_dir = dst_video_dir / "segmentations"
            dst_seg_dir.mkdir(exist_ok=True)
            for scene_dir in sorted(p for p in entry.iterdir() if p.is_dir()):
                excluded_tracks = excluded_by_scene.get((video, scene_dir.name))
                if not excluded_tracks:
                    _symlink(scene_dir, dst_seg_dir / scene_dir.name)
                    continue
                dst_scene_dir = dst_seg_dir / scene_dir.name
                dst_scene_dir.mkdir(exist_ok=True)
                for item in sorted(scene_dir.iterdir()):
                    track_id = extract_track_id(item.name)
                    if track_id is not None and track_id in excluded_tracks:
                        continue
                    _symlink(item, dst_scene_dir / item.name)


def regenerate(
    dataset_root: Path,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    num_clips: int = DEFAULT_NUM_CLIPS,
    clip_len_frames: int = DEFAULT_CLIP_LEN_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    training_videos: tuple[str, ...] = TRAINING_SPLIT_VIDEOS,
) -> dict[str, Any]:
    """Select, materialise the clips view, derive the manifest, then the training view."""
    reject_held_out(training_videos)
    candidates = discover_candidate_tracks(dataset_root, training_videos, min_frames)
    if not candidates:
        raise ProbeSetError(
            [f"no eligible tracks under {dataset_root} for videos {training_videos}"]
        )
    selected = select_probe_clips(candidates, seed, num_clips, clip_len_frames)
    if len(selected) < num_clips:
        raise ProbeSetError(
            [
                f"only materialised {len(selected)}/{num_clips} probe clips "
                f"({len(candidates)} candidate tracks)"
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / CLIPS_VIEW_NAME
    _replace_tree(clips_dir)
    for clip in selected:
        materialize_clip(dataset_root, clips_dir, clip)

    manifest = manifest_from_view(
        clips_dir,
        seed=seed,
        clip_len_frames=clip_len_frames,
        min_frames=min_frames,
        training_videos=training_videos,
        dataset_root=dataset_root,
    )
    if manifest["num_probe_clips"] != num_clips:
        raise ProbeSetError(
            [
                f"view contains {manifest['num_probe_clips']} tracks after "
                f"materialising {len(selected)} clips; expected {num_clips}"
            ]
        )

    training_view = output_dir / TRAINING_VIEW_NAME
    _replace_tree(training_view)
    materialize_training_view(
        dataset_root=dataset_root,
        output_dir=training_view,
        training_videos=training_videos,
        excluded_training_keys=set(manifest["excluded_training_keys"]),
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return manifest
