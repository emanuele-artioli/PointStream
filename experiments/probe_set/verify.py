"""Verifier for a probe-set tree.

Fails on the faults that made v1 silently unusable: a view that does not
contain the tracks the manifest names, named frames that are not on disk,
and a coordinate system that is not the one the files use. A broken symlink
and a symlink to the *wrong* track are different failures; existence is not
identity.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.contracts import paths as ps_paths
from typing import Any

from experiments.probe_set.schema import (
    CLIPS_VIEW_NAME,
    CONDITION_DIR_SUFFIXES,
    COORDINATE_SYSTEM,
    HELD_OUT_VIDEOS,
    LEGACY_SCHEMA_ID,
    SCHEMA_ID,
    TRAINING_SPLIT_VIDEOS,
    TRAINING_VIEW_NAME,
    ProbeSetError,
)
from experiments.probe_set.select import TRACK_PRIMARY_RE, list_source_frame_ids


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _clip_key(clip: dict[str, Any]) -> str:
    if "key" in clip:
        return str(clip["key"])
    return f"{clip['video']}/{clip['scene']}/{clip['track']}"


def iter_view_track_dirs(view_dir: Path) -> list[Path]:
    """Primary track directories under a clips-style *or* dataset-style view.

    Clips view: ``<view>/<video>/<scene>/track_NNNN``.
    Dataset / training view: ``<view>/<video>/segmentations/<scene>/track_NNNN``.
    """
    found: list[Path] = []
    if not view_dir.is_dir():
        return found
    for video_dir in sorted(p for p in view_dir.iterdir() if p.is_dir() or p.is_symlink()):
        seg = video_dir / "segmentations"
        scene_parents = [seg] if seg.is_dir() else [video_dir]
        for parent in scene_parents:
            if not parent.is_dir():
                continue
            for scene_dir in sorted(p for p in parent.iterdir() if p.is_dir() or p.is_symlink()):
                if not scene_dir.is_dir():
                    continue
                for item in sorted(scene_dir.iterdir()):
                    if (item.is_dir() or item.is_symlink()) and TRACK_PRIMARY_RE.match(item.name):
                        found.append(item)
    return found


def view_track_key(track_dir: Path, view_dir: Path) -> str:
    """``video/scene/track`` from the logical view path, not the symlink target.

    Resolving first would turn a wholesale scene symlink into a dataset path,
    which is the identity we want to *check*, not the key we want to *name*.
    """
    parts = track_dir.relative_to(view_dir).parts
    if "segmentations" in parts:
        video = parts[0]
        scene = parts[parts.index("segmentations") + 1]
        track = parts[-1]
        return f"{video}/{scene}/{track}"
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/{parts[-1]}"
    raise ValueError(f"cannot parse track key from {track_dir} under {view_dir}")


def _anchor(path: Path) -> Path:
    """Resolve a manifest path, anchoring a relative one at the **data root**.

    The manifest records `assets/dataset/...` rather than a machine path, which
    is right — a probe set that hardcodes one checkout's location is not
    portable between them. But a relative path has to be resolved against
    something, and `Path.resolve()` silently uses the *current working
    directory*. That was harmlessly correct while `assets/` sat in the checkout
    and everything ran from the repo root; after the 2026-08-29 move it points
    at a directory that does not exist, and every containment check fails
    against a track it names correctly.

    `src.contracts.paths` is the one place that knows where the data is, so the
    manifest's relative paths are anchored there. An absolute path in a
    manifest is left alone.
    """
    if path.is_absolute():
        return path
    return ps_paths.data_root() / path


def _resolved_file(path: Path) -> Path | None:
    """Follow a symlink. None if missing or dangling."""
    if path.is_symlink():
        if not path.exists():
            return None
        return path.resolve()
    if path.is_file():
        return path.resolve()
    return None


def locked_split_violations(manifest: dict[str, Any]) -> list[str]:
    """The 5-train / 2-held-out split recorded 2026-07-11 must not drift."""
    violations: list[str] = []
    training = set(manifest.get("training_videos") or [])
    held = set(manifest.get("held_out_videos") or [])
    if training != set(TRAINING_SPLIT_VIDEOS):
        violations.append(
            f"training_videos {sorted(training)} != locked split {list(TRAINING_SPLIT_VIDEOS)}"
        )
    if held != set(HELD_OUT_VIDEOS):
        violations.append(
            f"held_out_videos {sorted(held)} != locked split {list(HELD_OUT_VIDEOS)}"
        )
    if training & held:
        violations.append(f"training and held-out overlap: {sorted(training & held)}")
    return violations


def collect_violations(
    root: Path,
    *,
    dataset_root: Path | None = None,
    check_locked_split: bool = False,
) -> list[str]:
    """Return every invariant failure. Empty means the tree is usable."""
    violations: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]

    try:
        manifest = load_manifest(manifest_path)
    except json.JSONDecodeError as exc:
        return [f"manifest is not JSON: {exc}"]

    schema = manifest.get("schema")
    if schema == LEGACY_SCHEMA_ID:
        violations.append(
            f"schema {LEGACY_SCHEMA_ID!r} is not trustworthy; need {SCHEMA_ID}"
        )
    elif schema != SCHEMA_ID:
        violations.append(f"schema {schema!r} is not {SCHEMA_ID}")

    coordinate_system = manifest.get("coordinate_system")
    if coordinate_system != COORDINATE_SYSTEM:
        violations.append(
            f"coordinate_system {coordinate_system!r} is not {COORDINATE_SYSTEM!r}"
        )

    if check_locked_split:
        violations.extend(locked_split_violations(manifest))

    training = set(manifest.get("training_videos") or [])
    held = set(manifest.get("held_out_videos") or [])
    if training & held:
        violations.append(f"training and held-out overlap: {sorted(training & held)}")

    clips = list(manifest.get("probe_clips") or [])
    claimed_num = manifest.get("num_probe_clips")
    if claimed_num is not None and claimed_num != len(clips):
        violations.append(
            f"num_probe_clips={claimed_num} but probe_clips has {len(clips)} entries"
        )

    declared_view = str(manifest.get("view") or CLIPS_VIEW_NAME)
    view_dir = root / declared_view
    if not view_dir.is_dir():
        fallback = root / TRAINING_VIEW_NAME
        if fallback.is_dir():
            violations.append(
                f"declared view {declared_view!r} is missing; "
                f"checking {TRAINING_VIEW_NAME!r} for identity (this is the v1 layout)"
            )
            view_dir = fallback
        else:
            violations.append(f"view directory missing: {view_dir}")
            return violations

    manifest_keys = [_clip_key(clip) for clip in clips]
    manifest_key_set = set(manifest_keys)
    if len(manifest_keys) != len(manifest_key_set):
        violations.append(f"manifest names duplicate tracks: {manifest_keys}")

    resolved_dataset = dataset_root
    if resolved_dataset is None and manifest.get("dataset_root"):
        resolved_dataset = _anchor(Path(str(manifest["dataset_root"])))

    for clip in clips:
        key = _clip_key(clip)
        video = str(clip.get("video", ""))
        scene = str(clip.get("scene", ""))
        track = str(clip.get("track", ""))
        track_dir = view_dir / video / scene / track
        if not track_dir.exists():
            # Dataset-style layout used by the v1 training_view.
            alt = view_dir / video / "segmentations" / scene / track
            if alt.exists():
                track_dir = alt
            else:
                violations.append(f"{key}: track directory missing under {view_dir}")
                continue

        frame_ids = [int(fid) for fid in clip.get("frame_ids") or []]
        claimed_count = clip.get("num_frames")
        if claimed_count is not None and int(claimed_count) != len(frame_ids):
            violations.append(
                f"{key}: num_frames={claimed_count} but frame_ids has {len(frame_ids)}"
            )
        on_disk = list_source_frame_ids(track_dir)
        if set(frame_ids) - set(on_disk):
            missing_ids = sorted(set(frame_ids) - set(on_disk))
            violations.append(
                f"{key}: manifest names frames that are not files: {missing_ids[:8]}"
            )
        if len(frame_ids) != len(on_disk) and set(frame_ids) <= set(on_disk):
            # Named frames exist but the directory has extras — allowed only
            # when we are looking at a dataset-style whole-track symlink (v1).
            # v2 clips views must contain exactly the named frames.
            if declared_view == CLIPS_VIEW_NAME and view_dir.name == CLIPS_VIEW_NAME:
                violations.append(
                    f"{key}: view has {len(on_disk)} frames, manifest names {len(frame_ids)}"
                )

        scene_dir = track_dir.parent
        for suffix in CONDITION_DIR_SUFFIXES:
            cond_dir = scene_dir / f"{track}{suffix}"
            if not cond_dir.is_dir():
                continue
            n_cond = len(list_source_frame_ids(cond_dir))
            if n_cond != len(on_disk):
                violations.append(
                    f"{key}: {suffix} has {n_cond} frames, crop has {len(on_disk)}"
                )

        named_source = (
            _anchor(Path(str(clip["source_track"])))
            if clip.get("source_track")
            else (
                resolved_dataset / video / "segmentations" / scene / track
                if resolved_dataset is not None
                else None
            )
        )
        for fid in frame_ids:
            frame_path = track_dir / f"frame_{int(fid):06d}.png"
            if frame_path.is_symlink() and not frame_path.exists():
                violations.append(f"{key}: broken symlink {frame_path.name}")
                continue
            resolved = _resolved_file(frame_path)
            if resolved is None:
                violations.append(f"{key}: frame_{int(fid):06d}.png is not a real file")
                continue
            if named_source is not None:
                try:
                    resolved.relative_to(named_source.resolve())
                except (ValueError, FileNotFoundError):
                    violations.append(
                        f"{key}: {frame_path.name} resolves to {resolved}, "
                        f"not under the named track {named_source}"
                    )

    view_keys = [view_track_key(path, view_dir) for path in iter_view_track_dirs(view_dir)]
    view_key_set = set(view_keys)
    missing_in_view = sorted(manifest_key_set - view_key_set)
    extra_in_view = sorted(view_key_set - manifest_key_set)
    if missing_in_view:
        violations.append(
            f"view is missing tracks the manifest names: {missing_in_view}"
        )
    if extra_in_view:
        violations.append(
            f"view contains tracks the manifest does not name: {extra_in_view}"
        )

    training_view = root / str(manifest.get("training_view") or TRAINING_VIEW_NAME)
    if training_view.is_dir():
        present_videos = {p.name for p in training_view.iterdir() if p.is_dir() or p.is_symlink()}
        leaked_held = sorted(present_videos & held)
        if leaked_held:
            violations.append(
                f"held-out video(s) present in training view: {leaked_held}"
            )
        if manifest_key_set:
            training_keys = {
                view_track_key(path, training_view)
                for path in iter_view_track_dirs(training_view)
            }
            leaked_probe = sorted(manifest_key_set & training_keys)
            if leaked_probe:
                violations.append(
                    f"probe tracks leaked into training view: {leaked_probe}"
                )
    elif clips:
        # A probe set used for eval-general honesty needs the training view so
        # the held-out exclusion is a fact on disk, not only a JSON field.
        violations.append(f"training view missing: {training_view}")

    return violations


def verify(
    root: Path,
    *,
    dataset_root: Path | None = None,
    check_locked_split: bool = False,
) -> None:
    """Raise ``ProbeSetError`` if the probe set at ``root`` is not usable."""
    violations = collect_violations(
        root, dataset_root=dataset_root, check_locked_split=check_locked_split
    )
    if violations:
        raise ProbeSetError(violations)
