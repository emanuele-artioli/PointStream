"""Seeded clip selection. Same rule as the v1 script, so the same seed
selects the same tracks; the coordinate system those tracks are *recorded*
in is the v2 change.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from experiments.probe_set.schema import HELD_OUT_VIDEOS, TRAINING_SPLIT_VIDEOS

TRACK_PRIMARY_RE = re.compile(r"^track_\d+$")
TRACK_ID_PREFIX_RE = re.compile(r"^(track_\d+)")
FRAME_ID_RE = re.compile(r"^frame_(\d+)\.png$")


@dataclass(frozen=True)
class TrackCandidate:
    """One primary colour-track directory eligible for probe selection."""

    video: str
    scene: str
    track: str
    source_frame_ids: tuple[int, ...]  # sorted; the numbers in the PNG names

    @property
    def key(self) -> str:
        return f"{self.video}/{self.scene}/{self.track}"


@dataclass(frozen=True)
class ProbeClip:
    """A contiguous window (by sorted source ids) sampled from one track."""

    video: str
    scene: str
    track: str
    source_frame_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.video}/{self.scene}/{self.track}"


def extract_track_id(name: str) -> str | None:
    """``track_0021_skeleton`` -> ``track_0021``; ``track_0021`` -> ``track_0021``."""
    match = TRACK_ID_PREFIX_RE.match(name)
    return match.group(1) if match else None


def list_source_frame_ids(track_dir: Path) -> tuple[int, ...]:
    """Frame numbers encoded in ``frame_XXXXXX.png`` names, sorted."""
    ids = sorted(
        int(match.group(1))
        for path in track_dir.glob("frame_*.png")
        if (match := FRAME_ID_RE.match(path.name)) is not None
    )
    return tuple(ids)


def discover_candidate_tracks(
    dataset_root: Path,
    videos: tuple[str, ...],
    min_frames: int,
) -> list[TrackCandidate]:
    """Primary tracks with >= min_frames colour frames and a sibling skeleton dir.

    Walk order is sorted, so sampling is independent of filesystem iteration.
    """
    candidates: list[TrackCandidate] = []
    for video in videos:
        seg_root = dataset_root / video / "segmentations"
        if not seg_root.is_dir():
            continue
        for scene_dir in sorted(p for p in seg_root.iterdir() if p.is_dir()):
            track_dirs = sorted(
                p
                for p in scene_dir.iterdir()
                if p.is_dir() and TRACK_PRIMARY_RE.match(p.name)
            )
            for track_dir in track_dirs:
                skeleton_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
                if not skeleton_dir.is_dir():
                    continue
                source_ids = list_source_frame_ids(track_dir)
                if len(source_ids) < min_frames:
                    continue
                candidates.append(
                    TrackCandidate(
                        video=video,
                        scene=scene_dir.name,
                        track=track_dir.name,
                        source_frame_ids=source_ids,
                    )
                )
    return candidates


def select_probe_clips(
    candidates: list[TrackCandidate],
    seed: int,
    num_clips: int,
    clip_len_frames: int,
) -> list[ProbeClip]:
    """Round-robin across videos; one seeded contiguous window per selected track."""
    if num_clips <= 0:
        return []

    rng = random.Random(seed)
    by_video: dict[str, list[TrackCandidate]] = {}
    for candidate in candidates:
        by_video.setdefault(candidate.video, []).append(candidate)

    videos_order = sorted(by_video.keys())
    for video in videos_order:
        rng.shuffle(by_video[video])

    selected: list[ProbeClip] = []
    video_cursor = {video: 0 for video in videos_order}
    round_robin_idx = 0
    guard = 0
    max_attempts = max(1, sum(len(pool) for pool in by_video.values())) * 2 + num_clips * 4

    while len(selected) < num_clips and videos_order and guard < max_attempts:
        guard += 1
        video = videos_order[round_robin_idx % len(videos_order)]
        round_robin_idx += 1
        pool = by_video[video]
        cursor = video_cursor[video]
        if cursor >= len(pool):
            if all(video_cursor[name] >= len(by_video[name]) for name in videos_order):
                break
            continue
        candidate = pool[cursor]
        video_cursor[video] = cursor + 1
        source_ids = candidate.source_frame_ids
        window_len = min(clip_len_frames, len(source_ids))
        max_start = len(source_ids) - window_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        window = source_ids[start : start + window_len]
        selected.append(
            ProbeClip(
                video=candidate.video,
                scene=candidate.scene,
                track=candidate.track,
                source_frame_ids=tuple(window),
            )
        )
    return selected


def reject_held_out(videos: tuple[str, ...]) -> None:
    """Refuse to sample probe clips from the locked held-out pair."""
    leaked = sorted(set(videos) & set(HELD_OUT_VIDEOS))
    if leaked:
        raise ValueError(
            f"refusing to sample from held-out video(s) {leaked} — "
            "the locked split reserves these for eval-general"
        )
    unknown = sorted(set(videos) - set(TRAINING_SPLIT_VIDEOS))
    if unknown:
        raise ValueError(f"unknown video(s) not in the training split: {unknown}")
