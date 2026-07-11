"""Fixed, seeded probe set for G2 checkpoint quality tracking (report 10, Phase 5.4).

Picks a small, deterministic sample of short clips from the curated dataset
(`assets/dataset/<video>/segmentations/scene_*/track_*`) — but *only* from the
training-split videos (`alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`,
`federer_djokovic`, `sinner_alcaraz` — report 10's "Methodology locked
2026-07-11" section). The held-out test videos (`alcaraz_highlights`,
`djokovic_zverev`) are never touched by this script and are rejected outright
if named explicitly.

This is a *second-level* split within the training-split videos: the probe
clips selected here must never appear in whatever a training run feeds into
`scripts/train_controlnet.py` / `train_pix2pix.py` / `train_spade4tennis.py`,
or "quality improving on the probe set" would just mean memorization rather
than generalization. `--materialize-training-view` builds exactly the
filtered `--data-root` those scripts should be pointed at: a symlink tree of
the training-split videos with every probe-selected track omitted at the
track granularity (matching the finest unit those scripts already operate at
— `TennisSkeletonDataset`/`ControlNetDataset` glob `*/segmentations/scene_*/
track_*` and sample reference frames per-track, so excluding a whole track is
the natural, simplest non-overlap boundary; not excluding at finer
frame-level granularity is a deliberate scope choice, noted here rather than
left implicit).

Usage
-----
    conda run -n pointstream python scripts/select_probe_set.py \\
        --dataset-root assets/dataset --num-clips 12 --clip-len-frames 48 \\
        --seed 20260712 --manifest assets/probe_set/manifest.json \\
        --materialize-training-view assets/probe_set/training_view
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

TRAINING_SPLIT_VIDEOS = (
    "alcaraz_perricard",
    "alcaraz_ruud",
    "djokovic_federer",
    "federer_djokovic",
    "sinner_alcaraz",
)
HELD_OUT_VIDEOS = ("alcaraz_highlights", "djokovic_zverev")

_TRACK_ID_RE = re.compile(r"^(track_\d+)")
_FRAME_ID_RE = re.compile(r"frame_(\d+)\.png$")


@dataclass(frozen=True)
class TrackCandidate:
    """One primary (color) track directory eligible for probe selection."""

    video: str
    scene: str
    track: str
    frame_ids: tuple[int, ...]  # sorted ascending

    @property
    def key(self) -> str:
        return f"{self.video}/{self.scene}/{self.track}"


@dataclass(frozen=True)
class ProbeClip:
    """A contiguous frame window sampled from one TrackCandidate."""

    video: str
    scene: str
    track: str
    frame_ids: tuple[int, ...]  # sorted ascending, contiguous by index into the track's own frame list

    @property
    def key(self) -> str:
        return f"{self.video}/{self.scene}/{self.track}"


def _extract_track_id(name: str) -> str | None:
    """'track_0021_skeleton' -> 'track_0021'; 'track_0021_caption.json' -> 'track_0021'; else None."""
    match = _TRACK_ID_RE.match(name)
    return match.group(1) if match else None


def discover_candidate_tracks(
    dataset_root: Path,
    videos: tuple[str, ...],
    min_frames: int,
) -> list[TrackCandidate]:
    """Enumerate primary track directories with >= min_frames color frames and a sibling skeleton dir.

    Deterministic ordering (sorted) so downstream sampling is reproducible
    regardless of filesystem iteration order.
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
                if p.is_dir() and _extract_track_id(p.name) == p.name  # primary track dir, no suffix
            )
            for track_dir in track_dirs:
                skeleton_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
                if not skeleton_dir.is_dir():
                    continue
                frame_ids = sorted(
                    int(m.group(1))
                    for f in track_dir.glob("frame_*.png")
                    if (m := _FRAME_ID_RE.search(f.name)) is not None
                )
                if len(frame_ids) < min_frames:
                    continue
                candidates.append(
                    TrackCandidate(video=video, scene=scene_dir.name, track=track_dir.name, frame_ids=tuple(frame_ids))
                )
    return candidates


def select_probe_clips(
    candidates: list[TrackCandidate],
    seed: int,
    num_clips: int,
    clip_len_frames: int,
) -> list[ProbeClip]:
    """Deterministically sample `num_clips` clips, round-robin across videos for diversity.

    Each selected track contributes one contiguous window of up to
    `clip_len_frames` consecutive (by sorted order, not necessarily
    frame-number-adjacent) frames, starting at a seeded random offset.
    Round-robins across the distinct videos present in `candidates` so the
    probe set spreads across videos/lighting conditions rather than
    clustering in whichever video happens to sort first.
    """
    if num_clips <= 0:
        return []

    rng = random.Random(seed)

    by_video: dict[str, list[TrackCandidate]] = {}
    for cand in candidates:
        by_video.setdefault(cand.video, []).append(cand)

    videos_order = sorted(by_video.keys())
    for video in videos_order:
        rng.shuffle(by_video[video])

    selected: list[ProbeClip] = []
    video_cursor = {video: 0 for video in videos_order}
    round_robin_idx = 0
    guard = 0
    max_attempts = max(1, sum(len(v) for v in by_video.values())) * 2 + num_clips * 4

    while len(selected) < num_clips and videos_order and guard < max_attempts:
        guard += 1
        video = videos_order[round_robin_idx % len(videos_order)]
        round_robin_idx += 1

        pool = by_video[video]
        cursor = video_cursor[video]
        if cursor >= len(pool):
            # Exhausted this video's candidates; skip it going forward.
            if all(video_cursor[v] >= len(by_video[v]) for v in videos_order):
                break
            continue

        candidate = pool[cursor]
        video_cursor[video] = cursor + 1

        frame_ids = candidate.frame_ids
        window_len = min(clip_len_frames, len(frame_ids))
        max_start = len(frame_ids) - window_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        window = frame_ids[start : start + window_len]

        selected.append(
            ProbeClip(video=candidate.video, scene=candidate.scene, track=candidate.track, frame_ids=tuple(window))
        )

    return selected


def build_manifest(
    probe_clips: list[ProbeClip],
    seed: int,
    clip_len_frames: int,
    min_frames: int,
    training_videos: tuple[str, ...],
) -> dict:
    return {
        "schema": "pointstream.probe_set.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "clip_len_frames": clip_len_frames,
        "min_frames": min_frames,
        "training_videos": list(training_videos),
        "held_out_videos": list(HELD_OUT_VIDEOS),
        "num_probe_clips": len(probe_clips),
        "probe_clips": [
            {
                "video": clip.video,
                "scene": clip.scene,
                "track": clip.track,
                "frame_ids": list(clip.frame_ids),
                "num_frames": len(clip.frame_ids),
            }
            for clip in probe_clips
        ],
        # Track-level keys excluded from training (see module docstring for why
        # track granularity, not frame granularity, is the exclusion boundary).
        "excluded_training_keys": sorted({clip.key for clip in probe_clips}),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def materialize_training_view(
    dataset_root: Path,
    output_dir: Path,
    training_videos: tuple[str, ...],
    excluded_training_keys: set[str],
) -> None:
    """Build a symlink tree at output_dir suitable for `--data-root`.

    Mirrors dataset_root, restricted to `training_videos`, with every track
    directory (and its `_skeleton`/`_canny`/`_caption.json`/`_metadata.json`/
    `_keypoints.json` siblings) named in `excluded_training_keys`
    ("<video>/<scene>/<track>") omitted. Whole video directories / whole scene
    directories with nothing excluded are symlinked wholesale (cheap, no need
    to descend); only scenes containing an excluded track are rebuilt
    entry-by-entry.
    """
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing training view at {output_dir}")
    output_dir.mkdir(parents=True)

    excluded_by_video_scene: dict[tuple[str, str], set[str]] = {}
    for key in excluded_training_keys:
        video, scene, track = key.split("/")
        excluded_by_video_scene.setdefault((video, scene), set()).add(track)

    for video in training_videos:
        src_video_dir = dataset_root / video
        if not src_video_dir.is_dir():
            continue
        dst_video_dir = output_dir / video
        dst_video_dir.mkdir()

        for entry in sorted(src_video_dir.iterdir()):
            if entry.name != "segmentations":
                (dst_video_dir / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
                continue

            dst_seg_dir = dst_video_dir / "segmentations"
            dst_seg_dir.mkdir()
            for scene_dir in sorted(entry.iterdir()):
                excluded_tracks = excluded_by_video_scene.get((video, scene_dir.name))
                if not excluded_tracks:
                    (dst_seg_dir / scene_dir.name).symlink_to(scene_dir.resolve(), target_is_directory=True)
                    continue

                dst_scene_dir = dst_seg_dir / scene_dir.name
                dst_scene_dir.mkdir()
                for item in sorted(scene_dir.iterdir()):
                    track_id = _extract_track_id(item.name)
                    if track_id is not None and track_id in excluded_tracks:
                        continue
                    (dst_scene_dir / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=Path("assets/dataset"))
    parser.add_argument(
        "--videos",
        nargs="+",
        default=list(TRAINING_SPLIT_VIDEOS),
        help="Training-split videos to sample from (default: all 5). Held-out videos are always rejected.",
    )
    parser.add_argument("--num-clips", type=int, default=12)
    parser.add_argument("--clip-len-frames", type=int, default=48)
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum track length to be eligible")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--manifest", type=Path, default=Path("assets/probe_set/manifest.json"))
    parser.add_argument(
        "--materialize-training-view",
        type=Path,
        default=None,
        help="If set, also build a symlink tree here for use as training scripts' --data-root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    held_out_requested = sorted(set(args.videos) & set(HELD_OUT_VIDEOS))
    if held_out_requested:
        parser.error(
            f"refusing to sample from held-out test video(s) {held_out_requested} — "
            "report 10's locked methodology reserves these for final evaluation only"
        )
    unknown = sorted(set(args.videos) - set(TRAINING_SPLIT_VIDEOS))
    if unknown:
        parser.error(f"unknown video(s) not in the training split: {unknown}")

    training_videos = tuple(args.videos)
    candidates = discover_candidate_tracks(args.dataset_root, training_videos, args.min_frames)
    if not candidates:
        parser.error(f"no eligible tracks found under {args.dataset_root} for videos {training_videos}")

    probe_clips = select_probe_clips(candidates, args.seed, args.num_clips, args.clip_len_frames)
    if len(probe_clips) < args.num_clips:
        print(
            f"warning: only found {len(probe_clips)}/{args.num_clips} eligible probe clips "
            f"({len(candidates)} candidate tracks total)"
        )

    manifest = build_manifest(probe_clips, args.seed, args.clip_len_frames, args.min_frames, training_videos)
    write_manifest(manifest, args.manifest)
    print(f"Wrote probe set manifest ({len(probe_clips)} clips) to {args.manifest}")

    videos_seen = sorted({clip.video for clip in probe_clips})
    print(f"Videos represented: {videos_seen}")

    if args.materialize_training_view is not None:
        materialize_training_view(
            dataset_root=args.dataset_root,
            output_dir=args.materialize_training_view,
            training_videos=training_videos,
            excluded_training_keys=set(manifest["excluded_training_keys"]),
        )
        print(f"Materialized probe-excluded training view at {args.materialize_training_view}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
