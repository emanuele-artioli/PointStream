"""Load aligned frames from the rebuilt probe set.

Channels are paired by position in the sorted ``frame_*.png`` list, never by
reconstructing a filename. Crop / canny / pose use global source ids;
``_skeleton`` is track-local from zero. 44% of tracks carry that offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from experiments.probe_set.schema import HELD_OUT_VIDEOS, SCHEMA_ID, TRAINING_SPLIT_VIDEOS
from src.components.generation._numpy import as_chw

DEFAULT_PROBE_ROOT = Path("assets") / "probe_set"
DEFAULT_KEYFRAME = 0
HEADLINE_OFFSET = 24
DEFAULT_OFFSETS = (8, 16, 24, 32)

#: Clip mode drives a temporal model over a *contiguous* run of frames, because
#: its motion module attends across adjacent timesteps. Sparse offsets like
#: ``DEFAULT_OFFSETS`` are not a clip. Offsets 1..8 also span the regime where
#: the static-copy floor moves most (21.5 dB down to 11.2 dB, plans/done/RESEARCH-HISTORY.md §2.5),
#: which is where an engine has room to show a difference.
CLIP_LENGTH = 8
CLIP_MODE_OFFSETS = tuple(range(1, CLIP_LENGTH + 1))


@dataclass(frozen=True)
class ProbeFrame:
    """One track-local frame, channels already paired by index."""

    key: str
    video: str
    scene: str
    track: str
    frame_index: int
    n_frames: int
    appearance_rgb: np.ndarray
    object_mask: np.ndarray
    pose_rgb: np.ndarray
    canny: np.ndarray
    motion_field: np.ndarray
    split: str
    caption: str | None = None


@dataclass(frozen=True)
class CodingSample:
    """Appearance from a keyframe, conditioning and reference from a later frame."""

    key: str
    video: str
    scene: str
    track: str
    appearance_frame_index: int
    target_frame_index: int
    offset: int
    n_frames: int
    appearance_rgb: np.ndarray
    reference_rgb: np.ndarray
    object_mask: np.ndarray
    pose_rgb: np.ndarray
    canny: np.ndarray
    motion_field: np.ndarray
    split: str
    caption: str | None = None


@dataclass(frozen=True)
class ProbeClip:
    key: str
    video: str
    scene: str
    track: str
    path: Path
    n_frames: int
    split: str
    record: dict[str, Any]
    caption: str | None = None


COLOUR_WORDS = (
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "orange",
    "purple",
    "pink",
    "navy",
    "maroon",
    "gold",
    "silver",
    "grey",
    "gray",
    "teal",
    "cyan",
    "magenta",
    "brown",
    "beige",
)


def caption_names_a_colour(caption: str | None) -> bool:
    if not caption:
        return False
    text = caption.lower()
    return any(word in text.split() or word in text for word in COLOUR_WORDS)


def load_track_caption(
    track_dir: Path,
    *,
    video: str | None = None,
    scene: str | None = None,
    track: str | None = None,
) -> str | None:
    """Per-track BLIP sidecar. Probe-set copy first, then the dataset original.

    Training read ``{track_dir.parent}/{track_dir.name}_caption.json``. Copying
    that file into the probe set at materialise time is cleaner than reaching
    across; this loader still falls back to ``assets/dataset`` so a probe set
    built before that copy still finds the captions.
    """
    sidecar = track_dir.parent / f"{track_dir.name}_caption.json"
    text = _caption_from_json(sidecar)
    if text:
        return text
    if video and scene and track:
        dataset = repo_root() / "assets" / "dataset" / video / "segmentations" / scene / f"{track}_caption.json"
        return _caption_from_json(dataset)
    return None


def _caption_from_json(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, dict):
        raw = payload.get("caption") or payload.get("text") or payload.get("prompt")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def probe_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return repo_root() / DEFAULT_PROBE_ROOT


def load_manifest(root: Path | None = None) -> dict[str, Any]:
    path = probe_root(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"probe-set manifest missing at {path}")
    return json.loads(path.read_text())


def list_clips(root: Path | None = None) -> tuple[ProbeClip, ...]:
    base = probe_root(root)
    manifest = load_manifest(base)
    schema = manifest.get("schema")
    if schema != SCHEMA_ID:
        raise ValueError(f"probe harness wants {SCHEMA_ID}, got {schema!r} at {base}")
    clips: list[ProbeClip] = []
    for record in manifest["probe_clips"]:
        video = str(record["video"])
        if video in HELD_OUT_VIDEOS:
            split = "held-out"
        elif video in TRAINING_SPLIT_VIDEOS:
            split = "train"
        else:
            split = "unknown"
        rel = Path(str(record["path"]))
        track_dir = (base / rel) if not rel.is_absolute() else rel
        if not track_dir.is_dir():
            track_dir = base / "clips" / video / str(record["scene"]) / str(record["track"])
        clips.append(
            ProbeClip(
                key=str(record["key"]),
                video=video,
                scene=str(record["scene"]),
                track=str(record["track"]),
                path=track_dir,
                n_frames=int(record["num_frames"]),
                split=split,
                record=record,
                caption=load_track_caption(
                    track_dir, video=video, scene=str(record["scene"]), track=str(record["track"])
                ),
            )
        )
    return tuple(clips)


def _sorted_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing probe channel directory: {directory}")
    frames = sorted(directory.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png under {directory}")
    return frames


def _load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = rgba[:, :, 3] > 0
    return rgb, mask


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_gray(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return array


def _condition_dir(track_dir: Path, suffix: str) -> Path:
    return track_dir.parent / f"{track_dir.name}{suffix}"


def _optical_flow(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    import cv2

    nxt_rgb = next_rgb
    if nxt_rgb.shape[:2] != prev_rgb.shape[:2]:
        nxt_rgb = cv2.resize(
            nxt_rgb, (prev_rgb.shape[1], prev_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    nxt = cv2.cvtColor(nxt_rgb, cv2.COLOR_RGB2GRAY)
    seed_flow = np.zeros((*prev.shape, 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, seed_flow, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.transpose(np.asarray(flow, dtype=np.float32), (2, 0, 1))


def _channel_files(clip: ProbeClip) -> tuple[list[Path], list[Path], list[Path]]:
    crop_files = _sorted_frames(clip.path)
    pose_files = _sorted_frames(_condition_dir(clip.path, "_skeleton"))
    canny_files = _sorted_frames(_condition_dir(clip.path, "_canny"))
    for label, files in (
        ("skeleton", pose_files),
        ("canny", canny_files),
    ):
        if len(files) != len(crop_files):
            raise ValueError(
                f"{clip.key} {label} has {len(files)} frames, crop has {len(crop_files)}. "
                "Channels must be paired by position."
            )
    return crop_files, pose_files, canny_files


def _motion_at(crop_files: list[Path], frame_index: int, appearance: np.ndarray) -> np.ndarray:
    neighbor = frame_index + 1 if frame_index + 1 < len(crop_files) else frame_index - 1
    neighbor_rgb, _ = _load_rgba(crop_files[neighbor])
    if neighbor < frame_index:
        return _optical_flow(neighbor_rgb, appearance)
    return _optical_flow(appearance, neighbor_rgb)


def load_frame(clip: ProbeClip, frame_index: int) -> ProbeFrame:
    """Load appearance, pose, canny, mask, and a motion field at ``frame_index``.

    Pairing is by position: the Nth file in each channel directory. A filename
    reconstructed from a global source id is how the unaligned v2 set went
    silent; this path does not do that.
    """
    if frame_index < 0 or frame_index >= clip.n_frames:
        raise IndexError(
            f"frame_index {frame_index} out of range for {clip.key} "
            f"({clip.n_frames} frames)"
        )
    crop_files, pose_files, canny_files = _channel_files(clip)
    appearance, mask = _load_rgba(crop_files[frame_index])
    pose = _load_rgb(pose_files[frame_index])
    canny = _load_gray(canny_files[frame_index])
    motion = _motion_at(crop_files, frame_index, appearance)
    return ProbeFrame(
        key=clip.key,
        video=clip.video,
        scene=clip.scene,
        track=clip.track,
        frame_index=frame_index,
        n_frames=clip.n_frames,
        appearance_rgb=appearance,
        object_mask=np.asarray(mask, dtype=bool),
        pose_rgb=pose,
        canny=canny,
        motion_field=motion,
        split=clip.split,
        caption=clip.caption,
    )


def load_coding_sample(
    clip: ProbeClip,
    appearance_index: int,
    offset: int,
) -> CodingSample:
    """Appearance from ``appearance_index``, conditioning and reference from later.

    ``offset`` is how far the target sits after the appearance frame. It must
    be positive: offset 0 is self-reconstruction, which this loader refuses.
    """
    if offset <= 0:
        raise ValueError(
            f"coding-task offset must be a later frame than appearance; got {offset}"
        )
    target_index = appearance_index + offset
    if appearance_index < 0 or target_index >= clip.n_frames:
        raise IndexError(
            f"appearance {appearance_index} offset {offset} out of range for "
            f"{clip.key} ({clip.n_frames} frames)"
        )
    crop_files, pose_files, canny_files = _channel_files(clip)
    if target_index >= len(crop_files) or appearance_index >= len(crop_files):
        raise IndexError(
            f"appearance {appearance_index} offset {offset} out of range for "
            f"{clip.key} ({len(crop_files)} crop files)"
        )
    appearance, _ = _load_rgba(crop_files[appearance_index])
    reference, mask = _load_rgba(crop_files[target_index])
    pose = _load_rgb(pose_files[target_index])
    canny = _load_gray(canny_files[target_index])
    motion = _motion_at(crop_files, target_index, reference)
    return CodingSample(
        key=clip.key,
        video=clip.video,
        scene=clip.scene,
        track=clip.track,
        appearance_frame_index=appearance_index,
        target_frame_index=target_index,
        offset=offset,
        n_frames=clip.n_frames,
        appearance_rgb=appearance,
        reference_rgb=reference,
        object_mask=np.asarray(mask, dtype=bool),
        pose_rgb=pose,
        canny=canny,
        motion_field=motion,
        split=clip.split,
        caption=clip.caption,
    )


def bundle_arrays(frame: ProbeFrame) -> dict[str, Any]:
    """Named arrays for a self-reconstruction ``ConditioningBundle``."""
    return {
        "appearance": as_chw(frame.appearance_rgb),
        "pose": as_chw(frame.pose_rgb),
        "mask": np.where(frame.object_mask, np.uint8(255), np.uint8(0)),
        "canny": frame.canny,
        "motion_field": frame.motion_field,
        "frame_index": frame.frame_index,
        "object_id": frame.key,
        "caption": frame.caption,
    }


def bundle_coding(sample: CodingSample) -> dict[str, Any]:
    """Named arrays: appearance from the keyframe, everything else from the target."""
    return {
        "appearance": as_chw(sample.appearance_rgb),
        "pose": as_chw(sample.pose_rgb),
        "mask": np.where(sample.object_mask, np.uint8(255), np.uint8(0)),
        "canny": sample.canny,
        "motion_field": sample.motion_field,
        "frame_index": sample.target_frame_index,
        "object_id": sample.key,
        "caption": sample.caption,
    }


def load_coding_sequence(
    clip: ProbeClip,
    appearance_index: int,
    offsets: Sequence[int],
) -> tuple[CodingSample, ...]:
    """One ``CodingSample`` per offset, sharing a single appearance keyframe.

    Clip mode hands all of these to ``generate_sequence`` in one call. Offsets
    are sorted and de-duplicated so the sequence is monotonic in time — a
    temporal model handed shuffled poses is being driven wrongly, which is the
    kind of fault this project keeps finding after the fact.
    """
    ordered = sorted({int(offset) for offset in offsets})
    if not ordered:
        raise ValueError("a clip-mode sequence needs at least one offset")
    return tuple(load_coding_sample(clip, appearance_index, offset) for offset in ordered)


def with_appearance(sample: CodingSample, appearance_rgb: np.ndarray) -> CodingSample:
    """The same coding sample with a different keyframe as its appearance.

    This is the cross-appearance control: hold the model, the pose, the target
    and the metric fixed and vary *only* which player is shown to the engine.
    The reference stays the true target, so an engine that uses appearance
    scores worse with someone else's keyframe.
    """
    return replace(sample, appearance_rgb=np.asarray(appearance_rgb, dtype=np.uint8))
