"""Load one aligned frame (or a cheap set) from the rebuilt probe set.

Channels are paired by position in the sorted ``frame_*.png`` list, never by
reconstructing a filename. That is the pose-offset fault BP7 closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.probe_set.schema import HELD_OUT_VIDEOS, SCHEMA_ID, TRAINING_SPLIT_VIDEOS
from src.components.generation._numpy import as_chw

DEFAULT_PROBE_ROOT = Path("assets") / "probe_set"
DIFFUSION_FRAME_INDEX = 24
ONE_PASS_FRAME_INDICES = (0, 16, 24, 32, 47)


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
    appearance, mask = _load_rgba(crop_files[frame_index])
    pose = _load_rgb(pose_files[frame_index])
    canny = _load_gray(canny_files[frame_index])
    neighbor = frame_index + 1 if frame_index + 1 < len(crop_files) else frame_index - 1
    neighbor_rgb, _ = _load_rgba(crop_files[neighbor])
    if neighbor < frame_index:
        motion = _optical_flow(neighbor_rgb, appearance)
    else:
        motion = _optical_flow(appearance, neighbor_rgb)
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
    )


def bundle_arrays(frame: ProbeFrame) -> dict[str, Any]:
    """Named arrays for ``ConditioningBundle`` construction."""
    return {
        "appearance": as_chw(frame.appearance_rgb),
        "pose": as_chw(frame.pose_rgb),
        "mask": np.where(frame.object_mask, np.uint8(255), np.uint8(0)),
        "canny": frame.canny,
        "motion_field": frame.motion_field,
        "frame_index": frame.frame_index,
        "object_id": frame.key,
    }
