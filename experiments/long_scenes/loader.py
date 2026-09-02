"""Dataset loader for BP46 validated long tennis scenes.

Provides a unified interface for downstream experiments (BP44, BP45, E1, E2)
to load verified multi-duration clips (48, 96, 192, 384 frames) along with
aligned source frames, player masks, and ObjectRequests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.headroom.real import (
    DATASET,
    PASTE_MAE_MAX,
    bbox_slices,
    list_tracks,
    load_rgb_stack,
    load_rgba,
    opaque_mae,
    pair_track,
)
from src.contracts import paths as ps_paths
from src.pipeline.reconstruction.reconstruct import ObjectRequest

BP46_CLIPS = ps_paths.outputs() / "bp46-long-scenes" / "clips"
BP21_CLIPS = ps_paths.outputs() / "bp21-headroom" / "clips"
MANIFEST_PATH = ps_paths.repo_root() / "manifests" / "bp46_long_tennis_scenes.json"


class LongSceneError(RuntimeError):
    """Raised when a requested long scene clip cannot be loaded or is invalid."""


@dataclass(frozen=True)
class LongSceneClip:
    """Loaded verified long scene with frames, masks, and object descriptors."""

    video: str
    scene: str
    context_id: str
    n_frames: int
    frames: np.ndarray
    """(T, H, W, 3) uint8 RGB."""
    masks: np.ndarray
    """(T, H, W) bool union mask of all player tracks."""
    objects: tuple[ObjectRequest, ...]
    paste_back_mae: float
    is_eligible: bool = True
    route: str = "pointstream"
    failure_reasons: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        t, h, w, _ = self.frames.shape
        return {
            "video": self.video,
            "scene": self.scene,
            "context_id": self.context_id,
            "n_frames": int(t),
            "resolution": f"{w}x{h}",
            "n_objects": len(self.objects),
            "player_pixel_fraction": float(self.masks.mean()),
            "paste_back_mae": self.paste_back_mae,
            "is_eligible": self.is_eligible,
            "route": self.route,
            "failure_reasons": list(self.failure_reasons),
        }


def get_long_scene_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Retrieve and parse the BP46 long scene manifest."""
    m_path = manifest_path or MANIFEST_PATH
    if not m_path.is_file():
        m_path = ps_paths.outputs() / "bp46-long-scenes" / "manifest.json"
    if not m_path.is_file():
        raise LongSceneError(f"BP46 manifest not found at {m_path}")

    return json.loads(m_path.read_text(encoding="utf-8"))


def load_long_scene_clip(
    video: str,
    scene: str,
    n_frames: int = 48,
    *,
    manifest_path: Path | None = None,
    allow_ineligible: bool = False,
) -> LongSceneClip:
    """Load an exact n_frames clip for the given video and scene.

    Args:
        video: Video identifier (e.g. 'alcaraz_highlights').
        scene: Scene identifier (e.g. 'scene_000').
        n_frames: Exact duration in frames (48, 96, 192, or 384).
        manifest_path: Optional explicit manifest path.
        allow_ineligible: When True, allows loading ineligible scenes (e.g. for fallback testing).

    Returns:
        LongSceneClip dataclass with exact duration frames and aligned objects.
    """
    manifest = get_long_scene_manifest(manifest_path)
    scene_record = None
    for s in manifest.get("scenes", []):
        if s.get("video") == video and s.get("scene") == scene:
            scene_record = s
            break

    if scene_record is None:
        raise LongSceneError(f"Scene {video}/{scene} not registered in BP46 manifest")

    context_id = scene_record.get("context_id", f"{video}_context")
    intv = scene_record.get("intervals", {}).get(str(n_frames))
    if intv is None:
        raise LongSceneError(f"Scene {video}/{scene} has no interval record for {n_frames} frames")
    is_eligible = intv.get("status") == "eligible"
    fail_reasons = intv.get("failure_reasons", [])
    if not is_eligible and not allow_ineligible:
        reasons = ", ".join(fail_reasons or ["not eligible"])
        raise LongSceneError(f"Scene {video}/{scene} is not eligible for {n_frames} frames: {reasons}")

    start_frame = int(intv.get("start_frame", 0))
    end_frame = int(intv.get("end_frame", 0))
    expected_count = end_frame - start_frame
    if expected_count != n_frames:
        if allow_ineligible:
            start_frame = 0
            end_frame = n_frames
        else:
            raise LongSceneError(
                f"{video}/{scene} interval [{start_frame}:{end_frame}] length {expected_count} != {n_frames}"
            )

    # Locate source frames
    extract_dir = BP46_CLIPS / video / scene / "extract_24"
    if not extract_dir.is_dir():
        extract_dir = BP21_CLIPS / video / scene / "extract_24"
    if not extract_dir.is_dir():
        raise LongSceneError(f"No extracted frames for {video}/{scene} in {extract_dir}")

    all_pngs = sorted(extract_dir.glob("frame_*.png"))
    pngs = all_pngs[start_frame:end_frame]
    if len(pngs) != n_frames:
        raise LongSceneError(
            f"{video}/{scene} found {len(pngs)} frames in [{start_frame}:{end_frame}], expected {n_frames}"
        )

    frames = load_rgb_stack(pngs)
    actual_frames = len(frames)
    height, width = int(frames.shape[1]), int(frames.shape[2])

    # Load tracks & build ObjectRequests
    scene_dir = DATASET / video / "segmentations" / scene
    tracks = list_tracks(scene_dir) if scene_dir.is_dir() else []
    if not tracks and not allow_ineligible:
        raise LongSceneError(f"No track directories found in {scene_dir}")

    frame_ids = list(range(start_frame, start_frame + actual_frames))
    index_of = {fid: idx for idx, fid in enumerate(frame_ids)}
    union_mask = np.zeros((actual_frames, height, width), dtype=bool)
    objects: list[ObjectRequest] = []
    errors: list[float] = []

    for track_dir in tracks:
        pairs = [p for p in pair_track(scene_dir, track_dir) if p.frame_id in index_of]
        if not pairs:
            continue
        stack = np.zeros((actual_frames, height, width), dtype=bool)
        first_crop: np.ndarray | None = None
        first_bbox: tuple[int, int, int, int] | None = None
        for pair in sorted(pairs, key=lambda p: p.frame_id):
            crop = load_rgba(pair.crop_path)
            rows, cols = bbox_slices(pair.bbox, crop.shape[0], crop.shape[1], height, width)
            slot = index_of[pair.frame_id]
            errors.append(opaque_mae(frames[slot], crop, rows, cols))
            stack[slot, rows, cols] |= crop[..., 3] >= 128
            if first_crop is None:
                first_crop = crop[..., :3]
                first_bbox = (int(cols.start), int(rows.start), int(cols.stop), int(rows.stop))
        if first_crop is None or first_bbox is None:
            continue
        union_mask |= stack
        objects.append(
            ObjectRequest(
                object_id=track_dir.name,
                appearance=first_crop,
                bbox=first_bbox,
                mask=stack,
                frame_index=int(min(index_of[p.frame_id] for p in pairs)),
            )
        )

    if not objects and not allow_ineligible:
        raise LongSceneError(f"{video}/{scene}: no tracks overlap frames [{start_frame}:{end_frame}]")
    mae = float(sum(errors) / len(errors)) if errors else 0.0
    if mae > PASTE_MAE_MAX and not allow_ineligible:
        raise LongSceneError(f"{video}/{scene}: paste-back MAE {mae:.3f} > {PASTE_MAE_MAX}")

    route = "pointstream" if is_eligible else "conventional_fallback"

    return LongSceneClip(
        video=video,
        scene=scene,
        context_id=context_id,
        n_frames=actual_frames,
        frames=frames,
        masks=union_mask,
        objects=tuple(objects),
        paste_back_mae=round(mae, 3),
        is_eligible=is_eligible,
        route=route,
        failure_reasons=tuple(fail_reasons),
    )
