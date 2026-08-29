"""One real clip for a tier run, with its player masks, re-verified before use.

The frames are BP21's cached 4K windows (`outputs/bp21-headroom/clips/...`),
which is the same pixel data BP20/BP21 measured on. Reusing them avoids a second
ffmpeg extraction and, more importantly, keeps this stream measuring the clip
the headroom argument was measured on rather than a lookalike.

Two things this module refuses to do, both because they have gone wrong here
before:

* **It does not rebuild a filename from a frame id.** Crops are paired with
  metadata rows by position in file order; the id comes off the row.
* **It does not trust the frame convention.** This dataset carries two, and 44%
  of tracks are offset. The crops are pasted back onto the loaded frames and the
  mean absolute error over the opaque pixels must come in under
  `PASTE_MAE_MAX`, or the clip is refused. A clip that fails paste-back is
  measuring the wrong region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.headroom.real import (
    DATASET,
    PASTE_MAE_MAX,
    TrackPair,
    bbox_slices,
    list_tracks,
    load_rgb_stack,
    load_rgba,
    opaque_mae,
    pair_track,
)
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.contracts import paths as ps_paths


#: BP21 wrote its 4K windows here, 48 frames per clip, named by frame id.
BP21_CLIPS = ps_paths.outputs() / "bp21-headroom" / "clips"

#: The default clip. `outputs/bp21-headroom/.../paste_back/diagnosis.json`
#: records `extract_24_frame_id` winning at MAE 0.0 on this scene, against 28.4
#: and 29.1 for the two rival conventions — an unusually clean separation, which
#: is why it is the one picked to prove a path on.
DEFAULT_VIDEO = "alcaraz_highlights"
DEFAULT_SCENE = "scene_000"


class ClipUnusable(RuntimeError):
    """The clip cannot be loaded, or its crops do not land on its pixels."""


@dataclass(frozen=True)
class TierClip:
    """Source frames, per-track player masks, and the objects the runner takes."""

    video: str
    scene: str
    frame_ids: tuple[int, ...]
    frames: np.ndarray
    """`(T, H, W, 3)` uint8 RGB."""
    objects: tuple[ObjectRequest, ...]
    union_mask: np.ndarray
    """`(T, H, W)` bool: every tracked object, for reporting player coverage."""
    paste_back_mae: float
    n_tracks: int

    def describe(self) -> dict[str, Any]:
        frames, height, width, _ = self.frames.shape
        return {
            "video": self.video,
            "scene": self.scene,
            "frame_ids": list(self.frame_ids),
            "n_frames": int(frames),
            "resolution": f"{width}x{height}",
            "source_bytes": int(self.frames.nbytes),
            "n_tracks": self.n_tracks,
            "player_pixel_fraction": float(self.union_mask.mean()),
            "paste_back_mae": self.paste_back_mae,
            "paste_back_threshold": PASTE_MAE_MAX,
        }


def load_tier_clip(
    *,
    video: str = DEFAULT_VIDEO,
    scene: str = DEFAULT_SCENE,
    n_frames: int = 8,
) -> TierClip:
    """The first `n_frames` of BP21's cached window for `video`/`scene`.

    Raises:
        ClipUnusable: If the cached window or the dataset scene is missing, if
            no track survives pairing, or if paste-back exceeds `PASTE_MAE_MAX`.
    """
    window = BP21_CLIPS / video / scene / "window"
    if not window.is_dir():
        raise ClipUnusable(f"no cached BP21 window at {window}")
    pngs = sorted(window.glob("frame_*.png"))
    if len(pngs) < n_frames:
        raise ClipUnusable(f"{window} holds {len(pngs)} frames; {n_frames} were asked for")
    chosen = pngs[:n_frames]
    frame_ids = tuple(int(path.stem.split("_")[-1]) for path in chosen)
    frames = load_rgb_stack(list(chosen))
    height, width = int(frames.shape[1]), int(frames.shape[2])

    scene_dir = DATASET / video / "segmentations" / scene
    if not scene_dir.is_dir():
        raise ClipUnusable(f"no dataset scene at {scene_dir}")
    tracks = list_tracks(scene_dir)
    if not tracks:
        raise ClipUnusable(f"{scene_dir} has no track directories")

    index_of = {frame_id: position for position, frame_id in enumerate(frame_ids)}
    errors: list[float] = []
    objects: list[ObjectRequest] = []
    union = np.zeros((len(frame_ids), height, width), dtype=bool)

    for track_dir in tracks:
        pairs = [
            pair for pair in pair_track(scene_dir, track_dir) if pair.frame_id in index_of
        ]
        if not pairs:
            continue
        stack = np.zeros((len(frame_ids), height, width), dtype=bool)
        first_crop: np.ndarray | None = None
        first_bbox: tuple[int, int, int, int] | None = None
        for pair in sorted(pairs, key=lambda item: item.frame_id):
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
        union |= stack
        objects.append(
            ObjectRequest(
                object_id=track_dir.name,
                appearance=first_crop,
                bbox=first_bbox,
                mask=stack,
                frame_index=int(min(index_of[pair.frame_id] for pair in pairs)),
            )
        )

    if not objects:
        raise ClipUnusable(
            f"{video}/{scene}: no track overlaps frames {frame_ids[0]}..{frame_ids[-1]}"
        )
    if not errors:
        raise ClipUnusable(f"{video}/{scene}: paste-back produced no samples to check")
    mae = float(sum(errors) / len(errors))
    if mae > PASTE_MAE_MAX:
        raise ClipUnusable(
            f"{video}/{scene}: paste-back MAE {mae:.3f} exceeds {PASTE_MAE_MAX}. "
            "The crops do not land on these pixels, so any region score would be "
            "measuring the wrong region."
        )
    return TierClip(
        video=video,
        scene=scene,
        frame_ids=frame_ids,
        frames=frames,
        objects=tuple(objects),
        union_mask=union,
        paste_back_mae=mae,
        n_tracks=len(objects),
    )


__all__ = [
    "BP21_CLIPS",
    "ClipUnusable",
    "TierClip",
    "TrackPair",
    "load_tier_clip",
]
