"""E1 sequence loader: BP46 long scenes, not the BP21 8-frame windows.

The long-scene package lives in ``experiments.long_scenes`` (D1). This module
is the BP45 adapter so the sweep does not go through the short cached windows.
"""

from __future__ import annotations

from typing import Any

#: BP46 diagnostic partition: near-static vs smooth-pan. scene_003 of
#: federer_djokovic is 85 frames, so only the 48-frame span is valid there.
DIAGNOSTIC_SCENES: dict[str, tuple[str, ...]] = {
    "alcaraz_highlights": ("scene_000", "scene_028"),
    "federer_djokovic": ("scene_001", "scene_003"),
}

DEFAULT_VIDEO = "alcaraz_highlights"
DEFAULT_SCENES: tuple[str, ...] = DIAGNOSTIC_SCENES[DEFAULT_VIDEO]
DEFAULT_SPAN_FRAMES = 48


def _load_bp46(video: str, scene: str, n_frames: int) -> Any:
    try:
        from experiments.long_scenes.loader import load_long_scene_clip
    except ImportError as exc:
        raise SystemExit(
            "E1 needs the BP46 long-scene loader "
            "(experiments.long_scenes.loader.load_long_scene_clip). "
            "Merge D1 before running the sweep."
        ) from exc
    return load_long_scene_clip(video, scene, n_frames)


def load_e1_sequence(video: str, scenes: list[str], *, n_frames: int) -> list[Any]:
    """Load each named scene at exactly ``n_frames``. Refuses a silent skip."""
    if n_frames < DEFAULT_SPAN_FRAMES:
        raise SystemExit(
            f"E1 scenes are at least {DEFAULT_SPAN_FRAMES} frames (2 s at 24 fps). "
            f"Got n_frames={n_frames}. The BP21 8-frame windows are not this search."
        )
    clips: list[Any] = []
    for scene in scenes:
        clip = _load_bp46(video, scene, n_frames)
        frames = getattr(clip, "frames", None)
        if frames is None:
            raise SystemExit(f"{video}/{scene}: long-scene clip has no frames")
        if int(frames.shape[0]) != n_frames:
            raise SystemExit(
                f"{video}/{scene}: loaded {frames.shape[0]} frames, expected {n_frames}"
            )
        clips.append(clip)
    if len(clips) != len(scenes):
        raise SystemExit(
            f"asked for {len(scenes)} scenes, loaded {len(clips)}. "
            "A missing scene must not silently shrink the sequence."
        )
    return clips


__all__ = [
    "DEFAULT_SCENES",
    "DEFAULT_SPAN_FRAMES",
    "DEFAULT_VIDEO",
    "DIAGNOSTIC_SCENES",
    "load_e1_sequence",
]
