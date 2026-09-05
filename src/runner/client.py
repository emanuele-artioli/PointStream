"""Byte-only independent client reconstruction for PointStream.

BP55 / Gate 1:
A byte-only independent client from payload and metadata to delivered
full-resolution frames, including decode, warp/restoration, foreground
reconstruction and correction. It receives NO source pixels and NO
encoder-side objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.pipeline.reconstruction.background import (
    BackgroundModelView,
    BackgroundResolver,
    MODE_NONE,
)
from src.pipeline.reconstruction.clips import Clip, as_clip
from src.pipeline.reconstruction.compositor import Placement, composite_clip
from src.pipeline.reconstruction.device import DevicePolicy


@dataclass(frozen=True)
class ClientPlacement:
    """One foreground object placement reconstructed on the client.

    Crops arrive as decoded RGB pixels from transmitted appearance bitstreams
    (e.g. JPEG decode), not as encoder-side source references.
    """

    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    frame_index: int = 0
    mask: np.ndarray | None = None
    object_id: str = "object"


def reconstruct_independent_client(
    *,
    background: BackgroundModelView | None,
    frame_count: int,
    height: int,
    width: int,
    placements: Sequence[ClientPlacement] | None = None,
    residual_payload: Any = None,
    resolver: BackgroundResolver | None = None,
    policy: DevicePolicy | None = None,
) -> Clip:
    """Reconstruct full-resolution frames from transmitted payloads only.

    Args:
        background: Decoded background model view (plate + homographies)
            or None.
        frame_count: Number of frames in this chunk.
        height: Target frame height.
        width: Target frame width.
        placements: Decoded foreground object placements (if any).
        residual_payload: Transmitted residual payload (if residual is on).
        resolver: Optional background resolver instance.
        policy: Optional device policy.

    Returns:
        Delivered RGB frames of shape (frame_count, height, width, 3).
    """
    active_policy = policy or DevicePolicy()
    active_resolver = resolver or BackgroundResolver()

    # 1. Background reconstruction / warp
    if background is None or background.mode == MODE_NONE or background.deferred_to_residual:
        bg_frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    else:
        bg_frames, _ = active_resolver.frames_for(
            background,
            frame_count=frame_count,
            height=height,
            width=width,
            policy=active_policy,
        )

    # 2. Foreground placement / compositing
    if placements:
        pipeline_placements = [
            Placement(
                crop=p.crop,
                bbox=p.bbox,
                frame_index=p.frame_index,
                mask=p.mask,
            )
            for p in placements
        ]
        frames = composite_clip(
            bg_frames,
            tuple(pipeline_placements),
            use_heuristic_mask=True,
        )
    else:
        frames = bg_frames

    # 3. Residual application (if transmitted)
    if residual_payload is not None and not getattr(residual_payload, "is_absent", True):
        from src.pipeline.residual.signal import apply_residual

        frames = apply_residual(frames, residual_payload)

    return as_clip(frames, path="independent_client_delivered")


__all__ = [
    "ClientPlacement",
    "reconstruct_independent_client",
]
