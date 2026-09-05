"""Byte-only independent client reconstruction for PointStream.

BP55 / Gate 1:
A byte-only independent client from payload and metadata to delivered
full-resolution frames, including decode, warp/restoration, foreground
reconstruction and correction. It receives NO source pixels and NO
encoder-side objects.
"""

from __future__ import annotations

from collections.abc import Sequence
import io
import json
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


def serialize_client_request(
    *,
    background: BackgroundModelView | None,
    frame_count: int,
    height: int,
    width: int,
    placements: Sequence[ClientPlacement] = (),
    residual_payload: Any = None,
) -> bytes:
    """Serialize explicit client inputs without executable object payloads."""
    if residual_payload is not None:
        raise ValueError("serialized residual client payload is not implemented")
    arrays: dict[str, np.ndarray] = {}
    background_meta: dict[str, Any] | None = None
    if background is not None:
        plate_key = None
        if background.plate is not None:
            plate_key = "background_plate"
            arrays[plate_key] = np.asarray(background.plate, dtype=np.uint8)
        background_meta = {
            "plate_key": plate_key,
            "homographies": background.homographies,
            "mode": background.mode,
            "deferred_to_residual": background.deferred_to_residual,
            "scene_id": background.scene_id,
            "width": background.width,
            "height": background.height,
            "payload_bytes": background.payload_bytes,
            "geometry_header": background.geometry_header.hex(),
            "geometry_header_bytes": background.geometry_header_bytes,
        }
    placement_meta: list[dict[str, Any]] = []
    for index, placement in enumerate(placements):
        crop_key = f"crop_{index}"
        arrays[crop_key] = np.asarray(placement.crop, dtype=np.uint8)
        mask_key = None
        if placement.mask is not None:
            mask_key = f"mask_{index}"
            arrays[mask_key] = np.asarray(placement.mask, dtype=np.uint8)
        placement_meta.append(
            {
                "crop_key": crop_key,
                "mask_key": mask_key,
                "bbox": placement.bbox,
                "frame_index": placement.frame_index,
                "object_id": placement.object_id,
            }
        )
    metadata = {
        "schema": 1,
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "background": background_meta,
        "placements": placement_meta,
    }
    arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def reconstruct_serialized_client(
    payload: bytes, *, resolver: BackgroundResolver | None = None
) -> Clip:
    """Reconstruct only from the validated NumPy/JSON client envelope."""
    if not isinstance(payload, bytes):
        raise TypeError("client payload must be bytes")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        if metadata.get("schema") != 1:
            raise ValueError("unsupported client payload schema")
        bg_meta = metadata.get("background")
        background = None
        if bg_meta is not None:
            plate = None
            if bg_meta["plate_key"] is not None:
                plate = np.asarray(arrays[bg_meta["plate_key"]], dtype=np.uint8)
            background = BackgroundModelView(
                plate=plate,
                homographies=tuple(tuple(row) for row in bg_meta["homographies"]),
                mode=bg_meta["mode"],
                deferred_to_residual=bool(bg_meta["deferred_to_residual"]),
                scene_id=bg_meta["scene_id"],
                width=int(bg_meta["width"]),
                height=int(bg_meta["height"]),
                payload_bytes=bg_meta["payload_bytes"],
                geometry_header=bytes.fromhex(bg_meta["geometry_header"]),
                geometry_header_bytes=int(bg_meta["geometry_header_bytes"]),
            )
        placements = []
        for item in metadata["placements"]:
            bbox = item["bbox"]
            if len(bbox) != 4:
                raise ValueError("client placement bbox must have four coordinates")
            mask = None
            if item["mask_key"] is not None:
                mask = np.asarray(arrays[item["mask_key"]], dtype=np.uint8).astype(bool)
            placements.append(
                ClientPlacement(
                    crop=np.asarray(arrays[item["crop_key"]], dtype=np.uint8),
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    frame_index=int(item["frame_index"]),
                    mask=mask,
                    object_id=str(item["object_id"]),
                )
            )
    return reconstruct_independent_client(
        background=background,
        frame_count=int(metadata["frame_count"]),
        height=int(metadata["height"]),
        width=int(metadata["width"]),
        placements=tuple(placements),
        resolver=resolver,
    )


__all__ = [
    "ClientPlacement",
    "reconstruct_serialized_client",
    "reconstruct_independent_client",
    "serialize_client_request",
]
