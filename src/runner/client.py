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
    encoded_crop: bytes | None = None
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
    return_base: bool = False,
    generator: Any = None,
    objects: Sequence[Any] | None = None,
    params: Any = None,
    seed: int = 1337,
) -> Clip | tuple[Clip, Clip]:
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
        return_base: If True, return (delivered_clip, base_clip).
        generator: Optional client generator backend.
        objects: Optional objects to generate when generation is enabled.
        params: Optional generation params.
        seed: Random seed for deterministic generation.

    Returns:
        Delivered RGB frames of shape (frame_count, height, width, 3), or a
        pair (delivered_clip, base_clip) when return_base is True.
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
    pipeline_placements: list[Placement] = []
    if placements:
        pipeline_placements.extend(
            Placement(
                crop=p.crop,
                bbox=p.bbox,
                frame_index=p.frame_index,
                mask=p.mask,
            )
            for p in placements
        )

    if generator is not None and objects:
        to_generate = [item for item in objects if getattr(item, "supplied_crop", None) is None]
        if to_generate:
            from src.pipeline.reconstruction.dispatch import dispatch
            from src.pipeline.reconstruction.reconstruct import _bundle_for

            bundles = tuple(_bundle_for(item) for item in to_generate)
            crops, _ = dispatch(
                generator,
                bundles,
                seed=seed,
                params=params,
                policy=active_policy,
            )
            for item, crop in zip(to_generate, crops, strict=True):
                pipeline_placements.append(
                    Placement(
                        crop=crop,
                        bbox=item.bbox,
                        frame_index=item.frame_index,
                        mask=item.mask,
                    )
                )

    if pipeline_placements:
        base_frames = composite_clip(
            bg_frames,
            tuple(pipeline_placements),
            use_heuristic_mask=True,
        )
    else:
        base_frames = as_clip(bg_frames, path="independent_client_base")

    # 3. Residual application (if transmitted)
    delivered = base_frames.copy()
    if residual_payload is not None and not getattr(residual_payload, "is_absent", True):
        from src.pipeline.residual.signal import apply_residual

        delivered = apply_residual(delivered, residual_payload)

    delivered_clip = as_clip(delivered, path="independent_client_delivered")
    if return_base:
        return delivered_clip, as_clip(base_frames, path="independent_client_base")
    return delivered_clip


def serialize_client_request(
    *,
    background: BackgroundModelView | None,
    frame_count: int,
    height: int,
    width: int,
    placements: Sequence[ClientPlacement] = (),
    residual_payload: Any = None,
    require_compressed: bool = False,
) -> bytes:
    """Serialize explicit client inputs without executable object payloads."""
    arrays: dict[str, np.ndarray] = {}
    background_meta: dict[str, Any] | None = None
    if background is not None:
        plate_key = None
        if background.plate is not None and background.wire_codec is None:
            plate_key = "background_plate"
            arrays[plate_key] = np.asarray(background.plate, dtype=np.uint8)
        wire_payload_keys = []
        wire_header_keys = []
        for packet_index, packet in enumerate(background.wire_payloads):
            payload_key = f"background_payload_{packet_index}"
            header_key = f"background_header_{packet_index}"
            arrays[payload_key] = np.frombuffer(packet, dtype=np.uint8)
            arrays[header_key] = np.frombuffer(
                background.wire_geometry_headers[packet_index], dtype=np.uint8
            )
            wire_payload_keys.append(payload_key)
            wire_header_keys.append(header_key)
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
            "wire_payload_keys": wire_payload_keys,
            "wire_header_keys": wire_header_keys,
            "wire_codec": background.wire_codec,
            "wire_codec_id": background.wire_codec_id,
        }

    placement_meta: list[dict[str, Any]] = []
    for index, placement in enumerate(placements):
        crop_key = f"crop_{index}"
        encoded_crop_key = None
        if placement.encoded_crop is not None:
            encoded_crop_key = f"encoded_crop_{index}"
            arrays[encoded_crop_key] = np.frombuffer(placement.encoded_crop, dtype=np.uint8)
        if placement.encoded_crop is None:
            arrays[crop_key] = np.asarray(placement.crop, dtype=np.uint8)
        mask_key = None
        if placement.mask is not None:
            mask_key = f"mask_{index}"
            arrays[mask_key] = np.asarray(placement.mask, dtype=np.uint8)
        placement_meta.append(
            {
                "crop_key": crop_key,
                "encoded_crop_key": encoded_crop_key,
                "mask_key": mask_key,
                "bbox": placement.bbox,
                "frame_index": placement.frame_index,
                "object_id": placement.object_id,
            }
        )

    # Residual payload serialization
    residual_meta: dict[str, Any] = {"present": False, "is_coded": False, "byte_count": 0}
    if residual_payload is not None and not getattr(residual_payload, "is_absent", False):
        if hasattr(residual_payload, "bitstream") and getattr(residual_payload, "is_coded", False):
            bitstream_bytes = residual_payload.bitstream
            arrays["residual_bitstream"] = np.frombuffer(bitstream_bytes, dtype=np.uint8)
            residual_meta = {
                "present": True,
                "is_coded": True,
                "codec_name": str(residual_payload.codec_name),
                "mode": str(residual_payload.mode),
                "shape": list(residual_payload.shape),
                "pix_fmt": str(residual_payload.pix_fmt),
                "scale": float(residual_payload.scale),
                "offset": float(residual_payload.offset),
                "fps": float(residual_payload.fps),
                "byte_count": len(bitstream_bytes),
                "bitstream_key": "residual_bitstream",
            }
        elif isinstance(residual_payload, dict) and (
            "bitstream" in residual_payload or "residual_stream" in residual_payload
        ):
            bitstream_bytes = bytes(
                residual_payload.get("bitstream") or residual_payload.get("residual_stream")
            )
            arrays["residual_bitstream"] = np.frombuffer(bitstream_bytes, dtype=np.uint8)
            residual_meta = {
                "present": True,
                "is_coded": True,
                "codec_name": str(residual_payload.get("codec_name", "avc")),
                "mode": str(residual_payload.get("mode", "clipped")),
                "shape": list(residual_payload.get("shape", (frame_count, height, width, 3))),
                "pix_fmt": str(residual_payload.get("pix_fmt", "yuv420p")),
                "scale": float(residual_payload.get("scale", 1.0)),
                "offset": float(residual_payload.get("offset", 128.0)),
                "fps": float(residual_payload.get("fps", 25.0)),
                "byte_count": len(bitstream_bytes),
                "bitstream_key": "residual_bitstream",
            }
        else:
            # Unencoded fallback array
            if require_compressed:
                raise ValueError(
                    "Cannot serialize unencoded fallback residual array as compressed evidence"
                )
            raw = getattr(residual_payload, "raw_frames", None)
            if raw is None:
                raw = getattr(residual_payload, "frames", None)
            if raw is None and isinstance(residual_payload, dict):
                raw = residual_payload.get("frames")
            if raw is None and isinstance(residual_payload, np.ndarray):
                raw = residual_payload
            if raw is None:
                raise ValueError("Residual payload has no bitstream and no raw frames")
            raw_arr = np.asarray(raw)
            arrays["residual_raw"] = raw_arr
            residual_meta = {
                "present": True,
                "is_coded": False,
                "codec_name": "raw",
                "mode": str(getattr(residual_payload, "mode", "clipped")),
                "shape": list(raw_arr.shape),
                "scale": float(getattr(residual_payload, "scale", 1.0)),
                "offset": float(getattr(residual_payload, "offset", 128.0)),
                "byte_count": int(raw_arr.nbytes),
                "raw_key": "residual_raw",
            }

    metadata = {
        "schema": 1,
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "background": background_meta,
        "placements": placement_meta,
        "residual": residual_meta,
    }
    arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def reconstruct_serialized_client(
    payload: bytes,
    *,
    resolver: BackgroundResolver | None = None,
    return_base: bool = False,
    require_compressed: bool = False,
) -> Clip | tuple[Clip, Clip]:
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
            wire_codec = bg_meta.get("wire_codec")
            if wire_codec is not None:
                from src.components.background.scale import (
                    TransmittedBackground,
                    decode_transmitted_stream,
                )

                packets = tuple(
                    TransmittedBackground(
                        payload=np.asarray(arrays[payload_key], dtype=np.uint8).tobytes(),
                        geometry_header=np.asarray(arrays[header_key], dtype=np.uint8).tobytes(),
                    )
                    for payload_key, header_key in zip(
                        bg_meta["wire_payload_keys"],
                        bg_meta["wire_header_keys"],
                        strict=True,
                    )
                )
                plate = decode_transmitted_stream(str(wire_codec), packets)
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
                wire_payloads=(),
                wire_geometry_headers=(),
                wire_codec=str(wire_codec) if wire_codec is not None else None,
                wire_codec_id=bg_meta.get("wire_codec_id"),
            )
        placements = []
        for item in metadata["placements"]:
            encoded_crop_key = item.get("encoded_crop_key")
            if encoded_crop_key is not None:
                import cv2

                encoded = np.asarray(arrays[encoded_crop_key], dtype=np.uint8)
                decoded_crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded_crop is None:
                    raise ValueError("JPEG appearance payload did not decode")
                crop = np.asarray(decoded_crop, dtype=np.uint8)
            else:
                crop = np.asarray(arrays[item["crop_key"]], dtype=np.uint8)
            bbox = item["bbox"]
            if len(bbox) != 4:
                raise ValueError("client placement bbox must have four coordinates")
            mask = None
            if item["mask_key"] is not None:
                mask = np.asarray(arrays[item["mask_key"]], dtype=np.uint8).astype(bool)
            placements.append(
                ClientPlacement(
                    crop=crop,
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    frame_index=int(item["frame_index"]),
                    mask=mask,
                    object_id=str(item["object_id"]),
                )
            )

        # Residual deserialization
        res_meta = metadata.get("residual")
        residual_payload = None
        if res_meta and res_meta.get("present"):
            if require_compressed and not res_meta.get("is_coded"):
                raise ValueError(
                    "Rejecting unencoded fallback residual array as compressed evidence"
                )
            from src.pipeline.residual.codec import TransmittedResidual

            if res_meta.get("is_coded"):
                bitstream_key = res_meta.get("bitstream_key", "residual_bitstream")
                if bitstream_key not in arrays:
                    raise ValueError(f"Missing residual bitstream key {bitstream_key!r} in payload")
                bitstream_bytes = np.asarray(arrays[bitstream_key], dtype=np.uint8).tobytes()
                residual_payload = TransmittedResidual(
                    bitstream=bitstream_bytes,
                    codec_name=res_meta["codec_name"],
                    mode=res_meta.get("mode", "clipped"),
                    shape=tuple(res_meta["shape"]),
                    pix_fmt=res_meta.get("pix_fmt", "yuv420p"),
                    scale=float(res_meta.get("scale", 1.0)),
                    offset=float(res_meta.get("offset", 128.0)),
                    fps=float(res_meta.get("fps", 25.0)),
                    is_coded=True,
                )
            else:
                raw_key = res_meta.get("raw_key", "residual_raw")
                if raw_key not in arrays:
                    raise ValueError(f"Missing residual raw key {raw_key!r} in payload")
                raw_frames = np.asarray(arrays[raw_key])
                residual_payload = TransmittedResidual(
                    bitstream=b"",
                    codec_name="raw",
                    mode=res_meta.get("mode", "clipped"),
                    shape=tuple(res_meta["shape"]),
                    scale=float(res_meta.get("scale", 1.0)),
                    offset=float(res_meta.get("offset", 128.0)),
                    is_coded=False,
                    raw_frames=raw_frames,
                )

    return reconstruct_independent_client(
        background=background,
        frame_count=int(metadata["frame_count"]),
        height=int(metadata["height"]),
        width=int(metadata["width"]),
        placements=tuple(placements),
        residual_payload=residual_payload,
        resolver=resolver,
        return_base=return_base,
    )


__all__ = [
    "ClientPlacement",
    "reconstruct_serialized_client",
    "reconstruct_independent_client",
    "serialize_client_request",
]
