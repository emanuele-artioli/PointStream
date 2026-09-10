"""Byte-only independent client reconstruction for PointStream.

BP55 / Gate 1:
A byte-only independent client from payload and metadata to delivered
full-resolution frames, including decode, warp/restoration, foreground
reconstruction and correction. It receives NO source pixels and NO
encoder-side objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from src.runner.accounting import MetadataSubledger
from src.runner.mask_wire import SCHEMA_VERSION, decode_mask, encode_mask, wire_declaration


@dataclass(frozen=True)
class ClientPlacement:
    """One foreground object placement reconstructed on the client.

    Crops arrive as decoded RGB pixels from transmitted appearance bitstreams
    (e.g. JPEG decode), not as encoder-side source references.
    """

    crop: np.ndarray | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    encoded_crop: bytes | None = None
    frame_index: int = 0
    mask: np.ndarray | None = None
    object_id: str = "object"
    is_generated: bool = False
    pose: np.ndarray | None = None
    motion_field: np.ndarray | None = None


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
            if not getattr(p, "is_generated", False) and p.crop is not None
        )

    if generator is not None:
        generated_items: list[Any] = []
        if placements:
            generated_items.extend(p for p in placements if getattr(p, "is_generated", False))
        if not generated_items and objects:
            generated_items = [item for item in objects if getattr(item, "is_generated", False)]
        if generated_items:
            from src.contracts.conditioning import ConditioningBundle
            from src.pipeline.reconstruction.dispatch import dispatch
            from src.pipeline.reconstruction.reconstruct import ObjectRequest, _bundle_for

            bundles = tuple(
                _bundle_for(item)
                if isinstance(item, ObjectRequest)
                else ConditioningBundle(
                    appearance=getattr(item, "crop", None),
                    pose=getattr(item, "pose", None),
                    mask=getattr(item, "mask", None),
                    motion_field=getattr(item, "motion_field", None),
                    bbox=item.bbox,
                    frame_index=item.frame_index,
                    object_id=getattr(item, "object_id", "object"),
                )
                for item in generated_items
            )
            crops, _ = dispatch(
                generator,
                bundles,
                seed=seed,
                params=params,
                policy=active_policy,
            )
            for item, crop in zip(generated_items, crops, strict=True):
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
    references: Mapping[str, bytes] | None = None,
    generator_meta: Mapping[str, Any] | None = None,
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

    ref_meta: dict[str, Any] = {}
    if references:
        for obj_id, ref_bytes in references.items():
            ref_key = f"ref_{obj_id}"
            arrays[ref_key] = np.frombuffer(ref_bytes, dtype=np.uint8)
            ref_meta[str(obj_id)] = {"key": ref_key, "byte_count": len(ref_bytes)}

    placement_meta: list[dict[str, Any]] = []
    for index, placement in enumerate(placements):
        crop_key = None
        encoded_crop_key = None
        mask_key = None
        pose_key = None
        motion_key = None
        mask_wire_meta = None
        if placement.mask is not None:
            mask_key = f"mask_{index}"
            mask_blob = encode_mask(placement.mask)
            arrays[mask_key] = np.frombuffer(mask_blob, dtype=np.uint8).copy()
            mask_wire_meta = {
                **wire_declaration(),
                "shape": [int(dim) for dim in np.asarray(placement.mask).shape],
                "payload_bytes": len(mask_blob),
            }

        if placement.is_generated:
            if placement.pose is not None:
                pose_key = f"pose_{index}"
                arrays[pose_key] = np.asarray(placement.pose, dtype=np.uint8)
            if placement.motion_field is not None:
                motion_key = f"motion_{index}"
                arrays[motion_key] = np.asarray(placement.motion_field, dtype=np.float32)
        else:
            if placement.encoded_crop is not None:
                encoded_crop_key = f"encoded_crop_{index}"
                arrays[encoded_crop_key] = np.frombuffer(placement.encoded_crop, dtype=np.uint8)
            elif placement.crop is not None:
                crop_key = f"crop_{index}"
                arrays[crop_key] = np.asarray(placement.crop, dtype=np.uint8)

        placement_meta.append(
            {
                "crop_key": crop_key,
                "encoded_crop_key": encoded_crop_key,
                "mask_key": mask_key,
                "mask_wire": mask_wire_meta,
                "pose_key": pose_key,
                "motion_key": motion_key,
                "bbox": [int(x) for x in placement.bbox],
                "frame_index": int(placement.frame_index),
                "object_id": str(placement.object_id),
                "is_generated": bool(placement.is_generated),
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
            b_val = residual_payload.get("bitstream") or residual_payload.get("residual_stream")
            bitstream_bytes = bytes(b_val) if b_val is not None else b""
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
        "mask_wire": wire_declaration(),
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "background": background_meta,
        "references": ref_meta,
        "generator": dict(generator_meta) if generator_meta is not None else None,
        "placements": placement_meta,
        "residual": residual_meta,
    }
    arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def account_serialized_request(
    payload: bytes,
    *,
    residual: int = 0,
    panorama: int = 0,
    actor_reference: int = 0,
) -> MetadataSubledger:
    """Split the envelope remainder into named metadata parts.

    Residual, panorama, and actor-reference charges are supplied by the caller
    (the same numbers the ledger already uses) so they are not counted again
    inside the subledger. ``subledger.total`` equals ``len(payload)`` minus
    those three charges.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("client payload must be bytes")
    remainder = max(0, len(payload) - int(residual) - int(panorama) - int(actor_reference))
    mask_payload = 0
    pose_motion = 0
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        for key in arrays.files:
            if key.startswith("mask_"):
                mask_payload += int(np.asarray(arrays[key]).nbytes)
            elif key.startswith("pose_") or key.startswith("motion_"):
                pose_motion += int(np.asarray(arrays[key]).nbytes)
        placements = metadata.get("placements") or []
        placement_headers = len(json.dumps(placements).encode("utf-8"))
        generator = metadata.get("generator")
        generator_metadata = (
            len(json.dumps(generator).encode("utf-8")) if generator is not None else 0
        )
    named = mask_payload + pose_motion + placement_headers + generator_metadata
    envelope_overhead = remainder - named
    if envelope_overhead < 0:
        raise ValueError(
            "metadata subledger exceeds envelope remainder: "
            f"named={named} remainder={remainder}"
        )
    return MetadataSubledger(
        mask_payload=mask_payload,
        pose_motion=pose_motion,
        placement_headers=placement_headers,
        generator_metadata=generator_metadata,
        envelope_overhead=envelope_overhead,
    )


def reconstruct_serialized_client(
    payload: bytes,
    *,
    resolver: BackgroundResolver | None = None,
    return_base: bool = False,
    require_compressed: bool = False,
    generator: Any = None,
    seed: int | None = None,
) -> Clip | tuple[Clip, Clip]:
    """Reconstruct only from the validated NumPy/JSON client envelope."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("client payload must be bytes")
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        if metadata.get("schema") != 1:
            raise ValueError("unsupported client payload schema")
        mask_decl = metadata.get("mask_wire")
        if mask_decl is not None and int(mask_decl.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported mask wire schema")
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

        decoded_references: dict[str, np.ndarray] = {}
        ref_meta = metadata.get("references") or {}
        import cv2
        for obj_id, info in ref_meta.items():
            key = info.get("key")
            if key and key in arrays:
                raw_ref = np.asarray(arrays[key], dtype=np.uint8).tobytes()
                dec = cv2.imdecode(np.frombuffer(raw_ref, dtype=np.uint8), cv2.IMREAD_COLOR)
                if dec is not None:
                    decoded_references[obj_id] = np.asarray(dec, dtype=np.uint8)

        gen_meta = metadata.get("generator")
        active_generator = generator
        active_seed = seed
        active_params = None
        if gen_meta is not None:
            if seed is not None and gen_meta.get("seed") is not None and seed != gen_meta["seed"]:
                raise ValueError(
                    f"Mismatched generation seed: requested {seed}, payload has {gen_meta['seed']}"
                )
            if active_seed is None:
                active_seed = gen_meta.get("seed", 1337)
            if gen_meta.get("params"):
                from src.contracts.conditioning import GenerationParams
                active_params = GenerationParams(**gen_meta["params"])
            if active_generator is not None and getattr(active_generator, "name", None) is not None:
                if (
                    gen_meta.get("name")
                    and active_generator.name != gen_meta["name"]
                    and active_generator.name != "injected"
                ):
                    raise ValueError(
                        f"Mismatched generator model: requested {active_generator.name}, payload has {gen_meta['name']}"
                    )
            if active_generator is None and gen_meta.get("name"):
                from src.components.generation import REGISTRY
                from src.contracts.conditioning import FrameGenerator
                from src.pipeline.reconstruction.dispatch import from_spec

                gen_name = gen_meta["name"]
                if REGISTRY.has(gen_name):
                    spec = REGISTRY.spec(gen_name)
                    backend = REGISTRY.build(gen_name)
                    if isinstance(backend, FrameGenerator):
                        active_generator = from_spec(spec, backend)

        pipeline_placements = []
        to_generate_bundles = []
        to_generate_placements = []

        for item in metadata["placements"]:
            bbox = (int(item["bbox"][0]), int(item["bbox"][1]), int(item["bbox"][2]), int(item["bbox"][3]))
            frame_index = int(item["frame_index"])
            object_id = str(item["object_id"])
            mask = None
            if item.get("mask_key") and item["mask_key"] in arrays:
                mask_blob = np.asarray(arrays[item["mask_key"]], dtype=np.uint8).tobytes()
                decoded_mask = decode_mask(mask_blob)
                declared = (item.get("mask_wire") or {}).get("shape")
                if declared is not None and list(decoded_mask.shape) != [int(dim) for dim in declared]:
                    raise ValueError("mask shape does not match placement metadata")
                mask = decoded_mask.astype(bool, copy=False)

            if item.get("is_generated", False):
                pose = None
                if item.get("pose_key") and item["pose_key"] in arrays:
                    pose = np.asarray(arrays[item["pose_key"]], dtype=np.uint8)
                motion = None
                if item.get("motion_key") and item["motion_key"] in arrays:
                    motion = np.asarray(arrays[item["motion_key"]], dtype=np.float32)

                ref_crop = decoded_references.get(object_id)
                from src.contracts.conditioning import ConditioningBundle
                bundle = ConditioningBundle(
                    appearance=ref_crop,
                    pose=pose,
                    mask=mask,
                    motion_field=motion,
                    bbox=bbox,
                    frame_index=frame_index,
                    object_id=object_id,
                )
                to_generate_bundles.append(bundle)
                to_generate_placements.append(
                    {"bbox": bbox, "frame_index": frame_index, "mask": mask, "object_id": object_id}
                )
            else:
                crop = None
                encoded_crop_key = item.get("encoded_crop_key")
                if encoded_crop_key is not None and encoded_crop_key in arrays:
                    encoded = np.asarray(arrays[encoded_crop_key], dtype=np.uint8)
                    decoded_crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if decoded_crop is None:
                        raise ValueError("JPEG appearance payload did not decode")
                    crop = np.asarray(decoded_crop, dtype=np.uint8)
                elif item.get("crop_key") and item["crop_key"] in arrays:
                    crop = np.asarray(arrays[item["crop_key"]], dtype=np.uint8)

                if crop is not None:
                    pipeline_placements.append(
                        Placement(crop=crop, bbox=bbox, frame_index=frame_index, mask=mask)
                    )

        if to_generate_bundles:
            if active_generator is None:
                raise ValueError("Payload requires generation but no generator backend is available")
            from src.pipeline.reconstruction.dispatch import dispatch
            crops, _ = dispatch(
                active_generator,
                tuple(to_generate_bundles),
                seed=active_seed or 1337,
                params=active_params,
                policy=DevicePolicy(),
            )
            for info, crop in zip(to_generate_placements, crops, strict=True):
                pipeline_placements.append(
                    Placement(
                        crop=crop,
                        bbox=info["bbox"],
                        frame_index=info["frame_index"],
                        mask=info["mask"],
                    )
                )

        active_policy = DevicePolicy()
        active_resolver = resolver or BackgroundResolver()
        if background is None or background.mode == MODE_NONE or background.deferred_to_residual:
            bg_frames = np.zeros(
                (int(metadata["frame_count"]), int(metadata["height"]), int(metadata["width"]), 3),
                dtype=np.uint8,
            )
        else:
            bg_frames, _ = active_resolver.frames_for(
                background,
                frame_count=int(metadata["frame_count"]),
                height=int(metadata["height"]),
                width=int(metadata["width"]),
                policy=active_policy,
            )

        if pipeline_placements:
            base_frames = composite_clip(
                bg_frames,
                tuple(pipeline_placements),
                use_heuristic_mask=True,
            )
        else:
            base_frames = as_clip(bg_frames, path="independent_client_base")

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

        delivered = base_frames.copy()
        if residual_payload is not None and not getattr(residual_payload, "is_absent", True):
            from src.pipeline.residual.signal import apply_residual

            delivered = apply_residual(delivered, residual_payload)

        delivered_clip = as_clip(delivered, path="independent_client_delivered")
        if return_base:
            return delivered_clip, as_clip(base_frames, path="independent_client_base")
        return delivered_clip


__all__ = [
    "ClientPlacement",
    "account_serialized_request",
    "reconstruct_serialized_client",
    "reconstruct_independent_client",
    "serialize_client_request",
]
