"""Opt-in packing of schema-1 client envelopes; physical archive length is rate.

Unpack before invoking the ordinary client. Native .npy members are copied
byte-for-byte. Geometry is retained. Coarse masks sample every scale-th pixel
from the top-left and repeat nearest neighbours, cropping to original shape.
This is a mask intervention, not an accuracy claim. Stale residuals are rejected.
"""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any
import zipfile

import numpy as np

from src.runner.mask_wire import decode_mask, encode_mask, wire_declaration

MANIFEST = "packet_packing.json"
FORMAT = "pointstream.client-packet-packing"
VERSION = 1
NATIVE_PREFIXES = ("residual_bitstream", "background_payload_", "encoded_crop_", "ref_")
SCALES = (1, 2, 4, 8)
MASK_CODECS = ("psm1", "rle")


def _native(name: str) -> bool:
    key = name.removesuffix(".npy")
    return any(key == prefix or key.startswith(prefix) for prefix in NATIVE_PREFIXES)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _npy(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.save(out, np.ascontiguousarray(array), allow_pickle=False)
    return out.getvalue()


def _archive(members: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            compression = zipfile.ZIP_STORED if _native(name) else zipfile.ZIP_DEFLATED
            archive.writestr(info, members[name], compress_type=compression, compresslevel=9)
    return out.getvalue()


def _read(payload: bytes) -> tuple[dict[str, bytes], dict[str, np.ndarray], dict[str, Any]]:
    if not isinstance(payload, bytes):
        raise TypeError("envelope must be bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate archive member")
            if any("/" in name or (not name.endswith(".npy") and name != MANIFEST) for name in names):
                raise ValueError("unsupported envelope members")
            members = {name: archive.read(name) for name in names}
        arrays = {name[:-4]: np.load(io.BytesIO(value), allow_pickle=False)
                  for name, value in members.items() if name.endswith(".npy")}
        metadata_array = arrays["metadata"]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise ValueError("metadata must be a uint8 byte vector")
        metadata = json.loads(metadata_array.tobytes())
    except (zipfile.BadZipFile, KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("unsupported or corrupt schema-1 NPZ envelope") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != 1:
        raise ValueError("only schema-1 client envelopes are supported")
    for key in ("frame_count", "height", "width"):
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            raise ValueError(f"invalid envelope {key}")
    if not isinstance(metadata.get("placements"), list):
        raise ValueError("placements must be a list")
    return members, arrays, metadata


def _masks(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {}
    for placement in metadata["placements"]:
        frame = placement.get("frame_index")
        if type(frame) is not int or not 0 <= frame < metadata["frame_count"]:
            raise ValueError("placement outside registered frame count")
        key = placement.get("mask_key")
        if not key:
            continue
        if key not in arrays or arrays[key].dtype != np.uint8 or arrays[key].ndim != 1:
            raise ValueError("mask must be a persisted PSM1 byte vector")
        mask = decode_mask(arrays[key].tobytes())
        if any(dim < 1 for dim in mask.shape):
            raise ValueError("mask dimensions must be positive")
        if mask.ndim == 3 and mask.shape[0] != metadata["frame_count"]:
            raise ValueError("mask stack frame count differs from envelope")
        declaration = placement.get("mask_wire") or {}
        if declaration.get("shape") is not None and list(mask.shape) != declaration["shape"]:
            raise ValueError("mask shape differs from placement declaration")
        masks[key] = mask
    if {key for key in arrays if key.startswith("mask_")} != set(masks):
        raise ValueError("unreferenced or unsupported mask arrays")
    return masks


def _codec_encode(mask: np.ndarray, codec: str) -> bytes:
    if codec == "psm1":
        return encode_mask(mask)
    # Use the installed PSR1 wire primitives, byte-compatible with E06.
    from src.runner.mask_rle import MODE_INTRA, encode_mask_stack
    stack = mask[None] if mask.ndim == 2 else mask
    if any(dim > 65535 for dim in stack.shape):
        raise ValueError("E06 RLE dimensions exceed uint16")
    return encode_mask_stack(stack, mode=MODE_INTRA)


def _codec_decode(blob: bytes, codec: str, ndim: int) -> np.ndarray:
    if codec == "psm1":
        return decode_mask(blob)
    from src.runner.mask_rle import decode_mask_stack
    stack = decode_mask_stack(blob)
    if ndim == 2:
        if stack.shape[0] != 1:
            raise ValueError("2-D RLE mask must contain one frame")
        return stack[0]
    return stack


def _expand(coarse: np.ndarray, original_shape: tuple[int, ...], scale: int) -> np.ndarray:
    expected = (*original_shape[:-2], (original_shape[-2] + scale - 1) // scale,
                (original_shape[-1] + scale - 1) // scale)
    if coarse.shape != expected:
        raise ValueError("coarse mask shape inconsistent with original dimensions")
    return np.repeat(np.repeat(coarse, scale, axis=-2), scale, axis=-1)[
        ..., :original_shape[-2], :original_shape[-1]]


def pack_client_envelope(payload: bytes, *, mask_scale: int = 1, mask_codec: str = "psm1") -> bytes:
    """Return a deterministic charged archive; scale 1 preserves mask pixels.

    Scales 2/4/8 may alter the predictor. Any residual-bearing input is rejected
    for these scales; correction must be regenerated against the changed client.
    """
    if type(mask_scale) is not int or mask_scale not in SCALES or mask_codec not in MASK_CODECS:
        raise ValueError("unsupported mask scale or codec")
    members, arrays, metadata = _read(payload)
    if MANIFEST in members:
        raise ValueError("unpack an adapter archive before repacking")
    if mask_scale > 1 and ((metadata.get("residual") or {}).get("present") or
                           any(key.startswith("residual_") for key in arrays)):
        raise ValueError("lossy masks require recomputed correction; stale residual rejected")
    masks = _masks(metadata, arrays)
    descriptors = {}
    changed = 0
    for key, mask in masks.items():
        coarse = mask[..., ::mask_scale, ::mask_scale]
        reconstructed = _expand(coarse, mask.shape, mask_scale)
        changed += int(np.count_nonzero(mask != reconstructed))
        blob = _codec_encode(coarse, mask_codec)
        members[key + ".npy"] = _npy(np.frombuffer(blob, dtype=np.uint8))
        descriptors[key] = {"original_shape": list(mask.shape), "coded_shape": list(coarse.shape)}
    members[MANIFEST] = _json({"format": FORMAT, "version": VERSION, "schema": 1,
        "mask_scale": mask_scale, "mask_codec": mask_codec, "lossy_masks": mask_scale > 1,
        "predictor_pixels_may_change": mask_scale > 1, "changed_mask_pixels": changed,
        "frame_count": metadata["frame_count"], "height": metadata["height"], "width": metadata["width"],
        "original_payload_sha256": hashlib.sha256(payload).hexdigest(), "masks": descriptors})
    return _archive(members)


def unpack_client_envelope(payload: bytes) -> bytes:
    """Restore a normal schema-1 NPZ for the existing independent client.

    A validated ordinary NPZ is already supported and returns unchanged. Other
    archive formats, including E06 floor containers, are explicitly rejected.
    Expanded envelope size is a decoder intermediate, never the charged rate.
    """
    members, arrays, metadata = _read(payload)
    if MANIFEST not in members:
        _masks(metadata, arrays)
        return payload
    spec = json.loads(members.pop(MANIFEST))
    scale, codec = spec.get("mask_scale"), spec.get("mask_codec")
    if (spec.get("format") != FORMAT or spec.get("version") != VERSION or spec.get("schema") != 1 or
            type(scale) is not int or scale not in SCALES or codec not in MASK_CODECS or
            spec.get("lossy_masks") is not (scale > 1)):
        raise ValueError("unsupported packet packing declaration")
    if any(spec.get(key) != metadata[key] for key in ("frame_count", "height", "width")):
        raise ValueError("packed envelope dimensions differ from original metadata")
    if scale > 1 and ((metadata.get("residual") or {}).get("present") or
                      any(key.startswith("residual_") for key in arrays)):
        raise ValueError("lossy masks carry stale correction")
    expected_keys = {p["mask_key"] for p in metadata["placements"] if p.get("mask_key")}
    if set(spec.get("masks", {})) != expected_keys:
        raise ValueError("packed mask descriptors do not cover placements")
    for key, descriptor in spec["masks"].items():
        shape = tuple(descriptor["original_shape"])
        if len(shape) not in (2, 3) or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("invalid original mask dimensions")
        if len(shape) == 3 and shape[0] != metadata["frame_count"]:
            raise ValueError("packed mask stack frame count mismatch")
        mask = _codec_decode(arrays[key].tobytes(), codec, len(shape))
        if list(mask.shape) != descriptor["coded_shape"]:
            raise ValueError("coded mask dimensions mismatch")
        blob = encode_mask(_expand(mask, shape, scale))
        members[key + ".npy"] = _npy(np.frombuffer(blob, dtype=np.uint8))
        for placement in metadata["placements"]:
            if placement.get("mask_key") == key:
                original_declared = (placement.get("mask_wire") or {}).get("shape")
                if original_declared is not None and original_declared != list(shape):
                    raise ValueError("original mask dimensions differ from placement")
                placement["mask_wire"] = {**wire_declaration(), "shape": list(shape), "payload_bytes": len(blob)}
    members["metadata.npy"] = _npy(np.frombuffer(_json(metadata), dtype=np.uint8))
    restored = _archive(members)
    _, restored_arrays, restored_metadata = _read(restored)
    _masks(restored_metadata, restored_arrays)
    return restored


def packing_info(payload: bytes) -> dict[str, Any]:
    """Inventory actual charged ZIP members, including all framing bytes."""
    members, _, metadata = _read(payload)
    spec = json.loads(members[MANIFEST]) if MANIFEST in members else None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = [{"name": item.filename, "uncompressed_member_bytes": item.file_size,
                    "physical_member_bytes": item.compress_size, "native_stored": _native(item.filename)}
                   for item in archive.infolist()]
    stored = sum(item["physical_member_bytes"] for item in entries)
    return {"complete_file_bytes": len(payload), "physical_member_bytes": stored,
            "zip_framing_bytes": len(payload) - stored, "frame_count": metadata["frame_count"],
            "original_dimensions": [metadata["height"], metadata["width"]], "packing": spec,
            "entries": entries, "rate_boundary": "charged archive length; unpacked intermediates excluded"}
