"""Lossless E06 floor packs: mask RLE, keyed XOR-delta RLE, thin placement.

Physical T is the packed file length. Native B/F/R members stay stored.
Unpack restores a reconstructable schema-1 envelope; it is not a second charge.
"""

from __future__ import annotations

import io
import json
import struct
import time
from typing import Any, Literal
import zipfile

import numpy as np

from experiments.tier.e06_pack import NATIVE_PREFIXES, _is_native
from src.runner.mask_wire import decode_mask, encode_mask, wire_declaration

MANIFEST_NAME = "floor_manifest.json"
MASKS_RLE_NAME = "masks.rle"
PLACEMENT_I16_NAME = "placement_i16.bin"
HOMOGRAPHY_NAME = "homographies_f64.bin"
THIN_META_NAME = "thin_meta.json"
RLE_MAGIC = b"PSR1"
RLE_VERSION = 1
MODE_INTRA = 0
MODE_XOR_KEY = 1
KEY_PERIOD = 4
MaskCodec = Literal["rle_intra", "rle_xor_key4", "parent_packed_bits"]
PlacementCodec = Literal["parent_json", "delta_int16_bbox_float32_H"]

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")


def load_npz_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def is_floor_pack(payload: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return MANIFEST_NAME in archive.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError("payload is not a zip envelope") from exc


def _runs(flat: np.ndarray) -> list[tuple[int, int]]:
    values = np.ascontiguousarray(flat, dtype=np.uint8).reshape(-1)
    out: list[tuple[int, int]] = []
    index = 0
    n = int(values.size)
    while index < n:
        value = int(values[index])
        stop = index + 1
        while stop < n and int(values[stop]) == value:
            stop += 1
        length = stop - index
        while length > 0:
            chunk = min(length, 65535)
            out.append((value, chunk))
            length -= chunk
        index = stop
    return out


def encode_rle_frame(mask: np.ndarray) -> bytes:
    binary = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    runs = _runs(binary)
    if len(runs) > 65535:
        raise ValueError("too many RLE runs")
    parts = bytearray(_U16.pack(len(runs)))
    for value, count in runs:
        parts.append(value & 0xFF)
        parts += _U16.pack(count)
    return bytes(parts)


def decode_rle_frame(
    blob: bytes, *, height: int, width: int, offset: int = 0
) -> tuple[np.ndarray, int]:
    if offset + 2 > len(blob):
        raise ValueError("truncated mask RLE")
    n_runs = int(_U16.unpack_from(blob, offset)[0])
    cursor = offset + 2
    expected = height * width
    pixels = np.empty(expected, dtype=np.uint8)
    filled = 0
    for _ in range(n_runs):
        if cursor + 3 > len(blob):
            raise ValueError("truncated mask RLE run")
        value = int(blob[cursor])
        count = int(_U16.unpack_from(blob, cursor + 1)[0])
        cursor += 3
        if value not in (0, 1) or count < 1:
            raise ValueError("corrupt mask RLE run")
        if filled + count > expected:
            raise ValueError("mask RLE overflow")
        pixels[filled : filled + count] = value
        filled += count
    if filled != expected:
        raise ValueError("mask RLE underfill")
    return pixels.reshape(height, width), cursor


def encode_mask_stack(masks: np.ndarray, *, mode: int, key_period: int = KEY_PERIOD) -> bytes:
    stack = np.ascontiguousarray(masks != 0, dtype=np.uint8)
    if stack.ndim != 3:
        raise ValueError(f"mask stack must be (T, H, W); got {stack.shape}")
    n_frames, height, width = (int(dim) for dim in stack.shape)
    if mode not in (MODE_INTRA, MODE_XOR_KEY):
        raise ValueError(f"unsupported RLE mode {mode}")
    parts = bytearray(RLE_MAGIC)
    parts.append(RLE_VERSION)
    parts.append(mode)
    parts += _U16.pack(height)
    parts += _U16.pack(width)
    parts += _U16.pack(n_frames)
    parts.append(key_period & 0xFF)
    previous: np.ndarray | None = None
    for index, frame in enumerate(stack):
        intra = mode == MODE_INTRA or index % key_period == 0 or previous is None
        if intra:
            coded = frame
        else:
            assert previous is not None
            coded = np.bitwise_xor(frame, previous)
        parts.append(1 if intra else 0)
        parts += encode_rle_frame(coded)
        previous = frame
    return bytes(parts)


def decode_mask_stack(blob: bytes) -> np.ndarray:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("mask RLE payload must be bytes")
    data = bytes(blob)
    if len(data) < 13 or data[:4] != RLE_MAGIC:
        raise ValueError("corrupt mask RLE magic")
    version = int(data[4])
    mode = int(data[5])
    if version != RLE_VERSION:
        raise ValueError(f"unsupported mask RLE version {version}")
    if mode not in (MODE_INTRA, MODE_XOR_KEY):
        raise ValueError(f"unsupported mask RLE mode {mode}")
    height = int(_U16.unpack_from(data, 6)[0])
    width = int(_U16.unpack_from(data, 8)[0])
    n_frames = int(_U16.unpack_from(data, 10)[0])
    key_period = int(data[12])
    if height < 1 or width < 1 or n_frames < 1 or key_period < 1:
        raise ValueError("corrupt mask RLE header")
    cursor = 13
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    previous: np.ndarray | None = None
    for index in range(n_frames):
        if cursor >= len(data):
            raise ValueError("truncated mask RLE stack")
        intra = int(data[cursor]) == 1
        cursor += 1
        coded, cursor = decode_rle_frame(data, height=height, width=width, offset=cursor)
        if intra:
            frame = coded
        else:
            if previous is None:
                raise ValueError("XOR-delta without a previous mask")
            frame = np.bitwise_xor(coded, previous)
        frames[index] = frame
        previous = frame
    if cursor != len(data):
        raise ValueError("trailing bytes in mask RLE")
    return frames


def _placement_masks(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> np.ndarray:
    height = int(metadata["height"])
    width = int(metadata["width"])
    placements = list(metadata.get("placements") or [])
    if not placements:
        raise ValueError("envelope has no placements")
    stack = np.zeros((len(placements), height, width), dtype=np.uint8)
    for index, item in enumerate(placements):
        key = item.get("mask_key")
        if not key:
            raise ValueError("placement missing mask_key")
        decoded = decode_mask(np.asarray(arrays[key], dtype=np.uint8).tobytes())
        if decoded.shape != (height, width):
            raise ValueError(f"mask shape {decoded.shape} != {(height, width)}")
        stack[index] = (decoded != 0).astype(np.uint8)
    return stack


def _write_native_or_aux(archive: zipfile.ZipFile, name: str, payload: bytes, *, native: bool) -> None:
    compress = zipfile.ZIP_STORED if native else zipfile.ZIP_DEFLATED
    archive.writestr(name, payload, compress_type=compress)


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def _copy_non_mask_members(arrays: dict[str, np.ndarray], archive: zipfile.ZipFile) -> None:
    for name, array in arrays.items():
        if name.startswith("mask_"):
            continue
        native = _is_native(name)
        _write_native_or_aux(archive, f"{name}.npy", _npy_bytes(array), native=native)


def pack_mask_rle(compact: bytes, *, mode: int) -> bytes:
    arrays = load_npz_arrays(compact)
    metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
    stack = _placement_masks(arrays, metadata)
    blob = encode_mask_stack(stack, mode=mode)
    codec: MaskCodec = "rle_intra" if mode == MODE_INTRA else "rle_xor_key4"
    manifest = {
        "schema": "pointstream.e06_floor_pack.v1",
        "mask_codec": codec,
        "placement_codec": "parent_json",
        "n_masks": int(stack.shape[0]),
        "key_period": KEY_PERIOD if mode == MODE_XOR_KEY else None,
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        _copy_non_mask_members(arrays, archive)
        _write_native_or_aux(archive, MASKS_RLE_NAME, blob, native=False)
        _write_native_or_aux(
            archive,
            MANIFEST_NAME,
            json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            native=False,
        )
    return stream.getvalue()


def _delta_int16(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int16)
    if np.any(arr < np.iinfo(np.int16).min) or np.any(arr > np.iinfo(np.int16).max):
        raise ValueError("bbox values do not fit in int16")
    out = np.empty(arr.shape, dtype=np.int16)
    out[0] = arr[0]
    if arr.shape[0] > 1:
        deltas = arr[1:] - arr[:-1]
        if np.any(deltas < np.iinfo(np.int16).min) or np.any(deltas > np.iinfo(np.int16).max):
            raise ValueError("bbox deltas do not fit in int16")
        out[1:] = deltas.astype(np.int16)
    return out


def _undelta_int16(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int16)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 4)
    return np.cumsum(arr.astype(np.int32), axis=0)


def pack_thin_placement(compact: bytes) -> bytes:
    arrays = load_npz_arrays(compact)
    metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
    placements = list(metadata.get("placements") or [])
    bboxes = np.asarray([item["bbox"] for item in placements], dtype=np.int32)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("placements must carry 4-wide bboxes")
    coded = _delta_int16(bboxes)
    background = dict(metadata.get("background") or {})
    homographies = np.asarray(background.get("homographies"), dtype=np.float64)
    if homographies.size == 0:
        raise ValueError("background homographies required for thin envelope")
    homographies = np.ascontiguousarray(homographies, dtype=np.float64)
    thin_placements = []
    for item in placements:
        thin_placements.append(
            {
                "crop_key": item.get("crop_key"),
                "encoded_crop_key": item.get("encoded_crop_key"),
                "mask_key": item.get("mask_key"),
                "pose_key": item.get("pose_key"),
                "motion_key": item.get("motion_key"),
                "frame_index": int(item["frame_index"]),
                "object_id": str(item["object_id"]),
                "is_generated": bool(item.get("is_generated")),
            }
        )
    thin_background = dict(background)
    thin_background["homographies"] = []
    thin_background["geometry_header"] = ""
    thin = {
        "schema": int(metadata["schema"]),
        "mask_wire": metadata.get("mask_wire") or wire_declaration(),
        "frame_count": int(metadata["frame_count"]),
        "height": int(metadata["height"]),
        "width": int(metadata["width"]),
        "background": thin_background,
        "references": metadata.get("references") or {},
        "generator": metadata.get("generator"),
        "placements": thin_placements,
        "residual": metadata.get("residual"),
        "homography_shape": list(homographies.shape),
    }
    manifest = {
        "schema": "pointstream.e06_floor_pack.v1",
        "mask_codec": "parent_packed_bits",
        "placement_codec": "delta_int16_bbox_float32_H",
        "homography_encoding": "float64_le",
        "homography_int16": False,
        "homography_int16_reason": (
            "int16 quantization of 3x3 homographies is not lossless and would change pixels"
        ),
        "n_masks": len(placements),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, array in arrays.items():
            if name == "metadata":
                continue
            _write_native_or_aux(archive, f"{name}.npy", _npy_bytes(array), native=_is_native(name))
        _write_native_or_aux(archive, PLACEMENT_I16_NAME, np.ascontiguousarray(coded).tobytes(), native=False)
        _write_native_or_aux(
            archive, HOMOGRAPHY_NAME, np.ascontiguousarray(homographies).tobytes(), native=False
        )
        _write_native_or_aux(
            archive, THIN_META_NAME, json.dumps(thin, separators=(",", ":")).encode("utf-8"), native=False
        )
        _write_native_or_aux(
            archive,
            MANIFEST_NAME,
            json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            native=False,
        )
    return stream.getvalue()


def _restore_masks(stack: np.ndarray, metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    placements = list(metadata.get("placements") or [])
    if int(stack.shape[0]) != len(placements):
        raise ValueError("mask count does not match placements")
    for index, item in enumerate(placements):
        key = item.get("mask_key") or f"mask_{index}"
        blob = encode_mask(stack[index].astype(bool))
        arrays[key] = np.frombuffer(blob, dtype=np.uint8).copy()
        item["mask_key"] = key
        item["mask_wire"] = {
            **wire_declaration(),
            "shape": [int(dim) for dim in stack[index].shape],
            "payload_bytes": len(blob),
        }
    return arrays


def unpack_floor_pack(payload: bytes) -> bytes:
    """Expand a floor pack to a schema-1 npz, or return a parent envelope unchanged."""
    if not is_floor_pack(payload):
        return bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        arrays: dict[str, np.ndarray] = {}
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[: -len(".npy")]
            buf = io.BytesIO(archive.read(name))
            arrays[key] = np.load(buf, allow_pickle=False)
        mask_codec = manifest.get("mask_codec")
        placement_codec = manifest.get("placement_codec")
        if placement_codec == "parent_json":
            metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        elif placement_codec == "delta_int16_bbox_float32_H":
            thin = json.loads(archive.read(THIN_META_NAME).decode("utf-8"))
            coded = np.frombuffer(archive.read(PLACEMENT_I16_NAME), dtype=np.int16)
            bboxes = _undelta_int16(coded.reshape(-1, 4))
            shape = tuple(int(dim) for dim in thin["homography_shape"])
            homographies = np.frombuffer(archive.read(HOMOGRAPHY_NAME), dtype=np.float64).reshape(shape)
            metadata = dict(thin)
            metadata.pop("homography_shape", None)
            background = dict(metadata.get("background") or {})
            background["homographies"] = homographies.tolist()
            metadata["background"] = background
            placements = []
            for item, bbox in zip(metadata.get("placements") or [], bboxes, strict=True):
                row = dict(item)
                row["bbox"] = [int(v) for v in bbox.tolist()]
                placements.append(row)
            metadata["placements"] = placements
        else:
            raise ValueError(f"unsupported placement codec {placement_codec!r}")
        if mask_codec in ("rle_intra", "rle_xor_key4"):
            if MASKS_RLE_NAME not in names:
                raise ValueError("floor pack missing masks.rle")
            stack = decode_mask_stack(archive.read(MASKS_RLE_NAME))
            arrays.update(_restore_masks(stack, metadata))
        elif mask_codec == "parent_packed_bits":
            pass
        else:
            raise ValueError(f"unsupported mask codec {mask_codec!r}")
        arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def unpack_timed(payload: bytes, timings: dict[str, float] | None = None) -> bytes:
    started = time.perf_counter()
    envelope = unpack_floor_pack(payload)
    if timings is not None:
        timings["floor_unpack_s"] = float(timings.get("floor_unpack_s", 0.0)) + (
            time.perf_counter() - started
        )
    return envelope


def physical_ledger(payload: bytes) -> dict[str, Any]:
    """Name B/F/M/R/H from physical zip members so they sum to file length T."""
    transport_t = len(payload)
    panorama_b = 0
    actor_f = 0
    residual_r = 0
    mask_m = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = []
        for info in archive.infolist():
            name = info.filename
            size = int(info.compress_size)
            stem = name.replace(".npy", "")
            item = {
                "name": name,
                "file_size": int(info.file_size),
                "compress_size": size,
                "stored_native": info.compress_type == zipfile.ZIP_STORED and _is_native(stem),
            }
            entries.append(item)
            if stem.startswith("background_payload_") or stem.startswith("background_header_"):
                panorama_b += size
            elif stem.startswith("ref_") or stem.startswith("encoded_crop_") or name.startswith("ref_"):
                actor_f += size
            elif stem.startswith("residual_bitstream") or stem.startswith("residual_raw"):
                residual_r += size
            elif stem.startswith("mask_") or name == MASKS_RLE_NAME:
                mask_m += size
    header_h = transport_t - panorama_b - actor_f - residual_r - mask_m
    if header_h < 0:
        raise ValueError(
            f"physical ledger over-assigned: B={panorama_b} F={actor_f} R={residual_r} "
            f"M={mask_m} T={transport_t}"
        )
    summed = panorama_b + actor_f + residual_r + mask_m + header_h
    if summed != transport_t:
        raise ValueError(f"ledger does not reconcile: sum={summed} T={transport_t}")
    return {
        "panorama": panorama_b,
        "actor_reference": actor_f,
        "residual": residual_r,
        "metadata": mask_m,
        "unallocated_H": header_h,
        "transport_total": transport_t,
        "reconciled": True,
        "note": (
            "B/F/R/M/H are physical compress_sizes. H holds zip framing, thin/fat metadata, "
            "homographies, and remaining aux. T is the file length."
        ),
        "entries": entries,
    }


def zip_inventory_floor(payload: bytes) -> dict[str, Any]:
    stored_native = 0
    compressed_aux = 0
    entries = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            stem = info.filename.replace(".npy", "")
            stored = info.compress_type == zipfile.ZIP_STORED and _is_native(stem)
            item = {
                "name": info.filename,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "stored_native": stored,
            }
            entries.append(item)
            if stored:
                stored_native += int(info.compress_size)
            else:
                compressed_aux += int(info.compress_size)
    return {
        "transport_total": len(payload),
        "stored_native_bytes": stored_native,
        "compressed_auxiliary_bytes": compressed_aux,
        "zip_framing_bytes": max(0, len(payload) - stored_native - compressed_aux),
        "entries": entries,
        "note": (
            "T is the packed file length. Native codec payloads are stored as-is. "
            "Do not treat T minus logical B+F+R as envelope remainder."
        ),
    }


def required_decode_fields(envelope: bytes) -> dict[str, Any]:
    arrays = load_npz_arrays(envelope)
    metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
    missing = []
    for key in ("schema", "frame_count", "height", "width", "background", "placements", "residual"):
        if key not in metadata:
            missing.append(key)
    background = metadata.get("background") or {}
    for key in ("homographies", "width", "height", "wire_payload_keys", "sidecar_codec"):
        if key not in background:
            missing.append(f"background.{key}")
    for index, item in enumerate(metadata.get("placements") or []):
        for key in ("bbox", "frame_index", "object_id", "mask_key"):
            if key not in item or item[key] in (None, ""):
                missing.append(f"placements[{index}].{key}")
        mask_key = item.get("mask_key")
        if mask_key and mask_key not in arrays:
            missing.append(f"array:{mask_key}")
    for payload_key in background.get("wire_payload_keys") or ():
        if payload_key not in arrays:
            missing.append(f"array:{payload_key}")
    if missing:
        raise ValueError(f"unpack dropped required decode fields: {missing}")
    return {
        "frame_count": metadata["frame_count"],
        "height": metadata["height"],
        "width": metadata["width"],
        "n_placements": len(metadata.get("placements") or []),
        "n_homographies": len(background.get("homographies") or ()),
    }


__all__ = [
    "KEY_PERIOD",
    "MODE_INTRA",
    "MODE_XOR_KEY",
    "NATIVE_PREFIXES",
    "decode_mask_stack",
    "encode_mask_stack",
    "is_floor_pack",
    "pack_mask_rle",
    "pack_thin_placement",
    "physical_ledger",
    "required_decode_fields",
    "unpack_floor_pack",
    "unpack_timed",
    "zip_inventory_floor",
]
