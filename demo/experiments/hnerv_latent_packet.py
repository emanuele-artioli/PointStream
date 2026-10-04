"""Self-contained lossless packets for frozen HNeRV latent codes.

Codes are HNeRV embeddings quantized by ``hnerv_utils.quant_tensor``: uint8
values below ``2**bit_depth`` in NCHW order ``[frames, channels, 9, 16]``.
The header stores the quantizer's min/scale arrays verbatim (dtype, shape,
byte order), so ``dequantize`` reproduces HNeRV's
``min + scale * code`` broadcast exactly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib
from typing import Any

import numpy as np


MAGIC = b"PSHN2\0"
SCHEMA = "pointstream.hnerv.latent.v2"
LAYOUT = "NCHW"
METHODS = ("packed", "zlib", "delta-zlib")
BIT_DEPTHS = (4, 6)
SEGMENT_LENGTHS = (1, 8, 32)
EMBED_HW = (9, 16)
COMPRESSORS = {"packed": "none", "zlib": "zlib-9", "delta-zlib": "zlib-9"}


def _array_record(value: Any) -> dict[str, Any]:
    """Preserve quantizer array bytes, dtype, shape, and byte order verbatim."""
    if isinstance(value, dict) and {"dtype", "shape", "data_b64"}.issubset(value):
        _decode_array(value)
        return {"dtype": value["dtype"], "shape": list(value["shape"]), "data_b64": value["data_b64"]}
    array = np.asarray(value)
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("quantizer min/scale must be finite floating-point arrays")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(record: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(record["data_b64"], validate=True)
    dtype = np.dtype(record["dtype"])
    if dtype.kind != "f":
        raise ValueError("quantizer min/scale must use a floating-point dtype")
    shape = tuple(int(item) for item in record["shape"])
    if any(item < 0 for item in shape) or math.prod(shape) * dtype.itemsize != len(raw):
        raise ValueError("quantizer metadata byte count does not match dtype and shape")
    value = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():
        raise ValueError("quantizer metadata must be finite")
    return value


def dequantize(codes: np.ndarray, quantizer: dict[str, Any]) -> np.ndarray:
    """HNeRV ``dequant_tensor`` in float32: ``min + scale * code``, broadcast."""
    minimum = _decode_array(_array_record(quantizer["min"])).astype(np.float32)
    scale = _decode_array(_array_record(quantizer["scale"])).astype(np.float32)
    return minimum + scale * np.asarray(codes).astype(np.float32)


def _pack(values: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) >= 1 << bits:
        raise ValueError(f"{bits}-bit codes must be below {1 << bits}")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint8)
    unpacked = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(unpacked, bitorder="big").tobytes()


def _unpack(payload: bytes, count: int, bits: int) -> np.ndarray:
    if count < 0 or len(payload) != (count * bits + 7) // 8:
        raise ValueError("packed payload has the wrong length")
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[: count * bits]
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint16)
    return (unpacked.reshape(count, bits).astype(np.uint16) * weights[None, :]).sum(axis=1).astype(np.uint8)


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 4 or shape[0] not in SEGMENT_LENGTHS or any(value <= 0 for value in shape):
        raise ValueError("codes must have NCHW shape [1|8|32, C, 9, 16]")
    if tuple(shape[2:]) != EMBED_HW or shape[1] not in (3, 4):
        raise ValueError("latent code map must be NCHW with the trained 3 or 4 channels over 9x16")


def _validate_metadata(metadata: dict[str, Any], shape: tuple[int, ...]) -> dict[str, Any]:
    checkpoint = metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint):
        raise ValueError("checkpoint_sha256 must be a complete lowercase SHA-256")
    start = metadata.get("frame_start")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("frame_start must be a nonnegative integer")
    fps = metadata.get("fps")
    if fps != {"numerator": 30, "denominator": 1}:
        raise ValueError("the packet prototype requires the recorded 30/1 fps timebase")
    frame_ids = metadata.get("frame_ids")
    if not isinstance(frame_ids, list) or any(not isinstance(item, str) or not item for item in frame_ids):
        raise ValueError("frame_ids must be an explicit ordered list of strings")
    if len(set(frame_ids)) != len(frame_ids) or len(frame_ids) != shape[0]:
        raise ValueError("frame_ids must be unique and match the packet frame count")
    quant = metadata.get("quantizer")
    if not isinstance(quant, dict) or "min" not in quant or "scale" not in quant:
        raise ValueError("exact quantizer min and scale arrays are required")
    records = {name: _array_record(quant[name]) for name in ("min", "scale")}
    for name, record in records.items():
        try:
            np.broadcast_shapes(tuple(record["shape"]), shape)
        except ValueError as exc:
            raise ValueError(f"quantizer {name} shape does not broadcast to the codes") from exc
    setup = metadata.get("decoder_setup_bytes")
    if isinstance(setup, bool) or not isinstance(setup, int) or setup < 0:
        raise ValueError("decoder_setup_bytes must be a nonnegative integer, reported separately")
    return {
        "checkpoint_sha256": checkpoint, "frame_start": start, "fps": fps,
        "frame_ids": frame_ids, "quantizer": records, "decoder_setup_bytes": setup,
    }


def _canonical(header: dict[str, Any]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_packet(codes: np.ndarray, metadata: dict[str, Any], *, method: str, bit_depth: int) -> bytes:
    codes = np.asarray(codes)
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(f"unsupported bit depth {bit_depth}")
    _validate_shape(codes.shape)
    if codes.dtype != np.uint8 or (codes.size and int(codes.max()) >= 1 << bit_depth):
        raise ValueError(f"codes must be uint8 {bit_depth}-bit quantized values")
    if method not in METHODS:
        raise ValueError(f"unsupported lossless packet method: {method}")
    normalized = _validate_metadata(metadata, codes.shape)
    modulus = 1 << bit_depth
    if method == "delta-zlib":
        delta = codes.copy()
        if len(codes) > 1:
            delta[1:] = (codes[1:].astype(np.int16) - codes[:-1].astype(np.int16)) % modulus
        payload = zlib.compress(_pack(delta, bit_depth), level=9)
    else:
        packed = _pack(codes, bit_depth)
        payload = packed if method == "packed" else zlib.compress(packed, level=9)
    header = {
        "schema": SCHEMA, "method": method, "layout": LAYOUT, "shape": list(codes.shape),
        "dtype": "uint8", "bit_depth": bit_depth, "frame_start": normalized["frame_start"],
        "frame_count": int(codes.shape[0]), "fps": normalized["fps"], "frame_ids": normalized["frame_ids"],
        "checkpoint_sha256": normalized["checkpoint_sha256"], "quantizer": normalized["quantizer"],
        "decoder_setup_bytes": normalized["decoder_setup_bytes"], "compressor": COMPRESSORS[method],
        "payload_bytes": len(payload), "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
        "codes_crc32": zlib.crc32(codes.tobytes(order="C")) & 0xFFFFFFFF,
    }
    header["header_crc32"] = zlib.crc32(_canonical(header)) & 0xFFFFFFFF
    header_bytes = _canonical(header)
    return MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + payload


def decode_packet(packet: bytes, *, expected_checkpoint_sha256: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not packet.startswith(MAGIC):
        raise ValueError("invalid HNeRV packet magic")
    if len(packet) < len(MAGIC) + 4:
        raise ValueError("truncated HNeRV packet header")
    header_start = len(MAGIC) + 4
    header_length = struct.unpack(">I", packet[len(MAGIC):header_start])[0]
    payload_start = header_start + header_length
    if payload_start > len(packet):
        raise ValueError("truncated HNeRV packet header")
    try:
        header = json.loads(packet[header_start:payload_start])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid HNeRV packet header") from exc
    if not isinstance(header, dict):
        raise ValueError("HNeRV packet header must be a JSON object")
    header_crc = header.pop("header_crc32", None)
    if header_crc != (zlib.crc32(_canonical(header)) & 0xFFFFFFFF):
        raise ValueError("header CRC mismatch")
    header["header_crc32"] = header_crc
    payload = packet[payload_start:]
    bits = header.get("bit_depth")
    if header.get("schema") != SCHEMA or header.get("layout") != LAYOUT or bits not in BIT_DEPTHS or header.get("dtype") != "uint8":
        raise ValueError("unsupported HNeRV packet schema, layout, or code type")
    if header.get("payload_bytes") != len(payload) or (zlib.crc32(payload) & 0xFFFFFFFF) != header.get("payload_crc32"):
        raise ValueError("payload length or CRC mismatch")
    if expected_checkpoint_sha256 is not None and header.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("packet checkpoint identity mismatch")
    shape = tuple(int(value) for value in header.get("shape", ()))
    _validate_shape(shape)
    if header.get("frame_count") != shape[0]:
        raise ValueError("frame count and tensor shape disagree")
    try:
        _validate_metadata({key: header[key] for key in ("checkpoint_sha256", "frame_start", "fps", "frame_ids", "quantizer", "decoder_setup_bytes")}, shape)
    except (KeyError, TypeError) as exc:
        raise ValueError("packet metadata is incomplete") from exc
    method = header.get("method")
    if method not in METHODS or header.get("compressor") != COMPRESSORS[method]:
        raise ValueError("packet compressor identifier does not match its method")
    count = math.prod(shape)
    expected_packed = (count * bits + 7) // 8
    if method == "packed":
        packed = payload
    else:
        try:
            inflater = zlib.decompressobj()
            packed = inflater.decompress(payload, expected_packed + 1)
            if len(packed) > expected_packed or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
                raise ValueError("zlib output exceeds expected packet dimensions")
        except zlib.error as exc:
            raise ValueError("corrupt zlib packet payload") from exc
    codes = _unpack(packed, count, bits).reshape(shape)
    if method == "delta-zlib" and shape[0] > 1:
        codes = codes.astype(np.uint16)
        for frame in range(1, shape[0]):
            codes[frame] = (codes[frame] + codes[frame - 1]) % (1 << bits)
        codes = codes.astype(np.uint8)
    if (zlib.crc32(codes.tobytes(order="C")) & 0xFFFFFFFF) != header.get("codes_crc32"):
        raise ValueError("decoded code CRC mismatch")
    return codes, header


def write_segment_packets(segments: list[dict[str, Any]], output_dir: Path, *, bit_depth: int) -> dict[str, Any]:
    """Write every method's packet file for independently quantized segments.

    Each segment is ``{"codes": [L, C, 9, 16] uint8, "metadata": {...}}`` with
    its own quantizer, so no packet needs a previous or later one.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix packet outputs into a nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for segment in segments:
        codes, metadata = np.asarray(segment["codes"]), segment["metadata"]
        length = int(codes.shape[0])
        for method in METHODS:
            packet = encode_packet(codes, metadata, method=method, bit_depth=bit_depth)
            path = output_dir / f"b{bit_depth}_{length:02d}_{metadata['frame_start']:05d}_{method}.pshn"
            decoded, header = decode_packet(packet, expected_checkpoint_sha256=metadata["checkpoint_sha256"])
            if not np.array_equal(decoded, codes):
                raise RuntimeError(f"packet failed exact round trip: {path}")
            with path.open("xb") as stream:
                stream.write(packet)
            records.append({
                "path": str(path), "method": method, "bit_depth": bit_depth, "segment_length": length,
                "frame_start": header["frame_start"], "file_bytes": path.stat().st_size,
                "payload_bytes": header["payload_bytes"], "header_bytes": len(packet) - header["payload_bytes"],
                "decoder_setup_bytes": header["decoder_setup_bytes"],
                "codes_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
            })
    return {"schema": SCHEMA, "record_count": len(records), "packets": records}


def summarize_packets(records: list[dict[str, Any]], *, frames: int, setup_bytes: int) -> list[dict[str, Any]]:
    """Latent-only and setup-inclusive totals per method/segment length.

    Setup (shared decoder) bytes are charged once per tested cut, never once
    per segment, and never inside latent-only totals.
    """
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["bit_depth"], record["segment_length"], record["method"]), []).append(record)
    rows = []
    for (bits, length, method), group in sorted(groups.items()):
        covered = sum(record["segment_length"] for record in group)
        if covered != frames:
            raise ValueError(f"{method}/{length} packets cover {covered} of {frames} frames")
        latent = sum(record["file_bytes"] for record in group)
        rows.append({
            "bit_depth": bits, "segment_length": length, "method": method, "packets": len(group),
            "latent_only_bytes": latent, "latent_only_kbps": 8 * latent * 30 / (1000 * frames),
            "header_bytes": sum(record["header_bytes"] for record in group),
            "decoder_setup_bytes": setup_bytes, "setup_inclusive_bytes": latent + setup_bytes,
        })
    for row in rows:
        baseline = next(r for r in rows if r["bit_depth"] == row["bit_depth"] and r["segment_length"] == row["segment_length"] and r["method"] == "packed")
        row["saving_vs_packed"] = 1 - row["latent_only_bytes"] / baseline["latent_only_bytes"]
    return rows
