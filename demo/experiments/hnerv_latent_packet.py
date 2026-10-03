"""Self-contained lossless packet prototypes for frozen HNeRV latent codes."""

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


MAGIC = b"PSHN1\0"
SCHEMA = "pointstream.hnerv.latent.v1"
METHODS = ("packed6", "zlib6", "delta-zlib6")
SEGMENT_LENGTHS = (1, 8, 32)


def _array_metadata(value: Any) -> dict[str, Any]:
    """Preserve quantizer array bytes, dtype, shape, and byte order verbatim."""
    if isinstance(value, dict) and {"dtype", "shape", "data_b64"}.issubset(value):
        _decode_array(value)
        return {"dtype": value["dtype"], "shape": list(value["shape"]), "data_b64": value["data_b64"]}
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("quantizer metadata must be finite numeric arrays")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(record: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(record["data_b64"], validate=True)
    dtype = np.dtype(record["dtype"])
    if dtype.kind not in "iuf":
        raise ValueError("quantizer metadata must use a numeric dtype")
    shape = tuple(int(item) for item in record["shape"])
    if any(item < 0 for item in shape) or math.prod(shape) * dtype.itemsize != len(raw):
        raise ValueError("quantizer metadata byte count does not match dtype and shape")
    value = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(value).all():
        raise ValueError("quantizer metadata must be finite")
    return value


def _pack6(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) >= 64:
        raise ValueError("six-bit codes must be in [0, 63]")
    shifts = np.arange(5, -1, -1, dtype=np.uint8)
    bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits, bitorder="big").tobytes()


def _unpack6(payload: bytes, count: int) -> np.ndarray:
    if count < 0 or len(payload) != (count * 6 + 7) // 8:
        raise ValueError("packed six-bit payload has the wrong length")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[: count * 6]
    weights = (1 << np.arange(5, -1, -1, dtype=np.uint8)).astype(np.uint8)
    return (bits.reshape(count, 6) * weights[None, :]).sum(axis=1).astype(np.uint8)


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
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
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique within an independently decoded segment")
    quant = metadata.get("quantizer")
    if not isinstance(quant, dict) or "min" not in quant or "scale" not in quant:
        raise ValueError("exact quantizer min and scale arrays are required")
    decoder_setup_bytes = metadata.get("decoder_setup_bytes")
    if isinstance(decoder_setup_bytes, bool) or not isinstance(decoder_setup_bytes, int) or decoder_setup_bytes < 0:
        raise ValueError("decoder_setup_bytes must be a nonnegative integer, reported separately")
    return {
        "checkpoint_sha256": checkpoint,
        "frame_start": start,
        "fps": fps,
        "frame_ids": frame_ids,
        "quantizer": {
            "min": _array_metadata(quant["min"]),
            "scale": _array_metadata(quant["scale"]),
        },
        "decoder_setup_bytes": decoder_setup_bytes,
    }


def encode_packet(codes: np.ndarray, metadata: dict[str, Any], *, method: str) -> bytes:
    codes = np.asarray(codes)
    if codes.ndim != 4 or codes.shape[0] not in SEGMENT_LENGTHS:
        raise ValueError("codes must have shape [1|8|32, H, W, C]")
    if codes.dtype != np.uint8 or (codes.size and int(codes.max()) >= 64):
        raise ValueError("codes must be uint8 six-bit quantized values")
    if method not in METHODS:
        raise ValueError(f"unsupported lossless packet method: {method}")
    normalized = _validate_metadata(metadata)
    if normalized["frame_ids"] is not None and len(normalized["frame_ids"]) != codes.shape[0]:
        raise ValueError("frame_ids length does not match the packet frame count")
    raw = codes.tobytes(order="C")
    packed = _pack6(codes)
    if method == "packed6":
        payload = packed
    elif method == "zlib6":
        payload = zlib.compress(packed, level=9)
    else:
        delta = codes.copy()
        if len(codes) > 1:
            delta[1:] = (codes[1:].astype(np.int16) - codes[:-1].astype(np.int16)) % 64
        payload = zlib.compress(_pack6(delta), level=9)
    header = {
        "schema": SCHEMA,
        "method": method,
        "shape": list(codes.shape),
        "dtype": "uint8",
        "bit_depth": 6,
        "frame_start": normalized["frame_start"],
        "frame_count": int(codes.shape[0]),
        "fps": normalized["fps"],
        "frame_ids": normalized["frame_ids"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "quantizer": normalized["quantizer"],
        "decoder_setup_bytes": normalized["decoder_setup_bytes"],
        "compressor": "none" if method == "packed6" else "zlib-9",
        "payload_bytes": len(payload),
        "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
        "codes_crc32": zlib.crc32(raw) & 0xFFFFFFFF,
    }
    canonical_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header["header_crc32"] = zlib.crc32(canonical_header) & 0xFFFFFFFF
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + payload


def decode_packet(packet: bytes, *, expected_checkpoint_sha256: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not packet.startswith(MAGIC):
        raise ValueError("invalid HNeRV packet magic")
    if len(packet) < len(MAGIC) + 4:
        raise ValueError("truncated HNeRV packet header")
    header_length = struct.unpack(">I", packet[len(MAGIC):len(MAGIC) + 4])[0]
    header_start = len(MAGIC) + 4
    payload_start = header_start + header_length
    if payload_start > len(packet):
        raise ValueError("truncated HNeRV packet header")
    try:
        header = json.loads(packet[header_start:payload_start])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid HNeRV packet header") from exc
    if not isinstance(header, dict):
        raise ValueError("HNeRV packet header must be a JSON object")
    payload = packet[payload_start:]
    header_crc = header.pop("header_crc32", None)
    canonical_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if header_crc != (zlib.crc32(canonical_header) & 0xFFFFFFFF):
        raise ValueError("header CRC mismatch")
    header["header_crc32"] = header_crc
    if header.get("schema") != SCHEMA or header.get("bit_depth") != 6 or header.get("dtype") != "uint8":
        raise ValueError("unsupported HNeRV packet schema or code type")
    if header.get("payload_bytes") != len(payload) or (zlib.crc32(payload) & 0xFFFFFFFF) != header.get("payload_crc32"):
        raise ValueError("payload length or CRC mismatch")
    if expected_checkpoint_sha256 is not None and header.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("packet checkpoint identity mismatch")
    shape = tuple(int(value) for value in header.get("shape", ()))
    if len(shape) != 4 or shape[0] not in SEGMENT_LENGTHS or any(value <= 0 for value in shape):
        raise ValueError("invalid latent code shape")
    if tuple(shape[1:3]) != (9, 16) or shape[3] not in (3, 4):
        raise ValueError("latent code map must match the trained 9x16x3/4 smoke shape")
    if header.get("frame_count") != shape[0]:
        raise ValueError("frame count and tensor shape disagree")
    try:
        _validate_metadata({
            "checkpoint_sha256": header["checkpoint_sha256"],
            "frame_start": header["frame_start"],
            "fps": header["fps"],
            "frame_ids": header["frame_ids"],
            "quantizer": header["quantizer"],
            "decoder_setup_bytes": header["decoder_setup_bytes"],
        })
    except (KeyError, TypeError) as exc:
        raise ValueError("packet metadata is incomplete") from exc
    frame_ids = header["frame_ids"]
    if len(frame_ids) != shape[0]:
        raise ValueError("packet frame identity count does not match tensor shape")
    count = math.prod(shape)
    expected_packed_bytes = (count * 6 + 7) // 8
    method = header.get("method")
    expected_compressor = "none" if method == "packed6" else "zlib-9"
    if header.get("compressor") != expected_compressor:
        raise ValueError("packet compressor identifier does not match its method")
    if method == "packed6":
        packed = payload
    elif method in ("zlib6", "delta-zlib6"):
        try:
            inflater = zlib.decompressobj()
            packed = inflater.decompress(payload, expected_packed_bytes + 1)
            if len(packed) > expected_packed_bytes or inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
                raise ValueError("zlib output exceeds expected packet dimensions")
        except zlib.error as exc:
            raise ValueError("corrupt zlib packet payload") from exc
    else:
        raise ValueError("unknown packet method")
    flat = _unpack6(packed, count)
    codes = flat.reshape(shape)
    if method == "delta-zlib6" and shape[0] > 1:
        codes = codes.copy()
        for frame in range(1, shape[0]):
            codes[frame] = (codes[frame].astype(np.uint16) + codes[frame - 1].astype(np.uint16)) % 64
        codes = codes.astype(np.uint8)
    if (zlib.crc32(codes.tobytes(order="C")) & 0xFFFFFFFF) != header.get("codes_crc32"):
        raise ValueError("decoded code CRC mismatch")
    for name in ("min", "scale"):
        _decode_array(header["quantizer"][name])
    return codes, header


def write_segment_packets(codes: np.ndarray, metadata: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write actual 1/8/32-frame packet files for one exact 32-frame cut."""
    codes = np.asarray(codes)
    if codes.ndim != 4 or codes.shape[0] != 32:
        raise ValueError("latent packet smoke requires exactly 32 ordered frames")
    if tuple(codes.shape[1:3]) != (9, 16) or codes.shape[3] not in (3, 4):
        raise ValueError("trained latent map must be 9x16x3 or 9x16x4")
    if codes.dtype != np.uint8 or (codes.size and int(codes.max()) >= 64):
        raise ValueError("codes must be uint8 six-bit quantized values")
    _validate_metadata(metadata)
    if len(metadata["frame_ids"]) != 32:
        raise ValueError("the 32-frame smoke requires exactly 32 unique frame identities")
    if metadata["frame_start"] != 120:
        raise ValueError("the primary HNeRV smoke cut must start at hold-out frame 120")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix packet outputs into a nonempty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_start = metadata.get("frame_start")
    frame_ids = metadata.get("frame_ids")
    records = []
    for length in SEGMENT_LENGTHS:
        for start in range(0, 32, length):
            segment = codes[start:start + length]
            segment_meta = dict(metadata)
            segment_meta["frame_start"] = base_start + start
            if frame_ids is not None:
                segment_meta["frame_ids"] = frame_ids[start:start + length]
            for method in METHODS:
                packet = encode_packet(segment, segment_meta, method=method)
                filename = f"{length:02d}_{start:05d}_{method}.pshn"
                path = output_dir / filename
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite packet: {path}")
                decoded, header = decode_packet(packet, expected_checkpoint_sha256=metadata["checkpoint_sha256"])
                if not np.array_equal(decoded, segment):
                    raise RuntimeError(f"packet failed exact round trip: {path}")
                with path.open("xb") as stream:
                    stream.write(packet)
                records.append({
                    "path": str(path), "method": method, "segment_length": length,
                    "frame_start": header["frame_start"], "stream_bytes": len(packet),
                    "decoder_setup_bytes": header["decoder_setup_bytes"],
                    "codes_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
                })
    return {"schema": SCHEMA, "record_count": len(records), "packets": records}
