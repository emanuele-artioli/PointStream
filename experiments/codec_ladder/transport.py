"""PLP1: one native base and optional static rectangular native pixel patch.

Every byte is counted; placement and framing are serialized. Decoder inputs are
only packet bytes and the installed dav1d executable. This profile has no masks,
source images, learned models or encoder caches at the receiver.
"""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.codecs import svtav1
from src.pointstream import codec

MAGIC = b"PLP1"
HEADER = struct.Struct(">4sIII")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def planes(frame: np.ndarray, width: int, height: int) -> list[np.ndarray]:
    flat = np.asarray(frame).reshape(-1)
    return [flat[:width * height].reshape(height, width),
            flat[width * height:width * height * 5 // 4].reshape(height // 2, width // 2),
            flat[width * height * 5 // 4:].reshape(height // 2, width // 2)]


def crop(frame: np.ndarray, width: int, height: int, box: list[int]) -> bytes:
    x, y, w, h = box
    return b"".join(p[y // scale:(y + h) // scale, x // scale:(x + w) // scale].tobytes()
                    for p, scale in zip(planes(frame, width, height), (1, 2, 2)))


def composite(base: np.ndarray, patch: np.ndarray, width: int, height: int, box: list[int]) -> bytes:
    x, y, w, h = box
    output = [plane.copy() for plane in planes(base, width, height)]
    for plane, pixels, scale in zip(output, planes(patch, w, h), (1, 2, 2)):
        plane[y // scale:(y + h) // scale, x // scale:(x + w) // scale] = pixels
    return b"".join(plane.tobytes() for plane in output)


def validate(meta: dict[str, Any], base: bytes, patch: bytes) -> None:
    if set(meta) != {"native", "box", "profile"} or meta["profile"] != "static-pixel-patch-v1":
        raise ValueError("unsupported patch metadata")
    codec.validate(meta["native"])
    codec.check_native(meta["native"], base)
    box = meta["box"]
    if box is None:
        if patch:
            raise ValueError("patch without placement")
        return
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or v % 2 for v in box)
            or min(box[:2]) < 0 or min(box[2:]) < 64
            or box[0] + box[2] > meta["native"]["width"] or box[1] + box[3] > meta["native"]["height"]):
        raise ValueError("invalid even patch placement")
    codec.check_native({**meta["native"], "width": box[2], "height": box[3]}, patch)


def pack(meta: dict[str, Any], base: bytes, patch: bytes = b"") -> bytes:
    validate(meta, base, patch)
    header = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body = HEADER.pack(MAGIC, len(header), len(base), len(patch)) + header + base + patch
    if len(body) + 32 > codec.MAX_PACKET:
        raise ValueError("patch packet too large")
    return body + hashlib.sha256(body).digest()


def unpack(data: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    if not HEADER.size + 32 < len(data) <= codec.MAX_PACKET:
        raise ValueError("invalid patch packet length")
    magic, header, base, patch = HEADER.unpack_from(data)
    if magic != MAGIC or not 1 <= header <= 4096 or HEADER.size + header + base + patch + 32 != len(data):
        raise ValueError("invalid patch packet header")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("patch packet checksum mismatch")
    offset = HEADER.size
    meta = json.loads(data[offset:offset + header])
    if not isinstance(meta, dict):
        raise ValueError("invalid patch metadata")
    offset += header
    native, enhancement = data[offset:offset + base], data[offset + base:-32]
    if pack(meta, native, enhancement) != data:
        raise ValueError("noncanonical patch packet")
    return meta, native, enhancement


def decode_native(native: bytes, path: Path, meta: dict[str, Any], decoder: str) -> dict[str, Any]:
    """Decode self-describing Y4M and verify actual geometry, chroma and frames."""
    source, y4m = path.with_suffix(".ivf"), path.with_suffix(".y4m")
    source.write_bytes(native)
    command = [decoder, "-i", str(source), "-o", str(y4m), "--muxer", "yuv4mpeg2", "--threads", "1",
               "--limit", str(meta["frames"]), "--sizelimit", str(meta["width"] * meta["height"]), "--quiet"]
    timing = svtav1.run(command, 600)
    expected = meta["width"] * meta["height"] * 3 // 2
    with y4m.open("rb") as reader, path.open("xb") as writer:
        header = reader.readline(4096).decode("ascii").split()
        required = ["YUV4MPEG2", f'W{meta["width"]}', f'H{meta["height"]}', f'F{meta["fps"][0]}:{meta["fps"][1]}']
        if not all(token in header for token in required) or not any(t in header for t in ("C420", "C420jpeg", "C420mpeg2")):
            raise ValueError("native decoded geometry/chroma differs from packet")
        for _ in range(meta["frames"]):
            if reader.readline(4096).strip() != b"FRAME":
                raise ValueError("native frame count/format mismatch")
            pixels = reader.read(expected)
            if len(pixels) != expected:
                raise ValueError("truncated native frame")
            writer.write(pixels)
        if reader.read(1):
            raise ValueError("unexpected extra native pixels")
    return {"command": command, "timing": timing, "decoded_sha256": digest(path.read_bytes()), "y4m_header": header}


def decode(data: bytes, output: Path, *, scratch: Path, decoder: str = "dav1d") -> dict[str, Any]:
    meta, base, patch = unpack(data)
    identity = svtav1.tool(decoder)
    if identity["sha256"] != meta["native"]["decoder_sha256"]:
        raise ValueError("installed decoder mismatch")
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="plp-receiver-", dir=scratch) as tmp:
        root = Path(tmp)
        base_path, patch_path = root / "base.yuv", root / "patch.yuv"
        receipts = [decode_native(base, base_path, meta["native"], decoder)]
        box, count = meta["box"], meta["native"]["frames"]
        width, height = meta["native"]["width"], meta["native"]["height"]
        if box is not None:
            receipts.append(decode_native(patch, patch_path, {**meta["native"], "width": box[2], "height": box[3]}, decoder))
        with output.open("xb") as target, base_path.open("rb") as source:
            if box is None:
                for _ in range(count):
                    target.write(source.read(width * height * 3 // 2))
            else:
                with patch_path.open("rb") as enhancement:
                    for _ in range(count):
                        b = np.frombuffer(source.read(width * height * 3 // 2), np.uint8)
                        p = np.frombuffer(enhancement.read(box[2] * box[3] * 3 // 2), np.uint8)
                        target.write(composite(b, p, width, height, box))
    return {"packet_sha256": digest(data), "packet_bytes": len(data), "native_base_bytes": len(base),
            "patch_bytes": len(patch), "placement_and_framing_bytes": len(data) - len(base) - len(patch),
            "decoded_sha256": digest(output.read_bytes()), "receipts": receipts,
            "decoder": identity, "source_access": "no source argument; OS denial requires external qualification"}
