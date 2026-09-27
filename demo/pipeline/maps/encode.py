"""Native map payloads: bit-packed zstd blobs, COCO RLE streams, optional gray AV1."""

from __future__ import annotations

import json
import subprocess
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - env without zstd uses zlib
    zstd = None

MASK_STREAM_SCHEMA = "pointstream.maps.coco_rle.v1"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def pack_binary_mask(mask: np.ndarray) -> bytes:
    """Pack a 2-D boolean/0-1 array to bits, then zstd (zlib fallback)."""
    bits = np.packbits(np.asarray(mask, dtype=np.uint8).ravel())
    raw = bits.tobytes()
    if zstd is not None:
        return zstd.ZstdCompressor(level=3).compress(raw)
    return zlib.compress(raw, level=6)


def unpack_binary_mask(payload: bytes, height: int, width: int) -> np.ndarray:
    if zstd is not None:
        raw = zstd.ZstdDecompressor().decompress(payload)
    else:
        raw = zlib.decompress(payload)
    bits = np.frombuffer(raw, dtype=np.uint8)
    flat = np.unpackbits(bits)[: height * width]
    return flat.reshape(height, width).astype(np.uint8)


def write_zstd_blob(payload: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def try_encode_gray_av1(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float = 30.0,
    crf: int = 40,
    preset: int = 8,
) -> Path | None:
    """Return the AV1 path, or None if ffmpeg/libsvtav1 is missing or fails."""
    try:
        return encode_gray_av1(frames, output_path, fps=fps, crf=crf, preset=preset)
    except (FileNotFoundError, OSError, RuntimeError):
        return None


def write_u8_stack(frames: list[np.ndarray], npy_path: Path, bin_path: Path | None = None) -> Path:
    """Persist a T×H×W uint8 volume as .npy, and optionally a raw .bin sibling."""
    if not frames:
        raise ValueError("no frames")
    stack = np.stack([np.ascontiguousarray(f, dtype=np.uint8) for f in frames], axis=0)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, stack)
    if bin_path is not None:
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        stack.tofile(bin_path)
    return npy_path


def encode_gray_av1(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float = 30.0,
    crf: int = 40,
    preset: int = 8,
) -> Path:
    """Encode HxW uint8 grayscale frames as AV1 4:0:0. Preview colormaps stay out."""
    if not frames:
        raise ValueError("no frames")
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libsvtav1",
        "-pix_fmt",
        "gray",
        "-crf",
        str(crf),
        "-preset",
        str(preset),
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            gray = frame if frame.ndim == 2 else frame[:, :, 0]
            proc.stdin.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg gray AV1 failed ({code}): {err[-400:]!r}")
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    return output_path


def compress_bytes(raw: bytes) -> tuple[bytes, str]:
    """zstd when present, zlib otherwise. Returns (blob, codec_name)."""
    if zstd is not None:
        return zstd.ZstdCompressor(level=3).compress(raw), "zstd"
    return zlib.compress(raw, level=6), "zlib"


def decompress_bytes(blob: bytes) -> bytes:
    if blob.startswith(ZSTD_MAGIC):
        if zstd is None:
            raise RuntimeError("zstd payload needs the zstandard package")
        return zstd.ZstdDecompressor().decompress(blob)
    if zstd is not None:
        try:
            return zstd.ZstdDecompressor().decompress(blob)
        except Exception:
            pass
    return zlib.decompress(blob)


def encode_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO RLE. pycocotools compressed counts if available, else Python runs."""
    binary = np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8))
    if binary.ndim != 2:
        raise ValueError(f"mask must be HxW, got {binary.shape}")
    height, width = int(binary.shape[0]), int(binary.shape[1])
    try:
        from pycocotools import mask as mask_util

        rle = mask_util.encode(binary)
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        return {"size": [height, width], "counts": counts}
    except Exception:
        return {"size": [height, width], "counts": _uncompressed_runs(binary)}


def decode_coco_rle(rle: dict[str, Any]) -> np.ndarray:
    size = rle["size"]
    height, width = int(size[0]), int(size[1])
    counts = rle["counts"]
    if isinstance(counts, (list, tuple)):
        return _from_uncompressed_runs(list(counts), height, width)
    if isinstance(counts, bytes):
        counts_bytes = counts
        counts_str = counts.decode("ascii")
    else:
        counts_str = str(counts)
        counts_bytes = counts_str.encode("ascii")
    try:
        from pycocotools import mask as mask_util

        decoded = mask_util.decode({"size": [height, width], "counts": counts_bytes})
        return (np.asarray(decoded) > 0).astype(np.uint8)
    except Exception:
        return _from_uncompressed_runs(_coco_string_to_runs(counts_str), height, width)


def _uncompressed_runs(mask: np.ndarray) -> list[int]:
    bits = np.asfortranarray((np.asarray(mask) > 0).astype(np.uint8)).ravel(order="F")
    if bits.size == 0:
        return []
    change_at = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    starts = np.concatenate(([0], change_at))
    ends = np.concatenate((change_at, [bits.size]))
    runs = (ends - starts).astype(int).tolist()
    if int(bits[0]) == 1:
        runs.insert(0, 0)
    return runs


def _from_uncompressed_runs(runs: list[int], height: int, width: int) -> np.ndarray:
    n = height * width
    if n == 0:
        return np.zeros((height, width), dtype=np.uint8)
    flat = np.zeros(n, dtype=np.uint8)
    pos = 0
    value = 0
    for run in runs:
        length = int(run)
        if length < 0:
            raise ValueError("negative RLE run")
        end = pos + length
        if end > n:
            raise ValueError("RLE overflow")
        if value:
            flat[pos:end] = 1
        pos = end
        value = 1 - value
    if pos != n:
        raise ValueError("RLE underfill")
    return np.asfortranarray(flat.reshape((height, width), order="F"))


def _coco_string_to_runs(s: str) -> list[int]:
    """Decode COCO compressed counts (maskApi.c rleFrString)."""
    runs: list[int] = []
    i = 0
    n = len(s)
    while i < n:
        x = 0
        k = 0
        more = 1
        while more and i < n:
            c = ord(s[i]) - 48
            i += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(int(x))
    return runs


def pack_rle_stream(
    frames: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    fps: float,
    classes: list[str],
) -> tuple[bytes, str]:
    """JSON COCO-RLE per frame, then zstd (zlib fallback). Empty instances = skip_frame."""
    doc = {
        "schema": MASK_STREAM_SCHEMA,
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
        "classes": list(classes),
        "mask_empty_policy": "skip_frame",
        "frames": frames,
    }
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    return compress_bytes(raw)


def unpack_rle_stream(blob: bytes) -> dict[str, Any]:
    doc = json.loads(decompress_bytes(blob).decode("utf-8"))
    if doc.get("schema") != MASK_STREAM_SCHEMA:
        raise ValueError(f"unknown mask stream schema {doc.get('schema')!r}")
    return doc


def union_mask_from_frame(frame_rec: dict[str, Any], height: int, width: int) -> np.ndarray:
    """OR of instance masks. Empty instances stay all-False (never whole-frame True)."""
    out = np.zeros((height, width), dtype=np.uint8)
    for inst in frame_rec.get("instances") or []:
        rle = inst.get("rle")
        if not rle:
            continue
        piece = decode_coco_rle(rle)
        if piece.shape != (height, width):
            raise ValueError(f"instance mask shape {piece.shape} != {(height, width)}")
        out |= (piece > 0).astype(np.uint8)
    return out


def instance_record(
    mask: np.ndarray,
    *,
    class_id: int,
    class_name: str,
    score: float = 1.0,
    bbox: list[float] | None = None,
) -> dict[str, Any] | None:
    """Skip empty masks. Never synthesizes a whole-frame True fill."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask must be HxW, got {binary.shape}")
    if not binary.any():
        return None
    if bbox is None or len(bbox) < 4:
        ys, xs = np.nonzero(binary)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "score": float(score),
        "bbox": [float(x) for x in bbox],
        "rle": encode_coco_rle(binary),
    }


def frame_record(index: int, instances: list[dict[str, Any]]) -> dict[str, Any]:
    return {"index": int(index), "instances": list(instances)}


def payload_filename(codec: str) -> str:
    """Native payload name. Preview overlays must not use this."""
    if codec == "zstd":
        return "masks.rle.zst"
    return "payload.bin"
