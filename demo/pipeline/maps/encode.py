"""Native map payloads: bit-packed zstd blobs and optional gray AV1.

Mask streams (COCO RLE) live in `src.segmentation.masks`.
"""

from __future__ import annotations

import subprocess
import zlib
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - env without zstd uses zlib
    zstd = None


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
