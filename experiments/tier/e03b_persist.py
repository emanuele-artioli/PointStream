"""Bounded persistent wrapper around the existing encode/decode path.

``timed_roundtrip`` deletes its temporary charged bitstream. This wrapper uses
the same ``src.components.codec.encode.encode`` / ``decode`` path and keeps the
bitstream, decoded RGB, command, and byte length in ``work_dir``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from src.components.codec.encode import BITSTREAM_SUFFIX, EncodeRecord, decode, encode
from src.components.codec.frames import even_size
from src.components.codec.measure import TimedRoundtrip
from src.components.codec import tools
from src.contracts.codecs import EncodeRequest


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class DecodeCountError(ValueError):
    """Raised when a decode is empty, partial, short, extra, or the wrong size."""


@dataclass(frozen=True)
class PersistentRoundtrip:
    """One charged encode/decode whose artifacts remain on disk."""

    trip: TimedRoundtrip
    bitstream_path: Path
    bitstream_sha256: str
    decoded_npy: Path
    encode_record: dict[str, Any]
    lossless_path: Path
    standalone_decode_path: Path
    standalone_shape: tuple[int, ...]
    ledger_matched: bool
    decode_geometry: dict[str, Any]
    standalone_pixels_match: bool


def _run_ffmpeg(argv: list[str], stdin_bytes: bytes | None) -> bytes:
    timeout = float(os.environ.get("PS_CODEC_TIMEOUT_SECONDS", "0")) or None
    result = subprocess.run(argv, input=stdin_bytes, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail[:400]}")
    return result.stdout


def probe_video_geometry(ffprobe: str, video_path: Path) -> tuple[int, int]:
    payload = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )
    streams = payload.get("streams") or []
    if not streams:
        raise DecodeCountError(f"{video_path}: no video stream")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    if width < 1 or height < 1:
        raise DecodeCountError(f"{video_path}: invalid geometry {width}x{height}")
    return width, height


def frames_from_rgb24(
    raw: bytes,
    *,
    width: int,
    height: int,
    expected_count: int,
    source: str,
) -> np.ndarray:
    """Reshape raw RGB24 only when the byte length is an exact expected frame count."""
    if width < 1 or height < 1:
        raise DecodeCountError(f"{source}: invalid geometry {width}x{height}")
    frame_bytes = int(height) * int(width) * 3
    if not raw:
        raise DecodeCountError(f"{source}: empty decode")
    if len(raw) % frame_bytes != 0:
        raise DecodeCountError(
            f"{source}: partial frame ({len(raw)} bytes, frame is {frame_bytes} bytes)"
        )
    count = len(raw) // frame_bytes
    if count != int(expected_count):
        raise DecodeCountError(f"{source}: decoded {count} frames, expected {expected_count}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3).copy()


def dump_decoded_rgb(
    ffmpeg_path: str,
    video_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dump RGB24 using the container's own width/height; never pad or trim."""
    ffprobe = str(Path(ffmpeg_path).with_name("ffprobe"))
    actual_width, actual_height = probe_video_geometry(ffprobe, video_path)
    raw = _run_ffmpeg(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        None,
    )
    if (actual_width, actual_height) != (int(expected_width), int(expected_height)):
        raise DecodeCountError(
            f"{video_path}: decoded {actual_width}x{actual_height}, "
            f"expected {expected_width}x{expected_height}"
        )
    frames = frames_from_rgb24(
        raw,
        width=actual_width,
        height=actual_height,
        expected_count=expected_count,
        source=str(video_path),
    )
    geometry = {
        "width": actual_width,
        "height": actual_height,
        "count": int(frames.shape[0]),
        "sha256": sha256_bytes(frames.tobytes()),
    }
    return frames, geometry


def _rgb_dump(ffmpeg_path: str, video_path: Path, height: int, width: int, count: int) -> np.ndarray:
    frames, _geometry = dump_decoded_rgb(
        ffmpeg_path,
        video_path,
        expected_width=width,
        expected_height=height,
        expected_count=count,
    )
    return frames


def _record_to_dict(record: EncodeRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["output"] = str(record.output)
    payload["command"] = list(record.command)
    payload["size_bytes"] = int(record.size_bytes)
    return payload


def persistent_timed_roundtrip(
    frames: np.ndarray,
    *,
    request: EncodeRequest,
    fps: float,
    work_dir: Path,
) -> PersistentRoundtrip:
    """Encode and decode RGB frames, leaving the charged bitstream in ``work_dir``."""
    clip = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if clip.ndim != 4 or clip.shape[3] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {tuple(clip.shape)}")
    clip = even_size(clip)
    count, height, width, _ = clip.shape
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = tools.resolve_ffmpeg()

    lossless = work_dir / "payload.mkv"
    _run_ffmpeg(
        [
            ffmpeg.path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "ffv1",
            str(lossless),
        ],
        clip.tobytes(),
    )
    dest = work_dir / f"payload{BITSTREAM_SUFFIX[request.codec_name]}"
    record = encode(lossless, dest, request, work_dir=work_dir)
    file_bytes = dest.stat().st_size
    ledger_matched = int(record.size_bytes) == int(file_bytes) and file_bytes > 0

    back = work_dir / "decoded.mkv"
    decode_started = time.perf_counter()
    decode(dest, back, request)
    decoded = _rgb_dump(ffmpeg.path, back, height, width, count)
    decode_seconds = time.perf_counter() - decode_started

    standalone = work_dir / "decoded_standalone.mkv"
    decode(dest, standalone, request)
    standalone_frames = _rgb_dump(ffmpeg.path, standalone, height, width, count)
    if not np.array_equal(decoded, standalone_frames):
        raise DecodeCountError(f"{dest}: ordinary and standalone decodes differ")
    decode_geometry = {
        "width": int(decoded.shape[2]),
        "height": int(decoded.shape[1]),
        "count": int(decoded.shape[0]),
        "sha256": sha256_bytes(np.ascontiguousarray(decoded).tobytes()),
    }

    decoded_npy = work_dir / "decoded_rgb.npy"
    np.save(decoded_npy, decoded)
    encode_record = _record_to_dict(record)
    encode_record["bitstream_sha256"] = sha256_path(dest)
    encode_record["encoder_sha256"] = sha256_path(Path(record.tool_path))
    encode_record["ffmpeg_sha256"] = sha256_path(Path(record.ffmpeg_path))
    (work_dir / "encode_record.json").write_text(
        json.dumps(encode_record, indent=2) + "\n", encoding="utf-8"
    )

    trip = TimedRoundtrip(
        size_bytes=int(record.size_bytes),
        frames=decoded,
        encode_seconds=float(record.encode_seconds),
        decode_seconds=float(decode_seconds),
        tool_path=record.tool_path,
        tool_version=record.tool_version,
        preset=record.preset,
        qp=record.rate,
    )
    return PersistentRoundtrip(
        trip=trip,
        bitstream_path=dest,
        bitstream_sha256=str(encode_record["bitstream_sha256"]),
        decoded_npy=decoded_npy,
        encode_record=encode_record,
        lossless_path=lossless,
        standalone_decode_path=standalone,
        standalone_shape=tuple(int(item) for item in standalone_frames.shape),
        ledger_matched=ledger_matched,
        decode_geometry=decode_geometry,
        standalone_pixels_match=True,
    )
