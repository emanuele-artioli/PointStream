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


def _run_ffmpeg(argv: list[str], stdin_bytes: bytes | None) -> bytes:
    timeout = float(os.environ.get("PS_CODEC_TIMEOUT_SECONDS", "0")) or None
    result = subprocess.run(argv, input=stdin_bytes, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail[:400]}")
    return result.stdout


def _rgb_dump(ffmpeg_path: str, video_path: Path, height: int, width: int, count: int) -> np.ndarray:
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
    decoded = np.frombuffer(raw, dtype=np.uint8)
    usable = (decoded.size // (height * width * 3)) * height * width * 3
    decoded = decoded[:usable].reshape(-1, height, width, 3)
    if decoded.shape[0] < count:
        pad = np.repeat(decoded[-1:], count - decoded.shape[0], axis=0)
        decoded = np.concatenate([decoded, pad], axis=0)
    return decoded[:count]


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
    )
