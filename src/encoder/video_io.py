from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DecodedVideo:
    tensor: torch.Tensor
    fps: float
    num_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    num_frames: int


def probe_video_metadata(video_path: str | Path) -> VideoMetadata:
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    width, height, fps, num_frames = _probe_video_with_ffprobe(source)
    if num_frames <= 0:
        # Fallback count via streaming decode if container metadata is incomplete.
        num_frames = sum(1 for _ in iter_video_frames_ffmpeg(source, width=width, height=height))

    return VideoMetadata(
        fps=float(fps),
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
    )


def iter_video_frames_ffmpeg(
    video_path: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> Iterator[np.ndarray]:
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    if width is None or height is None:
        width, height, _fps, _num_frames = _probe_video_with_ffprobe(source)

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )

    frame_size = int(width * height * 3)
    yielded_frames = 0

    try:
        if process.stdout is None:
            raise RuntimeError("FFmpeg stdout pipe is not available")

        while True:
            raw_frame = _read_exact(process.stdout, frame_size)
            if not raw_frame:
                break
            if len(raw_frame) < frame_size:
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3).copy()
            yielded_frames += 1
            yield frame
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr_output = b""
        if process.stderr is not None:
            stderr_output = process.stderr.read()
            process.stderr.close()

        return_code = process.wait()
        if yielded_frames == 0:
            stderr_text = stderr_output.decode("utf-8", errors="replace").strip()
            raise ValueError(
                f"FFmpeg produced no decodable frames for '{source}': {stderr_text or 'unknown error'}"
            )

        # Treat warnings/non-zero return as non-fatal if valid frame bytes were produced.
        _ = return_code


def decode_video_to_tensor(video_path: str | Path) -> DecodedVideo:
    metadata = probe_video_metadata(video_path)
    frames = list(
        iter_video_frames_ffmpeg(
            video_path,
            width=metadata.width,
            height=metadata.height,
        )
    )
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")

    frame_array = np.stack(frames, axis=0)
    # Shape: [Frames, Channels, Height, Width] where channels are BGR.
    tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0

    return DecodedVideo(
        tensor=tensor,
        fps=metadata.fps,
        num_frames=int(tensor.shape[0]),
        width=metadata.width,
        height=metadata.height,
    )


def _probe_video_with_ffprobe(video_path: Path) -> tuple[int, int, float, int]:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    process = subprocess.run(
        probe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if process.returncode != 0:
        stderr_text = (process.stderr or "").strip()
        raise ValueError(
            f"ffprobe failed for '{video_path}': {stderr_text or 'unknown error'}"
        )

    payload: dict[str, Any] = json.loads(process.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"ffprobe returned no video streams for: {video_path}")

    stream = streams[0]
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    fps = _parse_ffprobe_fps(stream.get("avg_frame_rate", "0/1"))
    num_frames = _parse_optional_int(stream.get("nb_frames"))
    if num_frames <= 0:
        duration = _parse_optional_float(stream.get("duration"))
        if duration > 0 and fps > 0:
            num_frames = int(round(duration * fps))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid width/height from ffprobe for: {video_path}")
    if fps <= 0:
        fps = 30.0

    return width, height, fps, num_frames


def _parse_ffprobe_fps(value: str) -> float:
    if "/" in value:
        num_str, den_str = value.split("/", maxsplit=1)
        num = float(num_str or 0.0)
        den = float(den_str or 1.0)
        if den == 0:
            return 0.0
        return num / den

    return float(value)


def _parse_optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_optional_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_exact(stream: Any, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)
