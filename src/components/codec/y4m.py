"""Minimal 4:2:0 y4m read/write, for pixel-domain ROI and synthetic sources.

Luma is the plane a QP change moves; chroma is carried through unchanged so a
region arm cannot be accused of also having resampled colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Y4M:
    """Decoded 4:2:0 y4m: luma plus copied chroma."""

    width: int
    height: int
    fps: float
    luma: np.ndarray  # (frames, height, width) uint8
    chroma: np.ndarray | None = None  # (frames, 2, h/2, w/2) uint8, or None

    @property
    def frames(self) -> int:
        return int(self.luma.shape[0])


def parse_header(path: Path) -> tuple[int, int, float]:
    """Width, height, fps from a y4m header. No frame data is read."""
    with path.open("rb") as handle:
        header = handle.readline()
    return _parse_header_line(header)


def read(path: Path) -> Y4M:
    """Read a 4:2:0 y4m into luma (and chroma, when present)."""
    data = path.read_bytes()
    header_end = data.index(b"\n")
    width, height, fps = _parse_header_line(data[: header_end + 1])

    luma_size = width * height
    chroma_w, chroma_h = width // 2, height // 2
    chroma_size = chroma_w * chroma_h * 2
    frame_size = luma_size + chroma_size

    lumas: list[np.ndarray] = []
    chromas: list[np.ndarray] = []
    offset = header_end + 1
    marker = b"FRAME"
    while offset < len(data):
        if not data.startswith(marker, offset):
            break
        offset = data.index(b"\n", offset) + 1
        plane = np.frombuffer(data, dtype=np.uint8, count=luma_size, offset=offset)
        lumas.append(plane.reshape(height, width).copy())
        chroma_off = offset + luma_size
        uv = np.frombuffer(data, dtype=np.uint8, count=chroma_size, offset=chroma_off)
        chromas.append(uv.reshape(2, chroma_h, chroma_w).copy())
        offset += frame_size

    if not lumas:
        raise ValueError(f"{path} contained no frames")
    return Y4M(
        width=width,
        height=height,
        fps=fps,
        luma=np.stack(lumas),
        chroma=np.stack(chromas),
    )


def write(path: Path, video: Y4M) -> None:
    """Write a 4:2:0 y4m. Missing chroma is filled with 128 (neutral)."""
    header = _header_line(video.width, video.height, video.fps)
    chroma_w, chroma_h = video.width // 2, video.height // 2
    chunks = [header]
    for index in range(video.frames):
        chunks.append(b"FRAME\n")
        chunks.append(np.ascontiguousarray(video.luma[index], dtype=np.uint8).tobytes())
        if video.chroma is None:
            chunks.append(b"\x80" * (chroma_w * chroma_h * 2))
        else:
            chunks.append(np.ascontiguousarray(video.chroma[index], dtype=np.uint8).tobytes())
    path.write_bytes(b"".join(chunks))


def from_luma(luma: np.ndarray, *, fps: float = 10.0) -> Y4M:
    """Build a y4m from ``(frames, height, width)`` uint8 luma, neutral chroma."""
    if luma.ndim != 3:
        raise ValueError(f"luma must be (frames, height, width), got {luma.shape}")
    frames, height, width = luma.shape
    if width % 2 or height % 2:
        raise ValueError(f"y4m 4:2:0 needs even size, got {width}x{height}")
    chroma = np.full((frames, 2, height // 2, width // 2), 128, dtype=np.uint8)
    return Y4M(width=width, height=height, fps=fps, luma=np.ascontiguousarray(luma, dtype=np.uint8), chroma=chroma)


def _parse_header_line(header: bytes) -> tuple[int, int, float]:
    text = header.decode("ascii", errors="replace").strip()
    if not text.startswith("YUV4MPEG2"):
        raise ValueError(f"not a y4m header: {text!r}")
    width = height = 0
    fps = 30.0
    for token in text.split():
        if token.startswith("W"):
            width = int(token[1:])
        elif token.startswith("H"):
            height = int(token[1:])
        elif token.startswith("F") and ":" in token:
            num, den = token[1:].split(":", 1)
            denom = float(den) or 1.0
            fps = float(num) / denom
        elif token.startswith("F"):
            fps = float(token[1:])
    if not width or not height:
        raise ValueError(f"y4m header has no frame size: {text!r}")
    return width, height, fps


def _header_line(width: int, height: int, fps: float) -> bytes:
    # fps as an integer ratio keeps ffmpeg and SvtAv1EncApp happy.
    num = int(round(fps * 1000))
    return f"YUV4MPEG2 W{width} H{height} F{num}:1000 Ip A0:0 C420mpeg2\n".encode("ascii")
