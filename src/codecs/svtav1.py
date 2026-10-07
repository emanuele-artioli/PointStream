"""SVT-AV1 encode and dav1d decode of raw 8-bit 4:2:0 video, with exact rate accounting.

The encoder is ``SvtAv1EncApp`` and the decoder ``dav1d``, the native tools of
the environment (conda-forge ``svt-av1`` 4.2.0, ``dav1d`` 1.5; docs/resources.md).
Input and output are raw planar files (Y, then U, then V per frame), so no
colour conversion happens inside the codec path. The stream is IVF; its rate is
the AV1 payload (every temporal unit's OBUs), and the IVF framing is recorded
apart.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

IVF_HEADER = 32
IVF_FRAME_HEADER = 12


def tool(name: str) -> dict[str, Any]:
    """Path, resolved path, sha256 and version line of a native tool on PATH (or an absolute path)."""
    path = name if os.path.isabs(name) else shutil.which(name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{name} is not on PATH")
    flag = "-version" if Path(name).name == "ffmpeg" else "--version"
    out = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
    first = (out.stdout or out.stderr).strip().splitlines()[0]
    real = os.path.realpath(path)
    digest = hashlib.sha256(Path(real).read_bytes()).hexdigest()
    return {"path": path, "real_path": real, "sha256": digest, "version": first}


def ivf_frames(data: bytes) -> list[int]:
    """Payload size of every IVF frame (temporal unit)."""
    if len(data) < IVF_HEADER or data[:4] != b"DKIF":
        raise ValueError("not an IVF file")
    header_size = struct.unpack("<H", data[6:8])[0]
    sizes, offset = [], header_size
    while offset < len(data):
        if offset + IVF_FRAME_HEADER > len(data):
            raise ValueError("truncated IVF frame header")
        size = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += IVF_FRAME_HEADER + size
        if offset > len(data):
            raise ValueError("truncated IVF frame")
        sizes.append(size)
    return sizes


def encode_command(
    encoder: str, source: Path, stream: Path, *, width: int, height: int, fps: Fraction, frames: int,
    crf: float, preset: int, threads: int, full_range: bool,
) -> list[str]:
    """One-pass CRF, random access, one keyframe for the whole clip (``--keyint -1``)."""
    return [
        encoder, "-i", str(source), "-b", str(stream),
        "-w", str(width), "-h", str(height), "--input-depth", "8",
        "--fps-num", str(fps.numerator), "--fps-denom", str(fps.denominator), "-n", str(frames),
        "--color-range", "1" if full_range else "0",
        "--rc", "0", "--crf", f"{crf:g}", "--preset", str(preset), "--keyint", "-1",
        "--lp", str(threads), "--progress", "0",
    ]


def decode_command(decoder: str, stream: Path, output: Path, *, threads: int) -> list[str]:
    return [decoder, "-i", str(stream), "-o", str(output), "--threads", str(threads), "--quiet"]


def run(command: list[str], timeout: float) -> float:
    started = time.time()
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"{command[0]} failed ({done.returncode}):\n{done.stderr[-4000:]}")
    return time.time() - started


def code(
    source: Path, work: Path, *, width: int, height: int, fps: Fraction, frames: int, crf: float,
    preset: int, threads: int, full_range: bool, encoder: str, decoder: str, timeout: float = 3600,
) -> dict[str, Any]:
    """Encode ``source`` and decode it again; returns the stream record and the decoded file."""
    stream, decoded = work / "stream.ivf", work / "decoded.yuv"
    encode = encode_command(encoder, source, stream, width=width, height=height, fps=fps, frames=frames,
                            crf=crf, preset=preset, threads=threads, full_range=full_range)
    encode_seconds = run(encode, timeout)
    decode = decode_command(decoder, stream, decoded, threads=threads)
    decode_seconds = run(decode, timeout)
    data = stream.read_bytes()
    sizes = ivf_frames(data)
    frame_bytes = width * height * 3 // 2
    return {
        "stream": str(stream), "decoded": str(decoded),
        "stream_sha256": hashlib.sha256(data).hexdigest(), "file_bytes": len(data),
        "payload_bytes": sum(sizes), "temporal_units": len(sizes),
        "decoded_frames": decoded.stat().st_size / frame_bytes,
        "encode_command": encode, "decode_command": decode,
        "encode_seconds": round(encode_seconds, 3), "decode_seconds": round(decode_seconds, 3),
    }
