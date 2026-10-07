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
import resource
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


#: ``--lp`` is a level of parallelism (0-6), not a thread count; level 6 lets the
#: encoder size its pools freely. The threads it may use are fixed by CPU
#: affinity instead (`run` with ``cpus``).
PARALLELISM_LEVEL = 6


def encode_command(
    encoder: str, source: Path, stream: Path, *, width: int, height: int, fps: Fraction, frames: int,
    crf: float, preset: int, full_range: bool, extra: list[str] | None = None,
) -> list[str]:
    """One-pass CRF, random access, one keyframe for the whole clip (``--keyint -1``)."""
    return [
        encoder, "-i", str(source), "-b", str(stream),
        "-w", str(width), "-h", str(height), "--input-depth", "8",
        "--fps-num", str(fps.numerator), "--fps-denom", str(fps.denominator), "-n", str(frames),
        "--color-range", "1" if full_range else "0",
        "--rc", "0", "--crf", f"{crf:g}", "--preset", str(preset), "--keyint", "-1",
        "--lp", str(PARALLELISM_LEVEL), "--progress", "0", *(extra or []),
    ]


def decode_command(decoder: str, stream: Path, output: Path, *, threads: int) -> list[str]:
    return [decoder, "-i", str(stream), "-o", str(output), "--threads", str(threads), "--quiet"]


def run(command: list[str], timeout: float, cpus: list[int] | None = None) -> dict[str, float]:
    """Run a tool, optionally confined to ``cpus``; wall seconds and the CPU seconds it used."""
    def confine() -> None:
        if cpus:
            os.sched_setaffinity(0, cpus)  # type: ignore[attr-defined,unused-ignore]

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, preexec_fn=confine if cpus else None)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if done.returncode:
        raise RuntimeError(f"{command[0]} failed ({done.returncode}):\n{done.stderr[-4000:]}")
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return {"wall": wall, "cpu": cpu}


def code(
    source: Path, work: Path, *, width: int, height: int, fps: Fraction, frames: int, crf: float,
    preset: int, threads: int, full_range: bool, encoder: str, decoder: str, timeout: float = 7200,
    cpus: list[int] | None = None, extra: list[str] | None = None,
) -> dict[str, Any]:
    """Encode ``source`` and decode it again; returns the stream record and the decoded file."""
    stream, decoded = work / "stream.ivf", work / "decoded.yuv"
    encode = encode_command(encoder, source, stream, width=width, height=height, fps=fps, frames=frames,
                            crf=crf, preset=preset, full_range=full_range, extra=extra)
    load = os.getloadavg()
    encoded = run(encode, timeout, cpus)
    decode = decode_command(decoder, stream, decoded, threads=threads)
    decoded_time = run(decode, timeout, cpus)
    data = stream.read_bytes()
    sizes = ivf_frames(data)
    frame_bytes = width * height * 3 // 2
    return {
        "stream": str(stream), "decoded": str(decoded),
        "stream_sha256": hashlib.sha256(data).hexdigest(), "file_bytes": len(data),
        "payload_bytes": sum(sizes), "temporal_units": len(sizes),
        "decoded_frames": decoded.stat().st_size / frame_bytes,
        "encode_command": encode, "decode_command": decode,
        "encode_seconds": round(encoded["wall"], 3), "encode_cpu_seconds": round(encoded["cpu"], 3),
        "decode_seconds": round(decoded_time["wall"], 3), "decode_cpu_seconds": round(decoded_time["cpu"], 3),
        "cpus": cpus, "host_load_before": [round(v, 2) for v in load],
    }
