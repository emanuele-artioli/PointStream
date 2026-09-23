"""Reproduce FFmpeg/libvvenc empty outputs against direct vvencapp.

This is a diagnostic for the installed tools, not a new rate point. The input
is the saved 48-frame C1 error video, so the exact source is stable.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.components.codec.tools import resolve_ffmpeg, resolve_vvenc
from src.components.codec.y4m import parse_header

SOURCE = Path(
    "/home/itec/emanuele/pointstream-data/outputs/video-codec-probe/"
    "federer007/residual_full/vvc/qp36/source.y4m"
)
OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/video-codec-probe/"
    "federer007/vvc-wrapper-probe.json"
)
QPS = (32, 36, 40, 44, 46)


def _size(path: Path) -> int:
    return int(path.stat().st_size) if path.is_file() else 0


def _run_wrapper(ffmpeg: str, source: Path, dest: Path, qp: int) -> dict:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libvvenc",
        "-preset",
        "faster",
        "-qp",
        str(qp),
        "-qpa",
        "0",
        str(dest),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "path": ffmpeg,
        "command": command,
        "returncode": result.returncode,
        "bytes": _size(dest),
        "seconds": round(time.perf_counter() - started, 3),
        "stderr": (result.stderr or "").strip()[-500:],
    }


def _run_direct(vvenc: str, source: Path, dest: Path, qp: int) -> dict:
    width, height, fps = parse_header(source)
    command = [
        vvenc,
        "--input",
        str(source),
        "--size",
        f"{width}x{height}",
        "--framerate",
        str(max(1, int(round(fps)))),
        "--format",
        "yuv420",
        "--preset",
        "faster",
        "--qp",
        str(qp),
        "--qpa",
        "0",
        "--output",
        str(dest),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "path": vvenc,
        "command": command,
        "returncode": result.returncode,
        "bytes": _size(dest),
        "seconds": round(time.perf_counter() - started, 3),
        "stderr": (result.stderr or "").strip()[-500:],
    }


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    ffmpeg = resolve_ffmpeg()
    vvenc = resolve_vvenc()
    rows = []
    for qp in QPS:
        wrapper_dest = OUT.parent / f"wrapper_qp{qp}.vvc"
        direct_dest = OUT.parent / f"direct_qp{qp}.vvc"
        wrapper_dest.unlink(missing_ok=True)
        direct_dest.unlink(missing_ok=True)
        wrapper = _run_wrapper(ffmpeg.path, SOURCE, wrapper_dest, qp)
        direct = _run_direct(vvenc.path, SOURCE, direct_dest, qp)
        row = {"qp": qp, "ffmpeg_libvvenc": wrapper, "vvencapp": direct}
        rows.append(row)
        print(row, flush=True)
    report = {
        "source": str(SOURCE),
        "source_header": parse_header(SOURCE),
        "ffmpeg": {"path": ffmpeg.path, "version": ffmpeg.version},
        "vvencapp": {"path": vvenc.path, "version": vvenc.version},
        "rows": rows,
        "finding": (
            "On this source, FFmpeg/libvvenc can return exit 0 with zero bytes "
            "at some QPs while direct vvencapp emits a non-empty stream. Empty "
            "files are invalid measurements and must be rejected or recovered."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
