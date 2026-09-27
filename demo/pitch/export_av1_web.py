"""Optional H.264 wrapper for non-AV1 streams.

AV1 CRF 63 ladder files are served directly by the inspector — do not wrap
those. This helper remains for PointStream composites and the reference clip,
which are still H.264 for canvas playback.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def transcode(src: Path, dst: Path, ffmpeg: str) -> None:
    """Scale to 1080p H.264 CRF 23. Not used for AV1 demo streams."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-an",
        "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1",
        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(tmp),
    ]
    logger.info("%s -> %s", src.name, dst.name)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.replace(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="H.264 wrapper for PS/ref only (not AV1)")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    args = parser.parse_args()
    transcode(args.src, args.dst, args.ffmpeg)


if __name__ == "__main__":
    main()
