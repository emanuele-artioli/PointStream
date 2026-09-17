"""Transcode bitrate-matched AV1 ladder rungs to 1080p H.264 for the public inspector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLIPS = {
    "clip_02_factory001_worker001_00002": "clip_02",
    "clip_03_factory001_worker001_00000": "clip_03",
}

# Inspector keys → (scale, optional target_kbps)
AV1_EXPORTS = {
    "av1_180": ("320:180", 25),
    "av1_240": ("426:240", 40),
    "av1_360": ("640:360", 80),
    "av1_540": ("960:540", 250),
    "av1_720": ("1280:720", 300),
    "av1_1080": (None, 500),  # matched by name + target
}


def _clip(results: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in results["clips"] if c["clip_name"] == name)


def _src(clip: dict[str, Any], key: str) -> Path:
    scale, tgt = AV1_EXPORTS[key]
    for arm in clip["av1_arms"]:
        if key == "av1_1080":
            if "1080p" in arm["name"] and arm["target_kbps"] == tgt:
                return Path(arm["video_path"])
        elif arm.get("scale") == scale and arm.get("target_kbps") == tgt:
            return Path(arm["video_path"])
    raise FileNotFoundError(key)


def transcode(src: Path, dst: Path, ffmpeg: str) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("demo/outputs/results"))
    parser.add_argument("--pitch-dir", type=Path, default=Path("demo/outputs/pitch"))
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    args = parser.parse_args()
    results = json.loads((args.results_dir / "comparison_results.json").read_text())
    for long_name, short in CLIPS.items():
        clip = _clip(results, long_name)
        for key in AV1_EXPORTS:
            transcode(_src(clip, key), args.pitch_dir / f"web_{short}_{key}.mp4", args.ffmpeg)


if __name__ == "__main__":
    main()
