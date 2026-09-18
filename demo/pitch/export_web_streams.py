"""Transcode experiment reconstructions into Safari/Chrome-safe 1080p H.264.

The live inspector used a 1920x3240 stacked MP4 with the moov atom at the
end. Safari, iOS, and many Chrome hardware decoders refuse that file; Firefox
often software-decodes it. These exports are 1920x1080, High@L4.1, yuv420p,
faststart.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("demo/outputs/results")
DEFAULT_PITCH_DIR = Path("demo/outputs/pitch")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

PITCH_AV1_SCALE = {
    "clip_01_factory001_worker001_00001": "320:180",
    "clip_02_factory001_worker001_00002": "320:180",
    "clip_03_factory001_worker001_00000": "426:240",
}

CLIP_SHORT = {
    "clip_01_factory001_worker001_00001": "clip_01",
    "clip_02_factory001_worker001_00002": "clip_02",
    "clip_03_factory001_worker001_00000": "clip_03",
}

PS_KEYS = [
    ("ps_starve", "Extreme Starve", "ps_rec_ps_extreme_starve.mp4"),
    ("ps_heavy", "Heavy Starve", "ps_rec_ps_heavy_starve.mp4"),
    ("ps_low", "Low Teleop", "ps_rec_ps_low_teleop.mp4"),
    ("ps_std", "PS Standard", "ps_rec_ps_standard.mp4"),
    ("ps_1080", "Standard 1080p", "ps_rec_ps_standard_1080p.mp4"),
]


def _clip_record(results_data: dict[str, Any], clip_name: str) -> dict[str, Any] | None:
    for clip in results_data.get("clips", []):
        if clip.get("clip_name") == clip_name:
            return clip
    return None


def _av1_arm(clip: dict[str, Any], scale: str) -> dict[str, Any] | None:
    for arm in clip.get("av1_arms", []):
        if arm.get("scale") == scale:
            return arm
    return None


def transcode_1080(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        os.environ.get("FFMPEG", "ffmpeg"), "-y", "-i", str(src),
        "-an",
        "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.1",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    logger.info("transcode %s -> %s", src.name, dst.name)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.replace(dst)


def hstack_report(left: Path, right: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    cmd = [
        os.environ.get("FFMPEG", "ffmpeg"), "-y",
        "-i", str(left),
        "-i", str(right),
        "-filter_complex",
        "[0:v]scale=960:1080:flags=lanczos,format=yuv420p[l];"
        "[1:v]scale=960:1080:flags=lanczos,format=yuv420p[r];"
        "[l][r]hstack=inputs=2",
        "-an",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.1",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    logger.info("hstack report %s", dst.name)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.replace(dst)


def export_clip(clip_name: str, results_dir: Path, pitch_dir: Path, results_data: dict[str, Any]) -> None:
    short = CLIP_SHORT[clip_name]
    clip_dir = results_dir / clip_name
    clip = _clip_record(results_data, clip_name) or {}
    scale = PITCH_AV1_SCALE[clip_name]
    av1 = _av1_arm(clip, scale)
    if av1 and av1.get("video_path"):
        av1_src = Path(av1["video_path"])
    else:
        token = "320x180" if scale == "320:180" else "426x240"
        av1_src = next((clip_dir / "av1_ladder").glob(f"*{token}*.mp4"))

    jobs = [
        ("ref", clip_dir / "reference_trimmed.mp4"),
        ("av1", av1_src),
    ]
    for key, _needle, filename in PS_KEYS:
        jobs.append((key, clip_dir / filename))

    written: dict[str, Path] = {}
    for key, src in jobs:
        if not src.exists():
            raise FileNotFoundError(src)
        dst = pitch_dir / f"web_{short}_{key}.mp4"
        transcode_1080(src, dst)
        written[key] = dst

    hstack_report(
        written["ps_starve"],
        written["av1"],
        pitch_dir / f"side_by_side_demo_{clip_name}.mp4",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pitch-dir", type=Path, default=DEFAULT_PITCH_DIR)
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    args = parser.parse_args()
    os.environ["FFMPEG"] = str(args.ffmpeg)

    results_json = args.results_dir / "comparison_results.json"
    results_data = json.loads(results_json.read_text(encoding="utf-8"))
    args.pitch_dir.mkdir(parents=True, exist_ok=True)
    for clip_name in CLIP_SHORT:
        export_clip(clip_name, args.results_dir, args.pitch_dir, results_data)
        logger.info("exported %s", clip_name)


if __name__ == "__main__":
    main()
