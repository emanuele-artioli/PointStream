"""Creates a high-impact side-by-side demonstration video with zoom inset and real-time HUD."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("demo/outputs/results")
DEFAULT_PITCH_DIR = Path("demo/outputs/pitch")


def create_side_by_side_video(
    ref_mp4: Path,
    av1_mp4: Path,
    ps_mp4: Path,
    output_mp4: Path,
    stats: dict[str, Any] | None = None,
    max_frames: int = 300,
) -> Path:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    from demo.pipeline.background_codec import read_video_frames_robust

    frames_ref = read_video_frames_robust(ref_mp4, max_frames=max_frames)
    frames_av1 = read_video_frames_robust(av1_mp4, max_frames=max_frames)
    frames_ps = read_video_frames_robust(ps_mp4, max_frames=max_frames)

    n_frames = min(len(frames_ref), len(frames_av1), len(frames_ps))
    if n_frames == 0:
        raise RuntimeError("No frames could be decoded for side-by-side video.")

    h, w = frames_ref[0].shape[:2]
    fps = 30.0

    # Vertically stacked layout: 3 full-resolution panels (1920 x 1080 each)
    # Total canvas: 1920 x 3240, keeping native resolution and aspect ratio intact
    canvas_w = w
    canvas_h = h * 3

    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_idx in range(n_frames):
        f_r = frames_ref[frame_idx]
        f_a = frames_av1[frame_idx]
        f_p = frames_ps[frame_idx]

        if f_a.shape[:2] != (h, w):
            f_a = cv2.resize(f_a, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if f_p.shape[:2] != (h, w):
            f_p = cv2.resize(f_p, (w, h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Panel 1: Original Reference (Top, y: 0 to h)
        canvas[0:h, 0:w] = f_r

        # Panel 2: AV1 at matched latency/rate (Middle, y: h to 2*h)
        canvas[h:2 * h, 0:w] = f_a

        # Panel 3: PointStream (Bottom, y: 2*h to 3*h)
        canvas[2 * h:3 * h, 0:w] = f_p

        # Titles and HUD banners for each panel
        av1_label = stats.get("av1_label", "2. AV1 240p p7 (STARVED)") if stats else "2. AV1 240p p7 (STARVED)"
        ps_label = stats.get("ps_label", "3. POINTSTREAM EXTREME STARVE") if stats else "3. POINTSTREAM EXTREME STARVE"

        # Overlay Banner 1
        cv2.rectangle(canvas, (0, 0), (w, 55), (15, 23, 42), -1)
        cv2.putText(canvas, "1. ORIGINAL REFERENCE (1080p Native | 4,200 kbps HEVC)", (25, 38), font, 0.9, (255, 255, 255), 2)

        # Overlay Banner 2
        cv2.rectangle(canvas, (0, h), (w, h + 55), (15, 23, 42), -1)
        cv2.putText(canvas, av1_label, (25, h + 38), font, 0.9, (0, 165, 255), 2)

        # Overlay Banner 3
        cv2.rectangle(canvas, (0, 2 * h), (w, 2 * h + 55), (15, 23, 42), -1)
        cv2.putText(canvas, ps_label, (25, 2 * h + 38), font, 0.9, (50, 255, 120), 2)

        # Dividing lines between panels
        cv2.line(canvas, (0, h), (w, h), (70, 80, 95), 4)
        cv2.line(canvas, (0, 2 * h), (w, 2 * h), (70, 80, 95), 4)

        writer.write(canvas)

    writer.release()

    # Re-encode to universal H.264 (yuv420p) for seamless web/HTML5 playback
    temp_raw = output_mp4.with_name(f"raw_{output_mp4.name}")
    output_mp4.rename(temp_raw)
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_raw),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_raw.exists():
        temp_raw.unlink()

    logger.info(f"Vertically stacked demo video saved and encoded (H.264 yuv420p 1920x3240) to {output_mp4} ({output_mp4.stat().st_size} bytes)")
    return output_mp4


def _clip_record(results_data: dict[str, Any] | None, clip_name: str) -> dict[str, Any] | None:
    if not results_data:
        return None
    for clip in results_data.get("clips", []):
        if clip.get("clip_name") == clip_name:
            return clip
    return None


def _ps_variant(clip: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for variant in clip.get("pointstream_variants", []):
        if needle.lower() in variant.get("name", "").lower():
            return variant
    return None


def _av1_arm(clip: dict[str, Any], scale: str) -> dict[str, Any] | None:
    for arm in clip.get("av1_arms", []):
        if arm.get("scale") == scale:
            return arm
    return None


def process_clip(
    clip_name: str,
    results_dir: Path,
    pitch_dir: Path,
    results_data: dict[str, Any] | None = None,
    ps_needle: str = "Extreme Starve",
    av1_scale: str = "426:240",
) -> Path:
    """Stack Reference / starved AV1 / PointStream Extreme Starve for the live inspector."""
    clip_dir = results_dir / clip_name
    ref_mp4 = clip_dir / "reference_trimmed.mp4"
    clip = _clip_record(results_data, clip_name)

    ps_info = _ps_variant(clip, ps_needle) if clip else None
    av1_info = _av1_arm(clip, av1_scale) if clip else None

    if ps_info and ps_info.get("video_path"):
        ps_mp4 = Path(ps_info["video_path"])
    else:
        ps_mp4 = clip_dir / "ps_rec_ps_extreme_starve.mp4"

    if av1_info and av1_info.get("video_path"):
        av1_mp4 = Path(av1_info["video_path"])
    else:
        ladder_dir = clip_dir / "av1_ladder"
        candidates = list(ladder_dir.glob("*426x240*.mp4")) or list(ladder_dir.glob("*240p*.mp4"))
        if not candidates:
            raise FileNotFoundError(f"No AV1 240p videos found in {ladder_dir}")
        av1_mp4 = candidates[0]

    missing = [str(p) for p in (ref_mp4, av1_mp4, ps_mp4) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Starved reconstructions are local experiment outputs (gitignored). "
            "Run this on the machine that produced comparison_results.json. Missing: "
            + ", ".join(missing)
        )

    logger.info(f"Clip {clip_name}: AV1={av1_mp4.name} PS={ps_mp4.name}")

    ps_lat = 18.4
    if results_data:
        ps_lat = results_data.get("latency_profile", {}).get("parallel_end_to_end_latency_ms", ps_lat)

    stats: dict[str, Any] = {
        "av1_label": "2. AV1 240p p7 (STARVED)",
        "ps_label": "3. POINTSTREAM EXTREME STARVE (240p bg)",
    }
    if av1_info:
        a_kbps = av1_info.get("actual_kbps", 0.0)
        a_det = av1_info.get("teleop_utility", {}).get("detection_rate", 0.0) * 100.0
        a_err = av1_info.get("teleop_utility", {}).get("mpjpe_pixels", 0.0)
        stats["av1_label"] = (
            f"2. AV1 240p p7 (Starved: {a_kbps:.0f} kbps | Det: {a_det:.1f}% | Joint Err: {a_err:.0f}px)"
        )
    if ps_info:
        ps_kbps = ps_info.get("bitrate_kbps", 0.0)
        ps_det = ps_info.get("teleop_utility", {}).get("detection_rate", 0.0) * 100.0
        ps_err = ps_info.get("teleop_utility", {}).get("mpjpe_pixels", 0.0)
        stats["ps_label"] = (
            f"3. POINTSTREAM EXTREME STARVE ({ps_kbps:.0f} kbps | {ps_lat:.1f} ms | "
            f"Det: {ps_det:.1f}% | Joint Err: {ps_err:.0f}px)"
        )

    out_video = pitch_dir / f"side_by_side_demo_{clip_name}.mp4"
    return create_side_by_side_video(ref_mp4, av1_mp4, ps_mp4, out_video, stats=stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack Reference / AV1 240p / PointStream Extreme Starve for the public demo"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pitch-dir", type=Path, default=DEFAULT_PITCH_DIR)
    parser.add_argument("--clip-name", default="clip_01_factory001_worker001_00001")
    parser.add_argument("--ps-needle", default="Extreme Starve")
    parser.add_argument("--av1-scale", default="426:240")
    parser.add_argument("--all-clips", action="store_true", help="Process all available clips")
    args = parser.parse_args()

    results_json = args.results_dir / "comparison_results.json"
    results_data = None
    if results_json.exists():
        with open(results_json, "r", encoding="utf-8") as f:
            results_data = json.load(f)

    if args.all_clips:
        clips = sorted(d.name for d in args.results_dir.iterdir() if d.is_dir() and d.name.startswith("clip_"))
        for clip in clips:
            process_clip(clip, args.results_dir, args.pitch_dir, results_data, args.ps_needle, args.av1_scale)
    else:
        process_clip(args.clip_name, args.results_dir, args.pitch_dir, results_data, args.ps_needle, args.av1_scale)


if __name__ == "__main__":
    main()


