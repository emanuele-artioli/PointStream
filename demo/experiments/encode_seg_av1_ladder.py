"""1080p SAM 3.1 and YOLOE-26x egocentric masks, then the shared AV1 CRF ladder.

Both backends segment the egocentric foreground (arm, hand) through
`src.segmentation`. Painted frames stay at the source 1920x1080 size. Each
rung uses ``av1_output_args`` (SVT-AV1 CRF 63, preset 7).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_LADDER, av1_output_args
from demo.pipeline.maps.masks import paint
from src.segmentation import build, load_domain

CLIPS = ("clip_01", "clip_02", "clip_03")
FPS = 30.0
N_FRAMES = 300


def encode_sequence(seq_dir: Path, dest: Path, scale: str | None, ffmpeg: str, n_frames: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-framerate", f"{FPS:.6f}",
        "-start_number", "0",
        "-i", str(seq_dir / "%06d.png"),
        "-frames:v", str(n_frames),
        *av1_output_args(scale),
        str(dest),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        tail = res.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"AV1 encode failed for {dest}: {tail}")


def ladder(seq_dir: Path, dest_dir: Path, stem: str, ffmpeg: str, n_frames: int) -> dict[str, dict]:
    duration_s = n_frames / FPS
    out: dict[str, dict] = {}
    for name, scale in AV1_LADDER:
        dest = dest_dir / f"{stem}_{name}_crf{AV1_CRF}.mp4"
        encode_sequence(seq_dir, dest, scale, ffmpeg, n_frames)
        nbytes = dest.stat().st_size
        out[name] = {
            "bytes": nbytes,
            "kbps": round(nbytes * 8 / duration_s / 1000.0, 1),
            "path": str(dest),
        }
        print(f"{stem} {name} {out[name]['kbps']} kbps", flush=True)
    return out


def render(backend: str, clip: Path, png_dir: Path) -> float:
    """Paint each frame's masks to PNG; returns ms per frame (model load excluded)."""
    png_dir.mkdir(parents=True, exist_ok=True)
    masks = build(backend).segment(clip, load_domain("egocentric"), max_frames=N_FRAMES)
    masks.save(png_dir.parent / f"{backend}_masks")
    for index in range(len(masks)):
        if not cv2.imwrite(str(png_dir / f"{index:06d}.png"), paint(masks, index)):
            raise RuntimeError(f"failed to write {png_dir}")
    return float(masks.meta["timing"]["ms_per_frame"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--only", default="yoloe,sam31")
    args = parser.parse_args()
    os.environ["FFMPEG"] = args.ffmpeg
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    report: dict[str, dict] = {}
    for clip_name in CLIPS:
        clip = args.clips_dir / f"{clip_name}.mp4"
        report[clip_name] = {}
        if "yoloe" in wanted:
            png_dir = args.out / "png" / clip_name / "yoloe"
            extract_ms = render("yoloe-26x", clip, png_dir)
            rungs = ladder(png_dir, args.out / clip_name / "yoloe", "yoloe", args.ffmpeg, N_FRAMES)
            report[clip_name]["yoloe"] = {"ms_per_frame": round(extract_ms, 3), "rungs": rungs}
        if "sam31" in wanted:
            png_dir = args.out / "png" / clip_name / "sam31"
            extract_ms = render("sam31", clip, png_dir)
            rungs = ladder(png_dir, args.out / clip_name / "sam31", "sam31", args.ffmpeg, N_FRAMES)
            report[clip_name]["sam31"] = {"ms_per_frame": round(extract_ms, 3), "rungs": rungs}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
