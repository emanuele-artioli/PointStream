"""1080p SAM 3.1 and hands-only YOLOE, then the shared AV1 CRF ladder.

YOLOE is prompted with ``hand`` only. Tool and workbench prompts are omitted.
Painted frames stay at the source 1920x1080 size. Each rung uses
``av1_output_args`` (SVT-AV1 CRF 63, preset 7).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_LADDER, av1_output_args
from demo.pipeline.maps.sam31_video import extract_frames, resolve_checkpoint, resolve_python
from demo.pipeline.maps.yoloe_masks import (
    YOLOE_TRACK_CONF,
    YOLOE_TRACKER,
    YOLOE_IMGSZ,
    HandTrackFilter,
    load_prompt_map,
    load_yoloe_model,
    paint_class_masks,
)

CLIPS = ("clip_01", "clip_02", "clip_03")
FPS = 30.0
N_FRAMES = 300


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


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


def render_yoloe_hands(clip: Path, png_dir: Path) -> float:
    import torch

    png_dir.mkdir(parents=True, exist_ok=True)
    model, _weights, _extra = load_yoloe_model(["hand"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    names = {0: "hand"}
    track_filter = HandTrackFilter()
    step_ms: list[float] = []
    n_frames = 0
    t0 = time.perf_counter()
    for result in model.track(
        source=str(clip),
        stream=True,
        persist=True,
        retina_masks=True,
        verbose=False,
        device=device,
        half=True,
        imgsz=YOLOE_IMGSZ,
        conf=YOLOE_TRACK_CONF,
        tracker=str(YOLOE_TRACKER),
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if n_frames > 0:
            step_ms.append(dt_ms)
        if n_frames >= N_FRAMES:
            break
        orig = getattr(result, "orig_shape", None)
        frame_h = int(orig[0]) if orig is not None else 1080
        frame_w = int(orig[1]) if orig is not None else 1920
        painted = paint_class_masks(
            track_filter.class_masks(result, frame_h, frame_w, names),
            frame_h,
            frame_w,
        )
        if not cv2.imwrite(str(png_dir / f"{n_frames:06d}.png"), painted):
            raise RuntimeError(f"failed to write {png_dir}")
        n_frames += 1
        t0 = time.perf_counter()
    if n_frames <= 0:
        raise RuntimeError(f"YOLOE produced no frames for {clip}")
    return _percentile(step_ms, 50)


def render_sam(clip: Path, png_dir: Path, work: Path) -> float:
    python = resolve_python()
    checkpoint = resolve_checkpoint()
    if checkpoint is None or not Path(python).is_file():
        raise RuntimeError("SAM 3.1 checkpoint or python is missing")
    frames = work / "frames"
    extract_frames(clip, frames, ffmpeg=os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg"))
    # Keep the demo length.
    for extra in sorted(frames.glob("*.jpg"))[N_FRAMES:]:
        extra.unlink()
    prompts = work / "prompts.json"
    prompts.write_text(json.dumps(load_prompt_map("sam", clip.stem)))
    timing = work / "sam31_timing.json"
    cmd = [
        python, "-m", "demo.pipeline.maps.sam31_video", "--worker",
        "--frames", str(frames),
        "--out", str(work / "unused.mp4"),
        "--timing", str(timing),
        "--checkpoint", str(checkpoint),
        "--prompts", str(prompts),
        "--png-dir", str(png_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if res.returncode != 0 or not timing.is_file():
        tail = res.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"SAM worker failed ({res.returncode}): {tail}")
    data = json.loads(timing.read_text())
    steps = [float(v) for v in data.get("propagate_step_ms") or []]
    return (sum(steps) / len(steps)) if steps else 0.0


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
            png_dir = args.out / "png" / clip_name / "yoloe_hands"
            extract_ms = render_yoloe_hands(clip, png_dir)
            rungs = ladder(png_dir, args.out / clip_name / "yoloe", "yoloe", args.ffmpeg, N_FRAMES)
            report[clip_name]["yoloe"] = {"extract_ms_p50": round(extract_ms, 3), "rungs": rungs}
        if "sam31" in wanted:
            png_dir = args.out / "png" / clip_name / "sam31"
            extract_ms = render_sam(clip, png_dir, args.out / "work" / clip_name)
            rungs = ladder(png_dir, args.out / clip_name / "sam31", "sam31", args.ffmpeg, N_FRAMES)
            report[clip_name]["sam31"] = {"extract_ms_p50": round(extract_ms, 3), "rungs": rungs}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
