"""One-second inpaint smoke: SAM 3.1 arm/hand mask, feathered edge, DiffuEraser fill.

DiffuEraser is the kept filler: it is a video model, so it keeps temporal
coherence. SDXL, FLUX, and Qwen stay in demo.experiments.archive.neural_bg_still_fills.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from demo.experiments.neural_bg import cut_segment, feather_mask
from src.segmentation import build, load_domain
from src.segmentation.sources import extract_jpegs

logger = logging.getLogger(__name__)

PROMPT = "empty factory workstation, same lighting, no hands, no person, no tools"


def _load_frames(folder: Path) -> list[np.ndarray]:
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    return frames


def _composite(original: np.ndarray, filled: np.ndarray, soft: np.ndarray) -> np.ndarray:
    alpha = soft[..., None]
    mixed = original.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


DIFFUERASER_ROOT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser")
DIFFUERASER_PYTHON = Path("/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python")


def _write_rgb_video(frames: list[np.ndarray], dest: Path, ffmpeg: str, *, fps: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
    ]
    proc = subprocess.run(cmd, input=b"".join(frame.tobytes() for frame in frames), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not dest.is_file():
        tail = proc.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"video write failed for {dest}: {tail}")


def _fill_diffueraser(frames, masks, work: Path, ffmpeg: str) -> list[np.ndarray]:
    if not DIFFUERASER_PYTHON.is_file():
        raise FileNotFoundError(f"DiffuEraser python missing: {DIFFUERASER_PYTHON}")
    runner = DIFFUERASER_ROOT / "run_diffueraser.py"
    if not runner.is_file():
        raise FileNotFoundError(f"DiffuEraser runner missing: {runner}")
    clip = work / "diffueraser_input.mp4"
    mask_clip = work / "diffueraser_mask.mp4"
    out_dir = work / "diffueraser_out"
    _write_rgb_video(frames, clip, ffmpeg)
    mask_frames = [np.repeat(((mask > 0).astype(np.uint8) * 255)[..., None], 3, axis=2) for mask in masks]
    _write_rgb_video(mask_frames, mask_clip, ffmpeg)
    cmd = [
        str(DIFFUERASER_PYTHON), str(runner),
        "--input_video", str(clip),
        "--input_mask", str(mask_clip),
        "--video_length", str(len(frames)),
        "--max_img_size", str(max(frames[0].shape[0], frames[0].shape[1])),
        "--save_path", str(out_dir),
    ]
    res = subprocess.run(cmd, cwd=str(DIFFUERASER_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"DiffuEraser failed: {res.returncode}")
    result = out_dir / "diffueraser_result.mp4"
    if not result.is_file():
        raise FileNotFoundError(f"DiffuEraser wrote no result at {result}")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extract = subprocess.run(
        [ffmpeg, "-y", "-i", str(result), str(frames_dir / "%05d.png")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if extract.returncode != 0:
        raise RuntimeError("failed to extract DiffuEraser frames")
    filled = _load_frames(frames_dir)
    if not filled:
        raise RuntimeError("DiffuEraser produced no frames")
    out = []
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        soft = feather_mask(mask)
        recon = filled[min(index, len(filled) - 1)]
        if recon.shape[:2] != frame.shape[:2]:
            recon = cv2.resize(recon, (frame.shape[1], frame.shape[0]))
        out.append(_composite(frame, recon, soft))
    return out


def contact_sheet(originals, mask_vis, fills: dict[str, list[np.ndarray]], dest: Path) -> None:
    rows = []
    for index, frame in enumerate(originals):
        cells = [frame, mask_vis[index]]
        for name in fills:
            cells.append(fills[name][index] if index < len(fills[name]) else np.zeros_like(frame))
        thumb = [cv2.resize(cell, (320, 180)) for cell in cells]
        rows.append(np.concatenate(thumb, axis=1))
    sheet = np.concatenate(rows, axis=0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--scale", default="640:360")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.work.mkdir(parents=True, exist_ok=True)
    snippet = args.work / "snippet.mp4"
    ffmpeg = "/opt/local/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        ffmpeg = "ffmpeg"
    cut_segment(args.clip, snippet, start_s=1.0, duration_s=args.seconds, ffmpeg=ffmpeg)
    if args.scale:
        scaled = args.work / "snippet_scaled.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(snippet), "-vf", f"scale={args.scale}",
                        "-c:v", "libx264", "-crf", "12", str(scaled)], check=True)
        snippet = scaled
    frames_dir = args.work / "frames"
    count = max(1, int(round(args.seconds * 30)))
    extract_jpegs(snippet, frames_dir, count)
    frames = _load_frames(frames_dir)[:count]
    sam = build("sam31").segment(frames_dir, load_domain("egocentric"))
    sam.save(args.work / "sam")
    masks = [sam.foreground(index).astype(np.uint8) for index in range(len(sam))]
    n = min(len(frames), len(masks), 8)
    frames, masks = frames[:n], masks[:n]
    fills: dict[str, list[np.ndarray]] = {}
    notes = {
        "chosen": "diffueraser",
        "archived": ["sdxl_inpaint", "flux_fill", "qwen_edit"],
        "reason": "DiffuEraser is the video model; still-image fills stay archived",
    }
    if args.checkpoints.is_file():
        notes["checkpoint_manifest"] = str(args.checkpoints)
    try:
        fills["diffueraser"] = _fill_diffueraser(frames, masks, args.work, ffmpeg)
        notes["diffueraser"] = "ok"
    except Exception as exc:
        notes["diffueraser"] = f"stopped: {exc}"
        logger.exception("fill diffueraser failed")
    mask_vis = []
    for frame, mask in zip(frames, masks):
        vis = frame.copy()
        soft = feather_mask(mask)
        vis[soft > 0.2] = (0, 0, 255)
        mask_vis.append(vis)
    sheet = args.work / "contact_sheet.png"
    contact_sheet(frames, mask_vis, fills, sheet)
    report = {"notes": notes, "frames": n, "sheet": str(sheet), "device": args.device}
    (args.work / "smoke.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
