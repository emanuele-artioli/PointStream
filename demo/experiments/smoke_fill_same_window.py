"""30 fps 1080p inpaint of the same two press seconds as the 12 fps clip.

Scored only on the 24 frames that clip used. Sixty frames at 1920 do not fit
the 32 GB card, so this is two 30-frame windows, which is how the reference
fill was made.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/budget-smokes/fps30_2s_1080")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"
REPORT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/budget-smokes/report.json")


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_video(frames: list[np.ndarray], dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    folder = dest.parent / (dest.stem + "_png")
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:05d}.png"), frame)
    run([
        FFMPEG, "-y", "-framerate", "30", "-i", str(folder / "%05d.png"),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
    ])


def scored_indices() -> list[int]:
    picked = []
    for second in range(2):
        for i in range(12):
            picked.append(second * 30 + min(29, int(round(i * 30 / 12))))
    return picked


def fill_window(original: list[np.ndarray], masks: list[np.ndarray], save: Path) -> list[np.ndarray]:
    result = save / "diffueraser_result.mp4"
    if not result.exists() or result.stat().st_size == 0:
        save.mkdir(parents=True, exist_ok=True)
        write_video(original, save / "clip.mp4")
        write_video([np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2) for m in masks], save / "mask.mp4")
        run([
            DIFF_PY, "run_diffueraser.py",
            "--input_video", str(save / "clip.mp4"),
            "--input_mask", str(save / "mask.mp4"),
            "--video_length", "3",
            "--max_img_size", "1920",
            "--save_path", str(save),
        ], cwd=DIFF_DIR)
    extracted = save / "out"
    extracted.mkdir(exist_ok=True)
    if len(list(extracted.glob("*.png"))) < len(original):
        run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
    return [load_bgr(path) for path in sorted(extracted.glob("*.png"))[: len(original)]]


def main() -> None:
    frames = sorted((TRAIN / "frames").glob("*.jpg"))[:60]
    masks = sorted((TRAIN / "mask_frames").glob("*.png"))[:60]
    refs = sorted((TRAIN / "diffueraser" / "kept").glob("*.jpg"))[:60]
    if len(frames) < 60 or len(masks) < 60 or len(refs) < 60:
        raise SystemExit("press train does not have 60 frames")
    original = [load_bgr(path) for path in frames]
    mask = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 8 for path in masks]
    started = time.time()
    filled = fill_window(original[:30], mask[:30], OUT / "part0")
    filled += fill_window(original[30:], mask[30:], OUT / "part1")
    chosen = scored_indices()
    inside, outside = [], []
    for index in chosen:
        frame = filled[index]
        ref = load_bgr(refs[index])
        m = mask[index]
        if frame.shape[:2] != ref.shape[:2]:
            frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
        if m.shape[:2] != ref.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        delta = np.abs(ref.astype(np.int16) - frame.astype(np.int16)).mean(axis=2)
        if m.any():
            inside.append(float(delta[m].mean()))
        if (~m).any():
            outside.append(float(delta[~m].mean()))
    row = {
        "name": "fps30_2s_1080_on_12fps_frames",
        "max_img_size": 1920,
        "inpaint_frames": 60,
        "scored_frames": len(chosen),
        "chunk": 30,
        "seconds": round(time.time() - started, 1),
        "inside_mae_vs_1080p_30fps": round(float(np.mean(inside)), 3),
        "outside_mae_vs_1080p_30fps": round(float(np.mean(outside)), 3),
    }
    rows = json.loads(REPORT.read_text()) if REPORT.exists() else []
    rows = [item for item in rows if item.get("name") != row["name"]]
    rows.append(row)
    REPORT.write_text(json.dumps(rows, indent=2))
    print("SCORE", json.dumps(row), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
