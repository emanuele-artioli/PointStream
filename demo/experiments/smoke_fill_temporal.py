"""Inpaint several press seconds concatenated, then scored at coarser frame rates.

Each source second is already filled at 30 fps and 1080p. Four of those seconds
are concatenated so a 6 fps sample still has 24 frames, which DiffuEraser accepts.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-demo/clip_01_factory001_worker001_00001/_work")
BATCHES = ("t000000", "t000900", "t001800", "t002700")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/temporal-smokes")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"
FPS = 30


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def indices_for(n: int, fps: int) -> list[int]:
    stride = FPS / fps
    chosen = []
    cursor = 0.0
    while int(round(cursor)) < n and len(chosen) < int(round(n / FPS * fps)):
        chosen.append(min(n - 1, int(round(cursor))))
        cursor += stride
    # One second at `fps` from each source second, in order.
    per = n // len(BATCHES)
    picked = []
    for batch in range(len(BATCHES)):
        base = batch * per
        step = per / fps
        for i in range(fps):
            picked.append(base + min(per - 1, int(round(i * step))))
    return picked


def write_video(frames: list[np.ndarray], dest: Path) -> None:
    folder = dest.parent / (dest.stem + "_png")
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:05d}.png"), frame)
    run([FFMPEG, "-y", "-framerate", "30", "-i", str(folder / "%05d.png"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    originals = []
    masks = []
    reference = []
    for name in BATCHES:
        batch = ROOT / name
        originals.extend(load_bgr(path) for path in sorted((batch / "frames").glob("*.jpg")))
        masks.extend(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 8 for path in sorted((batch / "masks").glob("*.png")))
        reference.extend(load_bgr(path) for path in sorted((batch / "kept").glob("*.jpg")))
    n = min(len(originals), len(masks), len(reference))
    originals, masks, reference = originals[:n], masks[:n], reference[:n]
    rows = []
    for fps in (12, 6):
        chosen = indices_for(n, fps)
        if len(chosen) < 23:
            raise SystemExit(f"{fps} fps produced {len(chosen)} frames")
        save = OUT / f"fps{fps}"
        result = save / "diffueraser_result.mp4"
        started = time.time()
        if not result.exists() or result.stat().st_size == 0:
            save.mkdir(parents=True, exist_ok=True)
            clip = save / "clip.mp4"
            mclip = save / "mask.mp4"
            if not clip.exists():
                write_video([originals[i] for i in chosen], clip)
            if not mclip.exists():
                write_video([np.repeat((masks[i].astype(np.uint8) * 255)[..., None], 3, axis=2) for i in chosen], mclip)
            run(
                [
                    DIFF_PY, "run_diffueraser.py",
                    "--input_video", str(clip), "--input_mask", str(mclip),
                    "--video_length", str(max(3, len(chosen) // 30 + 2)),
                    "--max_img_size", "1920", "--save_path", str(save),
                ],
                cwd=DIFF_DIR,
            )
        extracted = save / "frames"
        extracted.mkdir(exist_ok=True)
        if len(list(extracted.glob("*.png"))) < len(chosen):
            run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
        got = [load_bgr(path) for path in sorted(extracted.glob("*.png"))[: len(chosen)]]
        inside = []
        outside = []
        for index, frame in zip(chosen, got):
            ref = reference[index]
            mask = masks[index]
            if frame.shape[:2] != ref.shape[:2]:
                frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
            delta = np.abs(ref.astype(np.int16) - frame.astype(np.int16)).mean(axis=2)
            if mask.shape[:2] != delta.shape:
                mask = cv2.resize(mask.astype(np.uint8), (delta.shape[1], delta.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if mask.any():
                inside.append(float(delta[mask].mean()))
            if (~mask).any():
                outside.append(float(delta[~mask].mean()))
        row = {
            "fps": fps,
            "frames": len(chosen),
            "source_seconds": len(BATCHES),
            "seconds": round(time.time() - started, 1),
            "inside_mae_vs_30fps_1080p": round(float(np.mean(inside)), 3) if inside else None,
            "outside_mae_vs_30fps_1080p": round(float(np.mean(outside)), 3) if outside else None,
        }
        rows.append(row)
        print("SCORE", json.dumps(row), flush=True)
        (OUT / "report.json").write_text(json.dumps(rows, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
