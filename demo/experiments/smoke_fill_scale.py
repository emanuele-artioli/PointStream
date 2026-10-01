"""Time DiffuEraser resolutions against the saved 1080p fill of one press second."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path("/home/itec/emanuele/Datasets/pointstream-demo/clip_01_factory001_worker001_00001/_work/t000000")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/scale-smokes")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = sorted((SRC / "kept").glob("*.jpg"))
    masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 8 for path in sorted((SRC / "masks").glob("*.png"))]
    rows = []
    for size in (1280, 960):
        save = OUT / f"size_{size}"
        result = save / "diffueraser_result.mp4"
        started = time.time()
        if not result.exists() or result.stat().st_size == 0:
            save.mkdir(parents=True, exist_ok=True)
            run(
                [
                    DIFF_PY, "run_diffueraser.py",
                    "--input_video", str(SRC / "clip.mp4"),
                    "--input_mask", str(SRC / "mask.mp4"),
                    "--video_length", "3",
                    "--max_img_size", str(size),
                    "--save_path", str(save),
                ],
                cwd=DIFF_DIR,
            )
        extracted = save / "frames"
        extracted.mkdir(exist_ok=True)
        if len(list(extracted.glob("*.png"))) < len(reference):
            run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
        got = [cv2.imread(str(path)) for path in sorted(extracted.glob("*.png"))]
        inside = []
        outside = []
        for ref_path, frame, mask in zip(reference, got, masks):
            ref = cv2.imread(str(ref_path))
            if frame.shape[:2] != ref.shape[:2]:
                frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
            if mask.shape[:2] != ref.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            delta = np.abs(ref.astype(np.int16) - frame.astype(np.int16)).mean(axis=2)
            if mask.any():
                inside.append(float(delta[mask].mean()))
            if (~mask).any():
                outside.append(float(delta[~mask].mean()))
        row = {
            "max_img_size": size,
            "seconds": round(time.time() - started, 1),
            "inside_mae_vs_1080p": round(float(np.mean(inside)), 3) if inside else None,
            "outside_mae_vs_1080p": round(float(np.mean(outside)), 3) if outside else None,
        }
        rows.append(row)
        print("SCORE", json.dumps(row), flush=True)
        (OUT / "report.json").write_text(json.dumps(rows, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
