"""Same 24-frame budget at 360p: one frame a second for 24s, or 24 fps for 1s.

Both are scored against the saved 1080p 30 fps DiffuEraser fill of the press train.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/budget-smokes")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"
SIZE = 360


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = sorted((TRAIN / "frames").glob("*.jpg"))
    masks = sorted((TRAIN / "mask_frames").glob("*.png"))
    refs = sorted((TRAIN / "diffueraser" / "kept").glob("*.jpg"))
    n = min(len(frames), len(masks), len(refs))
    # 24 fps for 1 second of a 30 fps source, and 1 fps for 24 seconds.
    def even(seconds: int, fps: int) -> list[int]:
        picked = []
        for second in range(seconds):
            for i in range(fps):
                picked.append(second * 30 + min(29, int(round(i * 30 / fps))))
        return picked

    cases = {
        "fps24_1s": (even(1, 24), 360),
        "fps12_2s": (even(2, 12), 360),
        "fps12_2s_720": (even(2, 12), 720),
        "fps12_2s_1080": (even(2, 12), 1920),
        "fps8_3s": (even(3, 8), 360),
        "fps1_24s": (list(range(0, 24 * 30, 30)), 360),
    }
    for name, (chosen, size) in cases.items():
        if any(i >= n for i in chosen) or len(chosen) < 23:
            raise SystemExit(f"{name} is not a legal {len(chosen)}-frame clip within {n} frames")
    rows = []
    report = OUT / "report.json"
    done = {}
    if report.exists():
        rows = json.loads(report.read_text())
        done = {row["name"] for row in rows}
    for name, (chosen, size) in cases.items():
        if name in done:
            print("SKIP", name, flush=True)
            continue
        save = OUT / name
        result = save / "diffueraser_result.mp4"
        started = time.time()
        if not result.exists() or result.stat().st_size == 0:
            save.mkdir(parents=True, exist_ok=True)
            original = [load_bgr(frames[i]) for i in chosen]
            mask = [cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE) > 8 for i in chosen]
            write_video(original, save / "clip.mp4")
            write_video([np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2) for m in mask], save / "mask.mp4")
            run([
                DIFF_PY, "run_diffueraser.py",
                "--input_video", str(save / "clip.mp4"),
                "--input_mask", str(save / "mask.mp4"),
                "--video_length", "3",
                "--max_img_size", str(size),
                "--save_path", str(save),
            ], cwd=DIFF_DIR)
        extracted = save / "out"
        extracted.mkdir(exist_ok=True)
        if len(list(extracted.glob("*.png"))) < len(chosen):
            run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
        got = [load_bgr(path) for path in sorted(extracted.glob("*.png"))[: len(chosen)]]
        inside, outside = [], []
        for index, frame in zip(chosen, got):
            ref = load_bgr(refs[index])
            mask = cv2.imread(str(masks[index]), cv2.IMREAD_GRAYSCALE) > 8
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
            "name": name,
            "max_img_size": size,
            "frames": len(chosen),
            "seconds": round(time.time() - started, 1),
            "inside_mae_vs_1080p_30fps": round(float(np.mean(inside)), 3) if inside else None,
            "outside_mae_vs_1080p_30fps": round(float(np.mean(outside)), 3) if outside else None,
        }
        rows.append(row)
        print("SCORE", json.dumps(row), flush=True)
        report.write_text(json.dumps(rows, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
