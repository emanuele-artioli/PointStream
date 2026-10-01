"""Fast fill grid: 360p and 540p, at 6 and 12 fps, continuous and joined.

The 30 fps cell at each size is only the first second. It separates resolution
error from frame-rate error. Scores are against the saved 1080p 30 fps fill.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
JOINED = Path("/home/itec/emanuele/Datasets/pointstream-demo/clip_01_factory001_worker001_00001/_work")
JOINED_BATCHES = ("t000000", "t000900", "t001800", "t002700")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/grid-smokes")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"
SOURCE_FPS = 30


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_span(frames: Path, masks: Path, refs: Path, count: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    original = [load_bgr(path) for path in sorted(frames.glob("*.jpg"))[:count]]
    mask = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 8 for path in sorted(masks.glob("*.png"))[:count]]
    reference = [load_bgr(path) for path in sorted(refs.glob("*.jpg"))[:count]]
    n = min(len(original), len(mask), len(reference))
    return original[:n], mask[:n], reference[:n]


def pick(n_per_second: int, seconds: int, fps: int) -> list[int]:
    picked = []
    for second in range(seconds):
        base = second * n_per_second
        for i in range(fps):
            picked.append(base + min(n_per_second - 1, int(round(i * n_per_second / fps))))
    return picked


def write_video(frames: list[np.ndarray], dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    folder = dest.parent / (dest.stem + "_png")
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:05d}.png"), frame)
    run([FFMPEG, "-y", "-framerate", "30", "-i", str(folder / "%05d.png"), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)])


def score(reference, masks, chosen, got) -> tuple[float | None, float | None]:
    inside, outside = [], []
    for index, frame in zip(chosen, got):
        ref = reference[index]
        mask = masks[index]
        if frame.shape[:2] != ref.shape[:2]:
            frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
        if mask.shape[:2] != ref.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        delta = np.abs(ref.astype(np.int16) - frame.astype(np.int16)).mean(axis=2)
        if mask.any():
            inside.append(float(delta[mask].mean()))
        if (~mask).any():
            outside.append(float(delta[~mask].mean()))
    return (
        round(float(np.mean(inside)), 3) if inside else None,
        round(float(np.mean(outside)), 3) if outside else None,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    continuous = load_span(TRAIN / "frames", TRAIN / "mask_frames", TRAIN / "diffueraser" / "kept", 120)
    joined_original, joined_masks, joined_refs = [], [], []
    for name in JOINED_BATCHES:
        batch = JOINED / name
        original, mask, reference = load_span(batch / "frames", batch / "masks", batch / "kept", 30)
        joined_original.extend(original)
        joined_masks.extend(mask)
        joined_refs.extend(reference)
    pools = {
        "continuous": continuous,
        "joined": (joined_original, joined_masks, joined_refs),
    }
    cases = []
    for size in (360, 540):
        for fps in (6, 12):
            for layout in ("continuous", "joined"):
                cases.append((layout, size, fps, 4))
        for layout in ("continuous",):
            cases.append((layout, size, 30, 1))
    report = OUT / "report.json"
    rows = json.loads(report.read_text()) if report.exists() else []
    done = {(row.get("layout"), row.get("max_img_size"), row.get("fps")) for row in rows}
    for layout, size, fps, seconds in cases:
        if (layout, size, fps) in done:
            print("SKIP", layout, size, fps, flush=True)
            continue
        original, masks, reference = pools[layout]
        per = 30
        chosen = pick(per, seconds, fps)
        if len(chosen) < 23:
            raise SystemExit(f"{layout} {size} {fps} has {len(chosen)} frames")
        save = OUT / f"{layout}_s{size}_fps{fps}"
        result = save / "diffueraser_result.mp4"
        started = time.time()
        row = {"layout": layout, "max_img_size": size, "fps": fps, "source_seconds": seconds, "frames": len(chosen)}
        try:
            if not result.exists() or result.stat().st_size == 0:
                save.mkdir(parents=True, exist_ok=True)
                write_video([original[i] for i in chosen], save / "clip.mp4")
                write_video([np.repeat((masks[i].astype(np.uint8) * 255)[..., None], 3, axis=2) for i in chosen], save / "mask.mp4")
                run(
                    [
                        DIFF_PY, "run_diffueraser.py",
                        "--input_video", str(save / "clip.mp4"),
                        "--input_mask", str(save / "mask.mp4"),
                        "--video_length", str(max(3, len(chosen) // 30 + 2)),
                        "--max_img_size", str(size),
                        "--save_path", str(save),
                    ],
                    cwd=DIFF_DIR,
                )
            extracted = save / "frames"
            extracted.mkdir(exist_ok=True)
            if len(list(extracted.glob("*.png"))) < len(chosen):
                run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
            got = [load_bgr(path) for path in sorted(extracted.glob("*.png"))[: len(chosen)]]
            inside, outside = score(reference, masks, chosen, got)
            row.update({
                "seconds": round(time.time() - started, 1),
                "inside_mae_vs_1080p_30fps": inside,
                "outside_mae_vs_1080p_30fps": outside,
            })
        except subprocess.CalledProcessError as exc:
            row.update({"seconds": round(time.time() - started, 1), "error": f"exit {exc.returncode}"})
        rows.append(row)
        print("SCORE", json.dumps(row), flush=True)
        report.write_text(json.dumps(rows, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
