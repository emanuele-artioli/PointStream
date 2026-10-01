"""Score DiffuEraser sampling recipes against the accepted 30 fps factory001 hold-out.

The hold-out frames, masks, and fill already exist. This script only re-inpaints
subsampled copies so the comparison is the sampling pattern, not a new mask.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/sample-smokes")
DIFF_PY = "/home/itec/emanuele/.conda/envs/pointstream-diffueraser/bin/python"
DIFF_DIR = "/home/itec/emanuele/pointstream-data/jobs/neural-bg/src/DiffuEraser"
FFMPEG = "/opt/local/bin/ffmpeg"
FPS = 30


def run(cmd: list[str], cwd: str | None = None) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 8


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_clip(frames: list[np.ndarray], dest: Path, *, color: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    folder = dest.parent / (dest.stem + "_png")
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"{index:05d}.png"), frame if color else frame)
    run(
        [
            FFMPEG, "-y", "-framerate", str(FPS), "-i", str(folder / "%05d.png"),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
        ]
    )


def fill_chunk(video: Path, mask: Path, save: Path, n_frames: int) -> list[np.ndarray]:
    save.mkdir(parents=True, exist_ok=True)
    result = save / "diffueraser_result.mp4"
    if not result.exists() or result.stat().st_size == 0:
        run(
            [
                DIFF_PY, "run_diffueraser.py",
                "--input_video", str(video),
                "--input_mask", str(mask),
                "--video_length", str(max(n_frames // FPS + 2, 3)),
                "--max_img_size", "1920",
                "--save_path", str(save),
            ],
            cwd=DIFF_DIR,
        )
    extracted = save / "extracted"
    extracted.mkdir(exist_ok=True)
    if len(list(extracted.glob("*.png"))) < n_frames:
        run([FFMPEG, "-y", "-i", str(result), str(extracted / "%05d.png")])
    paths = sorted(extracted.glob("*.png"))
    return [load_bgr(path) for path in paths[:n_frames]]


def inpaint(name: str, indices: list[int], originals: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    root = OUT / name
    if (root / "filled.json").exists():
        cached = json.loads((root / "filled.json").read_text())
        return [load_bgr(Path(path)) for path in cached]
    chunk = 30
    kept: list[np.ndarray | None] = [None] * len(indices)
    written = 0
    starts = list(range(0, len(indices), chunk))
    if len(indices) - starts[-1] < 23:
        starts[-1] = max(0, len(indices) - chunk)
    for start in starts:
        end = min(len(indices), start + chunk)
        part = root / f"part_{start:05d}"
        clip_frames = originals[start:end]
        mask_frames = [
            np.repeat((masks[i].astype(np.uint8) * 255)[..., None], 3, axis=2)
            for i in range(start, end)
        ]
        clip = part / "clip.mp4"
        mclip = part / "mask.mp4"
        if not clip.exists():
            write_clip(clip_frames, clip, color=True)
        if not mclip.exists():
            write_clip(mask_frames, mclip, color=True)
        filled = fill_chunk(clip, mclip, part / "diffueraser", end - start)
        for offset, frame in enumerate(filled):
            kept[start + offset] = frame
        written = sum(frame is not None for frame in kept)
        print(name, "chunk", start, end, "kept", written, flush=True)
    missing = [i for i, frame in enumerate(kept) if frame is None]
    if missing:
        raise SystemExit(f"{name} missing frames {missing[:8]}")
    paths = []
    folder = root / "filled"
    folder.mkdir(exist_ok=True)
    for index, frame in enumerate(kept):
        dest = folder / f"{index:05d}.png"
        cv2.imwrite(str(dest), frame)
        paths.append(str(dest))
    (root / "filled.json").write_text(json.dumps(paths))
    return kept  # type: ignore[return-value]


def score(name: str, indices: list[int], reference: list[np.ndarray], masks: list[np.ndarray], filled: list[np.ndarray], seconds: float) -> dict:
    inside = []
    outside = []
    flicker_fill = []
    flicker_ref = []
    for index, (ref, got, mask) in enumerate(zip(reference, filled, masks)):
        if got.shape[:2] != ref.shape[:2]:
            got = cv2.resize(got, (ref.shape[1], ref.shape[0]))
        delta = np.abs(ref.astype(np.int16) - got.astype(np.int16)).mean(axis=2)
        if mask.any():
            inside.append(float(delta[mask].mean()))
        if (~mask).any():
            outside.append(float(delta[~mask].mean()))
        if index:
            flicker_fill.append(float(np.abs(got.astype(np.int16) - filled[index - 1].astype(np.int16)).mean()))
            flicker_ref.append(float(np.abs(ref.astype(np.int16) - reference[index - 1].astype(np.int16)).mean()))
    row = {
        "name": name,
        "frames": len(indices),
        "source_span_frames": int(indices[-1] - indices[0] + 1) if indices else 0,
        "seconds": round(seconds, 1),
        "inside_mae": round(float(np.mean(inside)), 3) if inside else None,
        "outside_mae": round(float(np.mean(outside)), 3) if outside else None,
        "flicker": round(float(np.mean(flicker_fill)), 3) if flicker_fill else None,
        "reference_flicker": round(float(np.mean(flicker_ref)), 3) if flicker_ref else None,
    }
    print("SCORE", json.dumps(row), flush=True)
    return row


def bundle(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    frames = sorted((root / "frames").glob("*.jpg"))
    masks = sorted((root / "mask_frames").glob("*.png"))
    refs = sorted((root / "diffueraser" / "kept").glob("*.jpg"))
    n = min(len(frames), len(masks), len(refs))
    return frames[:n], masks[:n], refs[:n]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_frames, train_masks, train_refs = bundle(TRAIN)
    # The 978s hold-out is the worker walking the aisle. The train window is the
    # press. Score sampling only on that press window.
    press = slice(0, min(300, len(train_frames)))
    press_frames = train_frames[press]
    press_masks = train_masks[press]
    press_refs = train_refs[press]
    recipes = {
        "fps1_train": (train_frames, train_masks, train_refs, list(range(0, len(train_frames), 30))),
        "fps6_press": (press_frames, press_masks, press_refs, list(range(0, len(press_frames), 5))),
        "fps12_press": (press_frames, press_masks, press_refs, list(range(0, len(press_frames), 3))),
        "batch_1s_press": (press_frames, press_masks, press_refs, list(range(0, 30))),
    }
    report_path = OUT / "report.json"
    rows = json.loads(report_path.read_text()) if report_path.exists() else []
    done = {row["name"] for row in rows}
    for name, (frames, masks, refs, indices) in recipes.items():
        if name in done:
            print("SKIP", name, flush=True)
            continue
        if len(indices) < 23:
            raise SystemExit(f"{name} has {len(indices)} frames; DiffuEraser needs more than 22")
        started = time.time()
        filled = inpaint(
            name,
            indices,
            [load_bgr(frames[i]) for i in indices],
            [load_gray(masks[i]) for i in indices],
            )
        rows.append(
            score(
                name,
                indices,
                [load_bgr(refs[i]) for i in indices],
                [load_gray(masks[i]) for i in indices],
                filled,
                time.time() - started,
            )
        )
        (OUT / "report.json").write_text(json.dumps(rows, indent=2))
    print("DONE", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
