"""SAM 3.1 at 360p and 12 fps: one prompt versus the original two passes.

The clip is the same 24 press frames as the 12 fps budget smoke. Masks are
compared with each other and with the saved full-resolution hand-union-arm masks.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/sam-pass-smokes")
CODE = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/code")
SAM_PY = "/home/itec/emanuele/.conda/envs/pointstream-sam31/bin/python"
CHECKPOINT = Path(
    "/home/itec/emanuele/.cache/huggingface/hub/models--facebook--sam3.1/"
    "snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)
WIDTH, HEIGHT = 640, 360


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def indices() -> list[int]:
    picked = []
    for second in range(2):
        for i in range(12):
            picked.append(second * 30 + min(29, int(round(i * 30 / 12))))
    return picked


def binary_pngs(folder: Path) -> list[np.ndarray]:
    masks = []
    for path in sorted(folder.glob("*.png")):
        arr = np.asarray(Image.open(path).convert("RGB"))
        masks.append(arr.max(axis=2) > 0)
    return masks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def sam(frames: Path, dest: Path, prompt: str) -> None:
    png = dest / "png"
    if len(list(png.glob("*.png"))) >= 24:
        return
    dest.mkdir(parents=True, exist_ok=True)
    prompts = dest / "prompts.json"
    prompts.write_text(json.dumps({"hand": prompt}))
    run([
        SAM_PY, "-m", "demo.pipeline.maps.sam31_video", "--worker",
        "--frames", str(frames),
        "--out", str(dest / "sam.mp4"),
        "--timing", str(dest / "timing.json"),
        "--checkpoint", str(CHECKPOINT),
        "--prompts", str(prompts),
        "--png-dir", str(png),
        "--concepts", "hand",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(CODE) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # subprocess.run in sam() does not pass env. Set it on the process via os.environ.
    __import__("os").environ["PYTHONPATH"] = env["PYTHONPATH"]

    frame_paths = sorted((TRAIN / "frames").glob("*.jpg"))
    mask_paths = sorted((TRAIN / "mask_frames").glob("*.png"))
    chosen = indices()
    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    reference = []
    for i, index in enumerate(chosen):
        image = cv2.imread(str(frame_paths[index]))
        small = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        dest = frames / f"{i:05d}.jpg"
        if not dest.exists():
            cv2.imwrite(str(dest), small)
        mask = cv2.imread(str(mask_paths[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 8
        reference.append(mask)

    started = time.time()
    sam(frames, OUT / "two_hand", "hand")
    sam(frames, OUT / "two_arm", "arm")
    two_s = round(time.time() - started, 1)
    hand = binary_pngs(OUT / "two_hand" / "png")
    arm = binary_pngs(OUT / "two_arm" / "png")
    two = [np.logical_or(h, a) for h, a in zip(hand, arm)]

    started = time.time()
    sam(frames, OUT / "one", "hand and arm")
    one_s = round(time.time() - started, 1)
    one = binary_pngs(OUT / "one" / "png")

    def mean_iou(left, right) -> float:
        return round(float(np.mean([iou(a, b) for a, b in zip(left, right)])), 3)

    row = {
        "scale": "640x360",
        "fps": 12,
        "frames": len(chosen),
        "one_pass_prompt": "hand and arm",
        "one_pass_seconds": one_s,
        "two_pass_seconds": two_s,
        "iou_one_vs_two": mean_iou(one, two),
        "iou_one_vs_saved_1080p": mean_iou(one, reference),
        "iou_two_vs_saved_1080p": mean_iou(two, reference),
        "coverage_one": round(float(np.mean([m.mean() for m in one])), 4),
        "coverage_two": round(float(np.mean([m.mean() for m in two])), 4),
        "coverage_saved": round(float(np.mean([m.mean() for m in reference])), 4),
    }
    (OUT / "report.json").write_text(json.dumps(row, indent=2))
    print("SCORE", json.dumps(row), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
