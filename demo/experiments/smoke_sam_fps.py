"""SAM 3.1 at native 1080p, 12 fps, two passes, versus the saved 30 fps masks.

The 24 frames are the same press indices as the 12 fps inpaint smoke. Spatial
resolution stays native: the encoder always resizes to 1008, so a smaller JPEG
does not shrink the model. This script refuses a GPU without flash attention.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/filled-hnerv-factory001/train")
OUT = Path("/home/itec/emanuele/pointstream-data/jobs/neural-bg/sam-fps-smokes/fps12_native")
SAM_PY = "/home/itec/emanuele/.conda/envs/pointstream-sam31/bin/python"
CHECKPOINT = Path(
    "/home/itec/emanuele/.cache/huggingface/hub/models--facebook--sam3.1/"
    "snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)
ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def indices() -> list[int]:
    picked = []
    for second in range(2):
        for i in range(12):
            picked.append(second * 30 + min(29, int(round(i * 30 / 12))))
    return picked


def require_flash_gpu() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if major < 8:
        raise SystemExit(
            f"SAM 3.1 needs flash attention (compute capability >= 8). This GPU is {name} ({major}.{minor})."
        )
    return f"{name} cc{major}.{minor}"


def binary_pngs(folder: Path) -> list[np.ndarray]:
    masks = []
    for path in sorted(folder.glob("*.png")):
        arr = np.asarray(Image.open(path).convert("L"))
        masks.append(arr > 8)
    return masks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def sam(frames: Path, dest: Path, prompt: str) -> dict:
    png = dest / "png"
    timing = dest / "timing.json"
    if len(list(png.glob("*.png"))) >= 24 and timing.is_file():
        return json.loads(timing.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    prompts = dest / "prompts.json"
    prompts.write_text(json.dumps({"hand": prompt}))
    run([
        SAM_PY, "-m", "demo.pipeline.maps.sam31_video", "--worker",
        "--frames", str(frames),
        "--out", str(dest / "sam.mp4"),
        "--timing", str(timing),
        "--checkpoint", str(CHECKPOINT),
        "--prompts", str(prompts),
        "--png-dir", str(png),
        "--concepts", "hand",
    ])
    return json.loads(timing.read_text())


def main() -> None:
    gpu = require_flash_gpu()
    os.environ["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted((TRAIN / "frames").glob("*.jpg"))
    mask_paths = sorted((TRAIN / "mask_frames").glob("*.png"))
    chosen = indices()
    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    reference = []
    for i, index in enumerate(chosen):
        dest = frames / f"{i:05d}.jpg"
        if not dest.exists():
            os.symlink(frame_paths[index], dest)
        mask = np.asarray(Image.open(mask_paths[index]).convert("L")) > 8
        reference.append(mask)

    started = time.time()
    hand_timing = sam(frames, OUT / "hand", "hand")
    arm_timing = sam(frames, OUT / "arm", "arm")
    elapsed = round(time.time() - started, 1)
    hand = binary_pngs(OUT / "hand" / "png")
    arm = binary_pngs(OUT / "arm" / "png")
    union = [np.logical_or(h, a) for h, a in zip(hand, arm)]
    scores = [iou(pred, ref) for pred, ref in zip(union, reference)]
    row = {
        "gpu": gpu,
        "scale": "native",
        "fps": 12,
        "source_fps": 30,
        "frames": len(chosen),
        "source_indices": chosen,
        "prompts": ["hand", "arm"],
        "elapsed_s": elapsed,
        "hand_model_load_s": hand_timing.get("model_load_s"),
        "arm_model_load_s": arm_timing.get("model_load_s"),
        "hand_prompt_s": hand_timing.get("prompt_s"),
        "arm_prompt_s": arm_timing.get("prompt_s"),
        "mean_iou_vs_30fps_union": round(float(np.mean(scores)), 4),
        "min_iou_vs_30fps_union": round(float(np.min(scores)), 4),
        "per_frame_iou": [round(float(v), 4) for v in scores],
        "reference": "train mask_frames, the saved 30 fps hand-union-arm masks",
    }
    (OUT / "report.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
