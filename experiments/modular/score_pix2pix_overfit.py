"""Score Pix2Pix scene fit on raw dataset crops; diagnostic, not codec PSNR.

The source segmentation is used only to delimit pixels scored. This script
does not consume a decoded AV1 reference or a pose reconstructed from wire
joints, so its result cannot be entered as a PointStream bitstream row.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3  # noqa: F401  # host C++ runtime must load before Torch
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from scripts.train_pix2pix import UNetGenerator
from src.shared.tennis_dataset import TennisSkeletonDataset


def _psnr(mse: float) -> float:
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def _mse(target: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        raise ValueError("empty target object mask")
    return float(np.mean((target[mask] - predicted[mask]) ** 2))


def score(checkpoint: Path, *, max_items: int | None = None) -> dict[str, object]:
    """Compare deterministic correct-pose and same-track shifted-pose inputs."""
    dataset = TennisSkeletonDataset(
        "/home/itec/emanuele/pointstream-data/assets/dataset",
        target_size=256,
        include_reference=True,
        condition="pose_body",
        reference_mode="first",
        video_filter="alcaraz_perricard",
        scene_filter="scene_002",
        frame_start=1,
        max_frames=47,
    )
    if len(dataset) != 94:
        raise RuntimeError(f"expected 94 exact-scene object targets, got {len(dataset)}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["G"] if isinstance(state, dict) and "G" in state else state
    model = UNetGenerator().eval()
    model.load_state_dict(weights)
    torch.manual_seed(42)

    by_track: dict[str, list[int]] = {}
    for index, item in enumerate(dataset.items):
        by_track.setdefault(item[2], []).append(index)
    successor = {
        index: indices[(position + len(indices) // 2) % len(indices)]
        for indices in by_track.values()
        for position, index in enumerate(indices)
    }
    actual = []
    shuffled = []
    crop_actual = []
    crop_shuffled = []
    n = len(dataset) if max_items is None else min(max_items, len(dataset))
    with torch.no_grad():
        for index in range(n):
            pose, reference, target = dataset[index]
            wrong_pose, _, _ = dataset[successor[index]]
            inputs = torch.stack(
                (torch.cat((pose, reference)), torch.cat((wrong_pose, reference)))
            )
            prediction = ((model(inputs) + 1.0) / 2.0).clamp(0, 1).numpy()
            target_np = ((target + 1.0) / 2.0).clamp(0, 1).numpy()
            object_mask = np.max(target_np, axis=0) > (2.0 / 255.0)
            object_mask_3 = np.broadcast_to(object_mask, target_np.shape)
            actual.append(_mse(target_np, prediction[0], object_mask_3))
            shuffled.append(_mse(target_np, prediction[1], object_mask_3))
            crop_actual.append(float(np.mean((target_np - prediction[0]) ** 2)))
            crop_shuffled.append(float(np.mean((target_np - prediction[1]) ** 2)))
    return {
        "kind": "raw-dataset exact-video optimization diagnostic; not a codec score",
        "checkpoint": str(checkpoint),
        "items": n,
        "tracks": sorted(by_track),
        "target_frame_ids": "1..47 per track; frame 0 is reference",
        "correct_pose_object_psnr_db": _psnr(float(np.mean(actual))),
        "shuffled_pose_object_psnr_db": _psnr(float(np.mean(shuffled))),
        "correct_pose_crop_psnr_db": _psnr(float(np.mean(crop_actual))),
        "shuffled_pose_crop_psnr_db": _psnr(float(np.mean(crop_shuffled))),
        "shuffled_pose_policy": "half-track offset within the same object track",
        "wire_inputs": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    result = score(args.checkpoint, max_items=args.max_items)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
