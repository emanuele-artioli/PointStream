"""Audit source-only feature track/detection coverage; no ground-truth claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.tennis.k4b_pose import player_masks
from experiments.visor.h1 import match_detections
from src.segmentation.masks import ClipMasks, decompress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads((a.features / "features.json").read_text())
    clip = ClipMasks.load(a.features / "sam-masks.rle")
    detections = json.loads(decompress((a.features / "pose.json.zst").read_bytes()))
    names = ["person:000000000007", "person:000000000000"]
    if len(clip.frames) != 240 or len(detections) != 240:
        raise ValueError("native frame coverage")
    out = {
        k: {"mask": 0, "matched": 0, "missing_ticks": [], "edge_ticks": [], "centers": {}}
        for k in names
    }
    for t, frame in enumerate(clip.frames):
        masks = player_masks(frame, data["aliases"], names)
        boxes = np.asarray(detections[t]["boxes"], dtype=float).reshape(-1, 4)
        pairs = match_detections(masks, boxes * 0.5) if len(boxes) else {}
        for name in names:
            record = out[name]
            if name not in masks or not masks[name].any():
                record["missing_ticks"].append(t)
                continue
            record["mask"] += 1
            ys, xs = np.nonzero(masks[name])
            if xs.min() == 0 or xs.max() >= 959 or ys.min() == 0 or ys.max() >= 539:
                record["edge_ticks"].append(t)
            if name in pairs:
                record["matched"] += 1
            if t % 20 == 0 or t == 239:
                record["centers"][str(t)] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
    with a.output.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {name: {"mask": out[name]["mask"], "matched": out[name]["matched"]} for name in names}
        )
    )


if __name__ == "__main__":
    main()
