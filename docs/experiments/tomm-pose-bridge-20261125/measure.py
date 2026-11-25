"""Measure interpolation under artificial omissions of paid PSCP joints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiments.tennis.common_player_packet import unpack

INPUTS = {
    "match102": "3ebc2e455509db026919f17c17d2bfce3eba899fd29a0888c47440b576a219af",
    "match127": "a70a444b65f905285830a5fc2561e752708ac24ea0a5e02d538d85ac742dd99c",
}
COCO_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
     0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089],
)


def track_scores(joints: np.ndarray, boxes: np.ndarray, gap: int) -> dict:
    starts = np.arange(1, len(joints) - gap)
    omitted = starts[:, None] + np.arange(gap)
    target = joints[omitted]
    valid = target[..., 2] >= 0.3
    left = joints[starts - 1, None]
    right = joints[starts + gap, None]
    alpha = np.arange(1, gap + 1)[None, :, None, None] / (gap + 1)
    height = boxes[omitted, 3, None]
    area = boxes[omitted, 2, None] * height
    denominator = valid.sum(axis=(1, 2))
    if np.any(denominator == 0) or np.any(area <= 0):
        raise ValueError("Unscorable gap support")

    def score(prediction: np.ndarray, available: np.ndarray) -> dict[str, float]:
        errors = np.linalg.norm(prediction - target[..., :2], axis=-1)
        hits = valid & available & (errors <= 0.1 * height)
        pck = hits.sum(axis=(1, 2)) / denominator
        oks = np.exp(-(errors ** 2) / (2 * area * (2 * COCO_SIGMAS) ** 2))
        oks_by_gap = ((valid & available) * oks).sum(axis=(1, 2)) / denominator
        return {
            "mean_pck_01": float(pck.mean()),
            "p10_gap_pck_01": float(np.quantile(pck, 0.10)),
            "mean_oks": float(oks_by_gap.mean()),
        }

    bridge = score((1 - alpha) * left[..., :2] + alpha * right[..., :2],
                   (left[..., 2] > 0) & (right[..., 2] > 0))
    hold = score(left[..., :2], left[..., 2] > 0)
    return {
        "starts": len(starts),
        "paid_joint_observations": int(denominator.sum()),
        "bridge": bridge,
        "last_pose_hold": hold,
        "passes_render_gate": (
            bridge["mean_pck_01"] >= 0.95
            and bridge["p10_gap_pck_01"] >= 0.80
            and bridge["mean_oks"] > hold["mean_oks"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match", choices=INPUTS)
    parser.add_argument("carrier", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    carrier = args.carrier.read_bytes()
    identity = hashlib.sha256(carrier).hexdigest()
    if identity != INPUTS[args.match]:
        raise ValueError(f"Unexpected {args.match} carrier SHA256: {identity}")
    _, _, runs = unpack(carrier)
    if set(runs) != {f"{args.match}_{role}" for role in ("near", "far")}:
        raise ValueError("Unexpected carrier roles")
    results = {}
    for name, track in runs.items():
        if len(track) != 1 or not np.array_equal(track[0]["t"], np.arange(240)):
            raise ValueError(f"Expected one complete 240-tick track for {name}")
        results[name] = {
            str(gap): track_scores(track[0]["kp"], track[0]["boxes"], gap)
            for gap in (5, 10)
        }
    receipt = {
        "match": args.match,
        "revision": args.revision,
        "carrier_sha256": identity,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": f"CPU {platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": time.monotonic() - start,
        "results": results,
    }
    with args.output.open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
