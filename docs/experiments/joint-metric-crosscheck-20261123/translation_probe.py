"""Measure the amount of RGB error attributable to a bounded spatial shift."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RADIUS = 6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb(path, expected):
    assert digest(path) == expected
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def translation(source, decoded, ys, xs):
    h, w = source.shape[:2]
    valid = (ys >= RADIUS) & (xs >= RADIUS) & (ys < h - RADIUS) & (xs < w - RADIUS)
    ys, xs = ys[valid], xs[valid]
    train = (ys + xs) % 2 == 0
    source_luma = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    decoded_luma = cv2.cvtColor(decoded, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ay, ax = ys[train], xs[train]
    known = source_luma[ay, ax]
    choices = [(dy, dx) for dy in range(-RADIUS, RADIUS + 1) for dx in range(-RADIUS, RADIUS + 1)]
    dy, dx = min(
        choices,
        key=lambda s: (
            np.mean((known - decoded_luma[ay + s[0], ax + s[1]]) ** 2),
            s[0] ** 2 + s[1] ** 2,
            s[0],
            s[1],
        ),
    )
    vy, vx = ys[~train], xs[~train]
    assert len(ay) > 50 and len(vy) > 50
    target = source[vy, vx].astype(np.float32)
    cost = [
        float(np.mean((target - decoded[vy + y, vx + x]) ** 2, dtype=np.float64))
        for y, x in ((0, 0), (dy, dx))
    ]
    return dict(shift=[dy, dx], pixels=len(vy), rgb_mse=cost[0], rgb_mse_shifted=cost[1])


def main(args):
    cv2.setNumThreads(1)
    prior = json.loads((HERE / "shanghai-cpu.json").read_text())
    review = json.loads((args.review / "review.json").read_text())
    assert digest(args.ball) == prior["official_ball_sha256"]
    assert len(review["frames"]) == prior["sampled_ticks"] == 12
    with args.ball.open(encoding="utf-8-sig") as f:
        marks = {int(row["Frame"]): row for row in csv.DictReader(f)}
    rows = {arm: [] for arm in prior["arms"]}
    for f in review["frames"]:
        tick = f["frame_index"]
        source = rgb(args.review / f["source"], f["source_sha256"])
        box = marks[tick]
        assert int(box["Visibility"]) == 1
        cx, cy = int(box["X"]), int(box["Y"])
        ys, xs = np.mgrid[cy - 32 : cy + 32, cx - 32 : cx + 32]
        supports = {"ball": (ys.ravel(), xs.ravel())}
        for p in f["players"]:
            mask_path = args.review / p["proposal"]
            assert digest(mask_path) == p["sha256"]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            assert int(mask.sum()) == p["pixels"]
            supports[p["id"]] = np.nonzero(mask)
        for arm, joined in prior["arms"].items():
            folder = args.arms / arm
            ledger = json.loads((folder / "arm.json").read_text())
            assert ledger["bitstream_sha256"] == joined["bitstream_sha256"]
            row = next(r for r in ledger["frames"] if r["tick"] == tick)
            display = rgb(folder / row["path"], row["png_sha256"])
            assert hashlib.sha256(display.tobytes()).hexdigest() == row["rgb_sha256"]
            rows[arm].append(
                dict(
                    tick=tick,
                    regions={
                        k: translation(source, display, *mask) for k, mask in supports.items()
                    },
                )
            )
    out = {}
    for arm, data in rows.items():
        groups = {
            k: [
                cell
                for row in data
                for key, cell in row["regions"].items()
                if (key == "ball") == (k == "ball")
            ]
            for k in ("player", "ball")
        }
        totals = {
            k: {key: mean(row[key] for row in items) for key in ("rgb_mse", "rgb_mse_shifted")}
            for k, items in groups.items()
        }
        out[arm] = dict(rows=data, means=totals)
    receipt = dict(
        radius=RADIUS,
        crosscheck_sha256=digest(HERE / "shanghai-cpu.json"),
        script_sha256=digest(Path(__file__)),
        arms=out,
    )
    with args.out.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    for arm, data in out.items():
        print(
            arm,
            {
                k: round(100 * (1 - v["rgb_mse_shifted"] / v["rgb_mse"]), 2)
                for k, v in data["means"].items()
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "ball", "arms", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
