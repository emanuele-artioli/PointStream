"""Assemble a fixed-instant author figure from native receiver crops and checked control YUV."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pointstream.tracked_patch import yuv_to_rgb

plt.rcParams["pdf.fonttype"] = 42


HERE = Path(__file__).resolve().parent
TICKS = (90, 170)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def native_parts(arm, boxes):
    raw = Path(arm["path"])
    assert sha(raw) == arm["sha256"]
    w, h = arm["width"], arm["height"]
    assert raw.stat().st_size == 240 * h * w * 3 // 2
    decoded = np.memmap(raw, np.uint8, "r", shape=(240, h * 3 // 2, w))
    cropped = {}
    for t in TICKS:
        full = cv2.resize(
            yuv_to_rgb(decoded[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )
        for i in (0, 1):
            x0, y0, x1, y1 = boxes[t][f"player-{i}"]
            cropped[(t, i)] = full[y0:y1, x0:x1].copy()
    return cropped


def true_crop(root, ledger, t, name, idx):
    leaf = f"tick{t:03d}-{name}-player-{idx}.png"
    path = root / leaf
    assert sha(path) == ledger[t]["files_sha256"][leaf]
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main(a):
    scores = {m: read(HERE / (m + "-joint-q50-scores.json")) for m in ["fresh", "prior"]}
    assert scores["fresh"]["source_sha256"] == scores["prior"]["source_sha256"]
    assert scores["fresh"]["supports_sha256"] == scores["prior"]["supports_sha256"]
    assert all(s["frames"] == 240 and float(s["native_fps"]) == 25 for s in scores.values())
    idx = {m: {q["tick"]: q for q in s["samples"]} for m, s in scores.items()}
    boxes = {t: idx["fresh"][t]["source_crop_xyxy"] for t in TICKS}
    assert all(idx["prior"][t]["source_crop_xyxy"] == boxes[t] for t in TICKS)
    neural = read(HERE / "neural-q36-body-and-ball.json")["arm"]
    limit = scores["prior"]["arm"]["bytes"]
    feasible = [
        row for row in read(HERE / "ordinary-near-summary.json") if row["arm"]["bytes"] <= limit
    ]
    ordinary = min(feasible, key=lambda row: row["mean_dists"])["arm"]
    assert (
        ordinary["kbps"] <= scores["prior"]["arm"]["kbps"]
        and neural["kbps"] <= scores["prior"]["arm"]["kbps"]
    )
    av1 = native_parts(ordinary, boxes)
    dcvc = native_parts(neural, boxes)
    prices = read(HERE / "two-shot-price.json")
    titles = [
        "Source",
        f"Warm-budget AV1\n{ordinary['kbps']:.2f} kbps",
        f"Warm-budget DCVC-UF\n{neural['kbps']:.2f} kbps",
        f"Refresh on return\n{prices['send_new']['return_kbps']:.2f} kbps",
        f"Reuse on return\n{prices['reuse']['return_kbps']:.2f} kbps",
    ]
    fig, axs = plt.subplots(4, 5, figsize=(8.8, 5.1), layout="constrained")
    for row, (t, i) in enumerate([(t, i) for t in TICKS for i in (0, 1)]):
        s = true_crop(a.fresh, idx["fresh"], t, "source", i)
        assert np.array_equal(s, true_crop(a.prior, idx["prior"], t, "source", i))
        frames = [
            s,
            av1[t, i],
            dcvc[t, i],
            true_crop(a.fresh, idx["fresh"], t, "display", i),
            true_crop(a.prior, idx["prior"], t, "display", i),
        ]
        assert all(im.shape == s.shape for im in frames)
        for col, im in enumerate(frames):
            ax = axs[row, col]
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"{t / 25:.1f} s\n{('Far', 'Near')[i]}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=4,
                    fontsize=9,
                )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    for ext in ["pdf", "png"]:
        fig.savefig(a.out.with_suffix("." + ext), dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["fresh", "prior", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
