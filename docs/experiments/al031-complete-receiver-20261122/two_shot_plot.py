"""Show the physical price of two source windows including first-use state and cache refresh."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def draw():
    result = json.loads((HERE / "complete-two-shot-comparison.json").read_text())
    cases = result["cases"]
    assert all("mean_pointstream_dists" in row for row in cases.values())
    points = [r for r in result["ordinary_measured_frontier"] if r["kbps"] <= 175]
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.9, 3.55), layout="constrained")
    ax.plot(
        [r["kbps"] for r in points],
        [r["mean_dists"] for r in points],
        color="#00587a",
        marker=".",
        markersize=4,
        lw=1.5,
        label="SVT-AV1 achievable packet pairs",
    )
    for mode, label, marker, color in [
        ("reuse", "PointStream, reuse earlier appearance", "D", "#713493"),
        ("send_new", "PointStream, refresh appearance", "^", "#a54a15"),
    ]:
        case = cases[mode]
        ax.scatter(
            case["rate"],
            case["mean_pointstream_dists"],
            marker=marker,
            color=color,
            s=47,
            label=label,
            zorder=4,
        )
        best = case["best_ordinary"]
        ax.plot(
            [best["two_shot_kbps"], case["rate"]],
            [best["mean_dists"], case["mean_pointstream_dists"]],
            ls=":",
            c=color,
            lw=1.3,
        )
    ax.set_xlabel("Delivered bitrate over both 9.6-second shots (kbps)")
    ax.set_ylabel("Mean player DISTS (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#ced5d9", lw=0.6)
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"two-shot-price-quality.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    draw()
