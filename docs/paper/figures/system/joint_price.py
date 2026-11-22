"""Relate completely serialized reference bytes to the duration of active play."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments/joint-trajectory-20261122"


def main():
    d = json.loads((EXP / "result.json").read_text())["streams"]
    fig = plt.figure(figsize=(8.9, 5.5))
    ax = fig.add_axes([0.13, 0.40, 0.82, 0.44])
    durations = np.geomspace(4.8, 600, 250)
    colors = ("#00708a", "#813773")
    for (name, r), color in zip(d.items(), colors, strict=True):
        costs = [r[k] for k in ("base_bytes", "pose_bytes", "ball_bytes", "framing_bytes")]
        assert sum(costs) == r["bytes"] and r["bytes"] + r["reference_bytes"] == r["cold_bytes"]
        total = r["kbps"] + r["reference_bytes"] * 8 / (1000 * durations)
        assert abs(total[0] - r["cold_kbps"]) < 1e-9
        ax.plot(durations, total, color=color, lw=2.5, label=f"{r['kbps']:.3f} kbps ongoing")
        ax.scatter(durations[0], total[0], s=32, color=color)
    for price in (33, 65):
        ax.axhline(price, color="#677079", lw=0.9, linestyle="--")
        ax.text(
            650,
            price,
            f"{price} kbps",
            va="center",
            ha="right",
            fontsize=8,
            backgroundcolor="white",
        )
    low = d["20"]
    crossing = low["reference_bytes"] * 8 / 1000 / (33 - low["kbps"])
    ax.scatter(crossing, 33, color=colors[0], zorder=5)
    ax.annotate(
        f"{crossing:.0f} s to fit 33 kbps",
        xy=(crossing, 33),
        xytext=(-105, -18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": colors[0]},
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(4.4, 690),
        ylim=(18, 530),
        xlabel="Active seconds using the same references",
        ylabel="Average total bitrate (kbps)",
    )
    ax.set_xticks([4.8, 10, 30, 60, 180, 600], ["4.8", "10", "30", "60", "180", "600"])
    ax.set_yticks([20, 33, 65, 125, 250, 500], ["20", "33", "65", "125", "250", "500"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(
        "How long must a player reference remain useful?",
        x=0.035,
        y=0.97,
        ha="left",
        weight="bold",
        fontsize=15,
    )
    fig.text(
        0.035,
        0.895,
        "The first rally pays 244,339 bytes for both lossless images, their framing and reusable receiver metadata.",
        fontsize=9,
    )
    tab = fig.add_axes([0.045, 0.08, 0.91, 0.18])
    tab.axis("off")
    columns = [
        "Running rate",
        "Clean AV1 base",
        "Player poses",
        "Ball symbols",
        "Framing",
        "Total per 4.8 s",
    ]
    rows = [
        [
            f"{r['kbps']:.3f} kbps",
            *[
                f"{r[k]:,} B"
                for k in ("base_bytes", "pose_bytes", "ball_bytes", "framing_bytes", "bytes")
            ],
        ]
        for r in d.values()
    ]
    table = tab.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#ccd1d3")
        if row == 0:
            cell.set_facecolor("#e8ecee")
    fig.text(
        0.035,
        0.025,
        "Each curve prices repeated use at a measured running rate. The tested rally lasts 4.8 s; later cache validity is a scenario.",
        fontsize=8.5,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("joint-price." + ext), dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
