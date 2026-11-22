"""Plot preselected Basel holdout ball DISTS against wrapped total bitrate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
result = json.loads(
    (ROOT / "docs/experiments/ball-basel-confirmation-20261122/result.json").read_text()
)
fig, ax = plt.subplots(figsize=(7.3, 4.3), constrained_layout=True)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
colors = {"plain": "#D55E00", "roi": "#009E73", "tile": "#0072B2"}
markers = {320: "o", 640: "s", 1280: "^"}
for row in result["rows"]:
    available = row["kbps"] <= result["cap"]
    ax.scatter(
        row["kbps"],
        row["dists"],
        s=75,
        marker=markers[row["width"]],
        facecolor=colors[row["mode"]] if available else "none",
        edgecolor=colors[row["mode"]],
        linewidth=1.4,
        zorder=3,
    )
ax.axvline(result["cap"], color="black", linestyle=":", linewidth=1)
ax.set(
    title="Basel: separately held-out match",
    xlabel="Complete wrapped bitrate (kbps)",
    ylabel="Ball-crop DISTS ↓",
)
ax.grid(alpha=0.22)
ax.legend(
    handles=[
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor=colors[m],
            markeredgecolor=colors[m],
            label=label,
        )
        for m, label in [("plain", "Plain AV1"), ("roi", "ROI AV1"), ("tile", "Tracked tiles")]
    ]
    + [
        Line2D(
            [],
            [],
            color="none",
            marker=marker,
            markerfacecolor="#888888",
            markeredgecolor="#888888",
            label=f"{width}×{width * 9 // 16}",
        )
        for width, marker in markers.items()
    ]
    + [
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            label="Above 60.6 kbps cap",
        )
    ],
    frameon=False,
    fontsize=8,
)
out = Path(__file__).with_name("basel-holdout-rates")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=180)
