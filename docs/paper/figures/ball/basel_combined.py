"""Separate Basel holdout: visual fidelity and ball location for machines."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
d = ROOT / "docs/experiments/ball-basel-confirmation-20261122"
v = json.loads((d / "result.json").read_text())
m = json.loads((d / "machine-result.json").read_text())
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, (left, right) = plt.subplots(
    1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.1]}, constrained_layout=True
)
colors = {"plain": "#D55E00", "roi": "#009E73", "tile": "#0072B2"}
markers = {320: "o", 640: "s", 1280: "^"}
for row in v["rows"]:
    inside = row["kbps"] <= v["cap"]
    left.scatter(
        row["kbps"],
        row["dists"],
        s=70,
        marker=markers[row["width"]],
        facecolor=colors[row["mode"]] if inside else "none",
        edgecolor=colors[row["mode"]],
        linewidth=1.3,
        zorder=3,
    )
left.axvline(60.6, color="black", linestyle=":", linewidth=1)
left.set(
    title="(a) Ball appearance",
    xlabel="Complete wrapped bitrate (kbps)",
    ylabel="Ball-crop DISTS ↓",
)
left.grid(alpha=0.2)
handles = [
    Line2D(
        [],
        [],
        color="none",
        marker="o",
        markerfacecolor=colors[k],
        markeredgecolor=colors[k],
        label=label,
    )
    for k, label in [("plain", "Plain AV1"), ("roi", "ROI AV1"), ("tile", "Tracked tiles")]
]
handles += [
    Line2D(
        [],
        [],
        color="none",
        marker=marker,
        markerfacecolor="#888",
        markeredgecolor="#888",
        label=f"{w}×{w * 9 // 16}",
    )
    for w, marker in markers.items()
]
handles += [
    Line2D(
        [],
        [],
        color="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        label="Above 60.6 kbps",
    )
]
left.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
by = {row["mode"]: row for row in m["results"]}
items = [
    ("Paid tile centers", by["tile"], "direct_native_error", "#0072B2"),
    ("WASB on tile", by["tile"], "native_error", "#56B4E9"),
    ("WASB on 1280 ROI", by["roi"], "native_error", "#009E73"),
    ("WASB on 1280 plain", by["plain"], "native_error", "#D55E00"),
    ("WASB on source", by["source"], "native_error", "#777777"),
]
y = np.arange(len(items))
hits10 = []
hits5 = []
for _, row, key, _ in items:
    visible = [x for x in row["rows"] if x["visibility"] == 1]
    hits10.append(sum(x[key] is not None and x[key] <= 10 for x in visible))
    hits5.append(sum(x[key] is not None and x[key] <= 5 for x in visible))
right.barh(y, hits10, color=[x[3] for x in items], height=0.56)
for i, (a, b) in enumerate(zip(hits10, hits5, strict=True)):
    right.text(max(a, 0.12) + 0.12, i, f"{a}/12   ({b}/12 at 5 px)", va="center", fontsize=8)
right.set(
    yticks=y,
    yticklabels=[x[0] for x in items],
    xlim=(0, 16.8),
    xticks=[0, 3, 6, 9, 12],
    xlabel="Visible labeled frames localized within 10 native pixels",
    title="(b) Ball locations for machines",
)
right.invert_yaxis()
right.grid(axis="x", alpha=0.2)
fig.suptitle("Basel holdout: visual quality and machine localization")
out = Path(__file__).with_name("basel-holdout-visual-machine")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=180)
