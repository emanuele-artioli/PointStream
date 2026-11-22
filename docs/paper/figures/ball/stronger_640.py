"""Three-match sparse-label ball DISTS versus complete wrapped wire rate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
old = json.loads(
    (ROOT / "docs/experiments/starved-ball-transport-20261122/result.json").read_text()
)["rows"]
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True, constrained_layout=True)
matches = {
    "209": "Rotterdam 2024 · 9 labels",
    "269": "Washington 2023 · 12 labels",
    "235": "Indian Wells 2024 · 11 labels",
}
for ax, match in zip(axes, ("209", "269", "235"), strict=True):
    source = "match" + match + "_000"
    controls = ROOT / f"docs/experiments/ball-640-controls-20261122/match{match}-result.json"
    subsets = [
        (
            "Tracked ball tiles",
            "#0072B2",
            "o",
            [x for x in old if x["source"] == source and x["mode"] == "tracked"],
        ),
        (
            "Existing plain/ROI AV1",
            "#999999",
            "x",
            [x for x in old if x["source"] == source and x["mode"] in ("plain", "roi")],
        ),
    ]
    for label, color, marker, rows in subsets:
        rows = [x for x in rows if 25 <= x["wire_kbps"] <= 70]
        ax.scatter(
            [x["wire_kbps"] for x in rows],
            [x["ball_dists"] for x in rows],
            label=label,
            c=color,
            marker=marker,
            s=32,
            alpha=0.85,
        )
    if controls.exists():
        rows = json.loads(controls.read_text())["selected"]
        refine = (
            ROOT / f"docs/experiments/ball-640-controls-20261122/match{match}-refine-result.json"
        )
        if refine.exists():
            rows += json.loads(refine.read_text())["selected"]
        for mode, label, color, marker in [
            ("plain", "640×360 AV1", "#D55E00", "s"),
            ("roi", "640×360 ROI AV1", "#009E73", "^"),
        ]:
            data = [x for x in rows if x["mode"] == mode and 25 <= x["kbps"] <= 70]
            ax.scatter(
                [x["kbps"] for x in data],
                [x["mean_dists"] for x in data],
                label=label,
                c=color,
                marker=marker,
                s=44,
                edgecolors="black",
                linewidths=0.35,
            )
    else:
        ax.text(0.5, 0.5, "640 controls pending", ha="center", transform=ax.transAxes)
    ax.axvline(60.6, color="black", linestyle=":", linewidth=1)
    ax.set(xlim=(25, 70), xlabel="Complete wrapped bitrate (kbps)", title=matches[match])
    ax.grid(alpha=0.22)
axes[0].set_ylabel("Ball-crop DISTS ↓")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
fig.suptitle("Ball-crop fidelity across three broadcasts")
out = Path(__file__).with_name("ball-starved-640-controls")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=180)
