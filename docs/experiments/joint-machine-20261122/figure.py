"""Count correctly localized visible balls on the same twelve labeled instants."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[1] / "paper/figures/ball"


def main():
    data = json.loads((HERE / "result.json").read_text())
    raw = json.loads((HERE.parent / "ball-machine-twenty-20261122/result.json").read_text())
    assert data["frames"] == raw["frames"] == 240 and data["ordinary_prior_agreement"]
    assert data["label_sha256"] == raw["labels"]
    names = [
        "Detector on source",
        "Detector on AV1",
        "Detector on DCVC-UF",
        "Detector on PointStream",
        "PointStream coordinates",
    ]
    costs = data["costs_bytes"]
    rows = [raw["results"]["source"]["rows"]] + [
        data["results"][k]["rows"] for k in ("ordinary", "neural", "joint")
    ]
    rows.append([{"tick": int(t), "error": v} for t, v in data["direct_errors"].items()])
    rates = [None, costs["ordinary"], costs["neural"], costs["joint"], costs["joint"]]
    labels = [
        name + ("   ·   " + f"{size / 600:.2f} kbps" if size is not None else "")
        for name, size in zip(names, rates, strict=True)
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    positions = np.arange(len(rows))
    all_counts = []
    for y, entries in zip(positions, rows, strict=True):
        assert sorted(r["tick"] for r in entries) == list(range(10, 240, 20))
        counts = [
            sum(r["error"] is not None and r["error"] <= limit for r in entries)
            for limit in (5, 10)
        ]
        all_counts.append(counts)
        ax.plot(counts, [y, y], color="#a7afb5", zorder=1)
    for col, (marker, color, label) in enumerate(
        [("o", "#11466b", "Within 5 pixels"), ("D", "#d16c12", "Within 10 pixels")]
    ):
        ys = positions + (col - 0.5) * 0.14
        vals = [v[col] for v in all_counts]
        ax.scatter(vals, ys, marker=marker, color=color, s=46, label=label, zorder=3)
    ax.invert_yaxis()
    ax.set(
        yticks=positions,
        yticklabels=labels,
        xticks=range(0, 13, 2),
        xlim=(-0.45, 12.45),
        xlabel="Correct locations out of 12 visible balls",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="y", length=0)
    ax.spines[["right", "top", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.26, top=0.75)
    fig.suptitle(
        "How often can a receiver locate the ball?",
        x=0.04,
        y=0.97,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.04,
        0.87,
        "One 4.8-second serve. Every method is evaluated on the same source marks at 1920 × 1080.",
        fontsize=9,
    )
    fig.legend(loc="lower left", bbox_to_anchor=(0.35, 0.077), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.04,
        0.035,
        f"The detector reads decoded RGB. The coordinates and joint display use the same packet; cold start costs {costs['joint_cold'] / 600:.2f} kbps.",
        fontsize=8.7,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / ("joint-localization." + suffix), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
