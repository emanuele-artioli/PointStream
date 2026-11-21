"""Compare decoded RGB segmentation and direct receiver masks on identical bodies."""

from pathlib import Path
import json
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "experiments/starved-player-segmentation-20261122/result.json"
ORDER = [
    ("source", "Source RGB"),
    ("plain", "Ordinary AV1 RGB"),
    ("roi", "ROI AV1 RGB"),
    ("pointstream", "Joint PointStream RGB"),
    ("receiver_mask", "Receiver silhouette"),
]
STYLES = [
    ("all", "All 24 body labels", "#16375a", "o"),
    ("unprompted", "6 drawn without proposals", "#d57228", "D"),
    ("assisted", "18 reviewed proposals", "#00816c", "^"),
]


def main():
    rows = json.loads(DATA.read_text())["means"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, (s, label, color, marker) in enumerate(STYLES):
        ax.scatter(
            [rows[k][s]["mean"] for k, _ in ORDER],
            [i + (j - 1) * 0.16 for i in range(len(ORDER))],
            s=49,
            marker=marker,
            c=color,
            label=label,
            zorder=3,
        )
    ax.set_yticks(range(len(ORDER)), [label for _, label in ORDER])
    ax.set_ylim(len(ORDER) - 0.45, -0.6)
    ax.set_xlim(0.55, 0.95)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel("Mean body-mask IoU (higher is better)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "How accurately are players segmented?",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.27, top=0.83)
    for f in ("png", "pdf"):
        fig.savefig(HERE / f"decoded-masks.{f}", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
