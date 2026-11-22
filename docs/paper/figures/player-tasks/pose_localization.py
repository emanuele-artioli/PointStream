"""Plot person-localization scores on the two human-reviewed source rallies."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parents[2] / "experiments" / "al028-machine-masks-20261122"
ARMS = ["Ordinary AV1", "PointStream poses", "DCVC-UF"]
CASES = [
    ("Nadal serve", "nadal-boxes", ["ordinary", "pose", "neural"], "o", "#005b8a"),
    ("Alcaraz rally", "al028-boxes", ["av1_320_q52_75", "pose", "dcvc_q24"], "D", "#a44314"),
]


def main():
    fig, ax = plt.subplots(figsize=(5.9, 2.15), layout="constrained")
    handles = []
    for name, filename, arms, marker, color in CASES:
        scores = json.loads((RECEIPTS / f"{filename}.json").read_text())["summary"]
        for y, arm in enumerate(arms):
            value = scores[arm]["mean_iou"]
            assert scores[arm]["n"] == 24
            dy = 0.13 if marker == "o" else -0.13
            ax.scatter(value, y + dy, color=color, marker=marker, s=39, zorder=3)
            ax.text(value + 0.024, y + dy, f"{value:.2f}", va="center", fontsize=8, color=color)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", label=name))
    ax.set(yticks=range(len(ARMS)), yticklabels=ARMS, xlim=(0, 1), ylim=(2.45,-0.45),
           xlabel="Mean overlap with reviewed person boxes (IoU)")
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,1.01), ncol=2,
              frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.24)
    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "pose-localization.pdf", metadata={"CreationDate":None})


if __name__ == "__main__":
    main()
