"""Create the paper's human-reviewed joint foreground panels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CARD = HERE.parents[2] / "experiments/reviewed-foreground-20261122"
DATA = json.loads((CARD / "figure-data.json").read_text())
STYLE = {"PointStream": ("#007481", "o"), "AV1": ("#a74e0a", "s"), "DCVC-UF": ("#6d469b", "D")}
plt.rcParams.update(
    {"pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def rallies():
    fig, ax = plt.subplots(figsize=(5.95, 2.65))
    for mode, (color, marker) in STYLE.items():
        for y, row in enumerate(
            [DATA["pooled"], DATA["per_shot"]["Alcaraz–Hurkacz"], DATA["per_shot"]["Nadal–Kyrgios"]]
        ):
            x = row[mode]["foreground"]
            ax.scatter(x, y, color=color, marker=marker, s=40, zorder=4)
            ax.annotate(
                f"{x:.3f}",
                (x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    ax.set(
        ylim=(-0.32, 2.54),
        xlim=(0.25, 0.484),
        yticks=[0, 1, 2],
        yticklabels=[
            "Equal-match mean",
            "Alcaraz–Hurkacz\n11 visible instants",
            "Nadal–Kyrgios\n12 visible instants",
        ],
        xlabel="Foreground DISTS   (lower is better)",
    )
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#dee3e5", lw=0.65)
    ax.axhspan(-0.32, 0.40, color="#edf3f3", zorder=0)
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="none", label=s) for s, (c, m) in STYLE.items()
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.59, 1.01),
        columnspacing=1.4,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.81, bottom=0.23)
    save(fig, "two-rally-foreground")


def rate():
    fig, ax = plt.subplots(figsize=(5.95, 2.95))
    for mode in ("AV1", "DCVC-UF", "PointStream"):
        col, marker = STYLE[mode]
        pts = sorted(DATA["alcaraz_points"][mode], key=lambda p: p["kbps"])
        if mode == "AV1":
            frontier = []
            for p in pts:
                if not frontier or p["foreground"] < frontier[-1]["foreground"]:
                    frontier.append(p)
            ax.scatter(
                [p["kbps"] for p in pts],
                [p["foreground"] for p in pts],
                s=8,
                marker="s",
                color=col,
                alpha=0.31,
            )
            pts = frontier
        elif mode == "DCVC-UF":
            archive = [p for p in pts if p["archived_only"]]
            ax.scatter(
                [p["kbps"] for p in archive],
                [p["foreground"] for p in archive],
                s=24,
                marker=marker,
                edgecolor=col,
                facecolor="white",
            )
            pts = [p for p in pts if not p["archived_only"]]
        ax.plot(
            [p["kbps"] for p in pts],
            [p["foreground"] for p in pts],
            color=col,
            marker=marker,
            markersize=4,
            ls="--" if mode == "DCVC-UF" else "-",
            label=mode + (" (cached)" if mode == "PointStream" else ""),
        )
    ax.set(
        xlabel="Delivered bitrate (kbps)",
        ylabel="Foreground DISTS (lower is better)",
        xlim=(7.8, 42.3),
        ylim=(0.24, 0.60),
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.23)
    save(fig, "al028-foreground-rd")


if __name__ == "__main__":
    rallies()
    rate()
