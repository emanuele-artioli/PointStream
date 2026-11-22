"""Plot the reviewed development rallies and the independently selected broadcasts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CARDS = HERE.parents[2] / "experiments"


def plot():
    reviewed = json.loads((CARDS / "reviewed-foreground-20261122/figure-data.json").read_text())
    external = json.loads(
        (CARDS / "shanghai-joint-low-anchors-20261122/comparison-final-neural.json").read_text()
    )["measured_rungs"]
    sydney = json.loads(
        (CARDS / "sydney-joint-foreground-20261122/comparison-matched-rung-70.json").read_text()
    )["measured_rungs"]
    names = ("PointStream", "AV1", "DCVC-UF")
    rows = []
    for match, ceiling in [("Nadal–Kyrgios", 21.62), ("Alcaraz–Hurkacz", 20.41)]:
        scores = reviewed["per_shot"][match]
        rows.append((f"{match}  ·  {ceiling:.2f} kbps", [scores[n]["foreground"] for n in names]))
    rows.append(("Mean of the two rallies", [reviewed["pooled"][n]["foreground"] for n in names]))
    for match, arm in (
        ("Sinner–Shelton", external["50"]),
        ("Sinner–Shelton", external["70"]),
        ("Safiullin–Shapovalov", sydney["70"]),
    ):
        budget = arm["cases"]["cached"]
        controls = [budget["controls"][family]["best"] for family in ("ordinary", "neural")]
        assert all(c["bytes"] <= budget["bytes"] for c in controls)
        scores = [candidate["scores"]["foreground"] for candidate in (arm["joint"], *controls)]
        rows.append((f"{match}  ·  {budget['kbps']:.2f} kbps", scores))

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False}):
        fig, ax = plt.subplots(figsize=(7.0, 3.52))
        fig.subplots_adjust(left=0.32, right=0.98, bottom=0.19, top=0.83)
        ys = [5.22, 4.35, 3.48, 1.87, 1.00, 0.13]
        ax.axhspan(3.17, 3.91, color="#edf3f3", zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", lw=0.6, color="#dce1e3")
        for i, (name, color, marker) in enumerate(
            zip(names, ("#007481", "#a54e09", "#684599"), ("o", "s", "D"))
        ):
            values = [scores[i] for _, scores in rows]
            ax.scatter(values, ys, label=name, c=color, marker=marker, s=27, zorder=3)
            for value, y in zip(values, ys):
                ax.annotate(
                    f"{value:.3f}",
                    (value, y),
                    xytext=(0, 4.3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color=color,
                )
        ax.set(
            xlim=(0.254, 0.482),
            ylim=(-0.21, 5.84),
            yticks=ys,
            yticklabels=[label for label, _ in rows],
            xlabel="Foreground DISTS (lower is better)",
        )
        ax.tick_params(axis="y", length=0, pad=7, labelsize=8.7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        for label, y in [("Development rallies", 5.86), ("Independent rallies", 2.53)]:
            ax.text(
                0.015,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                va="bottom",
                weight="bold",
                fontsize=8,
            )
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.51, 1.12), ncol=3, frameon=False, fontsize=8
        )
        for suffix in ("pdf", "png"):
            fig.savefig(HERE / f"complete-foreground-matches.{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    plot()
