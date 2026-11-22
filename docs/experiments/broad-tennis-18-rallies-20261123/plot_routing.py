"""Draw the detected court and conventional intervals of the six frozen points."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def run():
    report = json.loads((HERE / "route-summary.json").read_text())
    labels = [
        "Williams–Sharapova · 14",
        "Williams–Sharapova · 41",
        "Williams–Sharapova · 68",
        "Djokovic–Tsonga · 22",
        "Djokovic–Tsonga · 65",
        "Djokovic–Tsonga · 108",
    ]
    colors = {"court": "#167480", "non_court": "#ac6543"}
    hatches = {"court": "", "non_court": "///"}
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7, 2.75))
        fig.subplots_adjust(left=0.28, right=0.97, bottom=0.20, top=0.81)
        for i, row in enumerate(report["rows"]):
            fps = row["native_fps"][0] / row["native_fps"][1]
            for s in row["shots"]:
                state = s["predicted"]
                ax.barh(
                    i,
                    (s["stop"] - s["start"]) / fps,
                    left=s["start"] / fps,
                    height=0.52,
                    color=colors[state],
                    hatch=hatches[state],
                    edgecolor="white",
                    linewidth=0.45,
                )
        ax.set(
            xlim=(0, 11),
            yticks=range(6),
            yticklabels=labels,
            xlabel="Seconds from annotated point start",
        )
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce1e3", lw=0.6)
        entries = [
            Patch(facecolor=colors[k], edgecolor="white", hatch=hatches[k], label=label)
            for k, label in [("court", "Court eligible"), ("non_court", "Conventional route")]
        ]
        ax.legend(
            handles=entries, loc="lower center", bbox_to_anchor=(0.46, 1.01), ncol=2, frameon=False
        )
        for suffix in ("pdf", "png"):
            fig.savefig(HERE / f"held-out-court-routing.{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    run()
