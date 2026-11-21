"""Plot body and complement measures at the modeled-warm operating point."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dcvc", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if data["count"] != 12 or len(data["rows"]) != 36:
        raise ValueError("Expected 12 sampled frames for each of three arms")
    names = ["70/30 body + complement", "Body union", "Complement", "Full frame"]
    keys = ["weighted_body", "body", "nonbody", "full"]
    arms = [
        ("plain", "SVT-AV1 · 66.105 kbps", "#19639b", "o"),
        ("roi", "AV1 ROI · 64.853 kbps", "#a96a04", "s"),
        ("pointstream", "PointStream · 65.473 kbps, modeled warm", "#087e6b", "D"),
    ]
    if args.dcvc:
        neural = json.loads(args.dcvc.read_text())
        if neural["dcvc"]["qp"] != 24 or neural["dcvc"]["historical_frame_hashes_matched"] != 240:
            raise ValueError("The DCVC comparison is not the verified QP24 decode")
        data["means"]["dcvc"] = neural["means"]["pointstream"]
        arms.insert(2, ("dcvc", "DCVC-UF · 49.778 kbps", "#7d4f91", "^"))
    for metric, xlabel in (("dists", "Regional DISTS  ↓"), ("lpips", "Spatial LPIPS  ↓")):
        fig, ax = plt.subplots(figsize=(9.7, 6.3))
        fig.suptitle("A starved-rate player and background tradeoff", x=.06, ha="left",
                     fontsize=15, weight="bold")
        fig.text(.06, .86, "One rally · 12 source-reviewed ticks · bodies define the foreground", fontsize=9)
        for j, (arm, label, color, marker) in enumerate(arms):
            values = [data["means"][arm][metric][key] for key in keys]
            offset = (j - (len(arms) - 1) / 2) * .16
            for i, value in enumerate(values):
                y = len(keys) - 1 - i + offset
                ax.scatter(value, y, color=color, marker=marker, s=52, label=label if i == 0 else None, zorder=3)
                ax.annotate(f"{value:.3f}", (value, y), xytext=(6, 0), textcoords="offset points",
                            va="center", color=color, fontsize=8)
        ax.set_yticks(range(len(names)), list(reversed(names)))
        ax.set_ylim(-.5, len(names) - .5)
        ax.set_xlim(0, .8 if metric == "lpips" else .52)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.53, .075),
                   ncol=2, frameon=False, fontsize=8.5)
        fig.text(.06, .01, "PointStream cold delivery is 469.335 kbps. Cached references are assumed paid and valid.", fontsize=8.5)
        fig.tight_layout(rect=(0, .23, 1, .83))
        fig.savefig(args.output.with_name(args.output.name + "-" + metric).with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(args.output.with_name(args.output.name + "-" + metric).with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
