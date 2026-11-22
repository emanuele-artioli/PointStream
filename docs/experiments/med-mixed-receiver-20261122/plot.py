"""Plot measured complete-delivery player fidelity on the mixed-mode W7 passage."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tennis.k6b_side_feasibility import hull

H = Path(__file__).resolve().parent


def main(a):
    controls = json.loads(a.controls.read_text())
    trials = json.loads(a.joint.read_text())
    ordinary = {
        e["arm"]["stream_sha256"]: e
        for k, rows in controls.items()
        if k.startswith("ordinary")
        for e in rows
        if e["arm"]["kbps"] <= 32
    }
    neural = [r for k, rows in controls.items() if k.startswith("neural") for r in rows]
    data = {"SVT-AV1": list(ordinary.values()), "DCVC-UF": neural, "PointStream": trials}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for label, color, marker in [
        ("SVT-AV1", "#24648c", "o"),
        ("DCVC-UF", "#8d457e", "s"),
        ("PointStream", "#ae4a17", "D"),
    ]:
        pts = sorted((e["arm"]["kbps"], e["mean_dists"]) for e in data[label])
        if not pts:
            continue
        if label == "SVT-AV1":
            line = [(r, -s) for r, s in hull([(r, -v) for r, v in pts])]
            ax.scatter(*zip(*pts), s=15, color=color, alpha=0.25, linewidth=0)
        else:
            line = pts
        ax.plot(
            *zip(*line),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=label,
            zorder=3,
        )
    ax.set_xlim(14, 32)
    ax.set_xlabel("Delivered bitrate (kbps), PointStream appearance cached")
    ax.set_ylabel("Masked player DISTS (lower is better)")
    ax.legend(
        frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.2
    )
    ax.grid(axis="y", linestyle=":", color="#abb3b8", alpha=0.8)
    ax.set_axisbelow(True)
    for suffix in ("pdf", "png"):
        fig.savefig(a.out.with_suffix("." + suffix), dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--joint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
