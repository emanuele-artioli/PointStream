"""Plot per-shot foreground-quality differences during the two routed intervals."""

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replicate import controls, load

HERE = Path(__file__).resolve().parent


def run():
    fields = ["active_dists", "fallback_dists", "mean_dists"]
    shots = {}
    for shot in ["000", "019", "024"]:
        p = load(shot + "-joint-q60.json")
        bank = controls(shot)
        effects = {}
        for name in ["ordinary", "neural"]:
            affordable = [
                r
                for k, rows in bank.items()
                if k.startswith(name)
                for r in rows
                if r["arm"]["bytes"] <= p["arm"]["bytes"]
            ]
            select = min(affordable, key=lambda row: row["mean_dists"])
            effects[name] = dict(
                control=select["arm"]["name"],
                control_kbps=select["arm"]["kbps"],
                by_interval={f: p[f] - select[f] for f in fields},
            )
        shots[shot] = dict(joint_kbps=p["arm"]["kbps"], effects=effects)
    (HERE / "mode-quality-by-shot.json").write_text(json.dumps(shots, indent=2) + "\n")
    plt.rcParams.update(
        {"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
    ax.axhline(0, color="#58616b", lw=1)
    for name, label, color, marker, offset in [
        ("ordinary", "SVT-AV1", "#185a83", "o", -0.10),
        ("neural", "DCVC-UF", "#a74716", "s", 0.10),
    ]:
        xs = [i + offset for i in range(3)]
        rows = [[r["effects"][name]["by_interval"][f] for f in fields] for r in shots.values()]
        for seq in rows:
            ax.plot(xs, seq, color=color, lw=0.75, alpha=0.32, marker=marker, ms=3.5)
        ax.plot(
            xs,
            [mean(values) for values in zip(*rows, strict=True)],
            marker=marker,
            color=color,
            lw=2.1,
            ms=6.5,
            label=label,
        )
    ax.set_xticks([0, 1, 2], ["Player mode", "Ordinary fallback", "Complete shot"])
    ax.set_ylabel("Player DISTS difference\n(PointStream minus control)")
    ax.set_xlim(-0.32, 2.32)
    ax.set_ylim(-0.28, 0.21)
    ax.grid(axis="y", color="#c9cdd1", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    for ext in ["pdf", "png"]:
        fig.savefig(HERE / ("mode-quality-difference." + ext), dpi=185)
    plt.close(fig)


if __name__ == "__main__":
    run()
