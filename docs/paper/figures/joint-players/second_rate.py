"""Plot the physical player rate–quality operating points on the second rally."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
CARDS = ROOT.parents[2] / "experiments"
HERE = CARDS / "al028-neural-display-20261122"


def load(path):
    return json.loads(path.read_text())


def joints(filename, rates):
    d = load(HERE / filename)
    return [
        (rates[arm], mean(p["dists"] for r in d["rows"] if r["arm"] == arm for p in r["body"]))
        for arm in rates
    ]


def main():
    ordinary = load(HERE / "ordinary-three-size-curves.json")["points"]
    neural = load(CARDS / "al028-q36-export-20261122/body-scores.json")
    neural_points = sorted((neural["rates_kbps"][k], v) for k, v in neural["mean_dists"].items())
    bills = {}
    for k in (11, 21):
        bill = load(CARDS / f"joint-replication-feasibility-20261122/al028-joint-{k}kbps.json")
        bills[f"joint-{k}"] = bill["ideal_prewarmed_kbps"]
    bills["joint-20"] = load(HERE / "common-budget.json")["selected"]["joint"]["kbps"]
    joint_points = joints("three-joint-rungs.json", bills)
    high = load(HERE / "upper-joint-rung.json")
    arms = {r["arm"] for r in high["rows"]}
    assert len(arms) == 1
    joint_points += joints(
        "upper-joint-rung.json",
        {arms.pop(): load(HERE / "common-budget-q36.json")["selected"]["joint"]["kbps"]},
    )
    joint_points.sort()
    fig, ax = plt.subplots(figsize=(8.4, 4.35))
    points = sorted((r["kbps"], r["player_dists"]) for r in ordinary)
    line = []
    for x, y in points:
        if not line or y < line[-1][1]:
            line.append((x, y))
    ax.scatter(*zip(*points), s=10, c="#ba6119", alpha=0.42, zorder=2)
    for coords, label, color, marker, style in [
        (line, "Ordinary AV1, best observed", "#a64a0a", "s", "-"),
        (neural_points, "DCVC-UF", "#654594", "D", "--"),
        (joint_points, "Joint PointStream, cached", "#00788a", "o", "-"),
    ]:
        ax.plot(
            *zip(*coords),
            marker=marker,
            color=color,
            ls=style,
            label=label,
            linewidth=1.8,
            markersize=4.5,
            zorder=3,
        )
    ax.set(
        xlim=(8, 43),
        xlabel="Total delivered bitrate (kbps)",
        ylabel="Player-region DISTS   ·   lower is better",
    )
    visible = [y for x, y in points + joint_points + neural_points if 8 <= x <= 43]
    ax.set_ylim(min(visible) - 0.025, max(visible) + 0.03)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.17)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.23, top=0.82)
    fig.text(0.045, 0.935, "When does sending player motion help?", fontsize=16, fontweight="bold")
    fig.text(
        0.045,
        0.875,
        "Alcaraz–Hurkacz · 9.6 s at 25 fps · both players pooled on 12 source-model mask pairs",
        fontsize=9,
    )
    fig.text(
        0.045,
        0.065,
        "Each marker is a delivered stream. AV1 used three resolutions. Lines connect measured settings.",
        fontsize=8.5,
    )
    fig.text(
        0.045,
        0.024,
        "The joint stream includes both players and the ball. Charging its 157,517-byte initialization adds 131.26 kbps on this shot.",
        fontsize=8.5,
    )
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"second-rally-rate-quality.{ext}", dpi=180)
    plt.close(fig)
    (ROOT / "second-rally-rates.json").write_text(
        json.dumps(dict(joint=joint_points, neural=neural_points, ordinary_frontier=line), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
