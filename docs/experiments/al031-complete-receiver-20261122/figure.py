"""Draw the physical price and measured player quality of fresh and returned appearance."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from experiments.tennis.k6b_side_feasibility import hull

ROOT = Path(__file__).resolve().parent


def read(n):
    return json.loads((ROOT / n).read_text())


def xy(r):
    return r["arm"]["kbps"], r["mean_dists"]


def curve(ax, rows, label, color, marker):
    dots = sorted(set(xy(r) for r in rows))
    front = hull([(rate, -d) for rate, d in dots])
    ax.scatter(
        [x for x, _ in dots],
        [y for _, y in dots],
        s=10,
        alpha=0.45,
        color=color,
        marker=marker,
        lw=0.5,
    )
    ax.plot(
        [x for x, _ in front],
        [-y for _, y in front],
        label=label,
        color=color,
        marker=marker,
        markersize=4,
        lw=1.65,
    )


def plot():
    joint = [read(f"fresh-joint-q{q}-scores.json") for q in [70, 60, 50]]
    prior = read("prior-joint-q50-scores.json")
    refresh = joint[-1]
    packets = read("two-shot-price.json")
    o = (
        read("ordinary-native-480-scores.json")
        + read("ordinary-near-summary.json")
        + [read("ordinary-fresh-cold.json")]
    )
    n = [read(f"neural-q{q}-body-and-ball.json") for q in [24, 36, 37]]
    assert all(
        row["source_sha256"] == refresh["source_sha256"] and len(row["body_rows"]) == 480
        for row in [*joint, prior, *n]
    )
    assert abs(prior["arm"]["kbps"] - packets["reuse"]["return_kbps"]) < 1e-8
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(5.9, 3.35), layout="constrained")
    curve(ax, o, "SVT-AV1", "#00587a", "o")
    curve(ax, n, "DCVC-UF", "#ab4c0d", "s")
    curve(ax, joint, "PointStream, warm fresh appearance", "#713493", "^")
    ax.scatter(
        [packets["send_new"]["return_kbps"]],
        [refresh["mean_dists"]],
        marker="^",
        s=58,
        ec="#713493",
        fc="white",
        linewidths=1.6,
        label="PointStream, charge fresh appearance",
        zorder=4,
    )
    ax.scatter(
        [prior["arm"]["kbps"]],
        [prior["mean_dists"]],
        marker="D",
        color="#191919",
        s=35,
        label="PointStream, actual 25.16 s cache hit",
        zorder=5,
    )
    ax.set(
        xscale="log",
        xlabel="Video bitrate on the returning shot (kbps)",
        ylabel="Player-region DISTS (lower is better)",
    )
    ax.set_xticks([8, 12, 16, 24, 40, 80, 160])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.grid(axis="y", alpha=0.22, lw=0.65)
    ax.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=0.90,
        fontsize=7.0,
    )
    for fmt in ["pdf", "png"]:
        fig.savefig(ROOT / f"cache-return-rate-quality.{fmt}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    plot()
