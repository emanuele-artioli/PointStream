"""Plot source-flow-compensated inter-frame error for the 12 reviewed body pairs."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "experiments/joint-trajectory-20261122/temporal-with-neural.json"


def main():
    result = json.loads(DATA.read_text())
    scores = {(r["tick"], r["region"], r["arm"]): r["error"] for r in result["rows"]}
    ticks = sorted({t for t, region, k in scores if region == "p1"})
    assert ticks == list(range(10, 240, 20))
    arms = [
        ("plain", "Ordinary AV1", "#133e68", "o"),
        ("neural", "DCVC-UF", "#734296", "D"),
        ("joint", "PointStream", "#b34a1d", "^"),
    ]
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    for arm, label, color, marker in arms:
        errs = [mean(scores[t, k, arm] for k in ("p1", "p2")) for t in ticks]
        avg = mean(errs)
        expected = mean(result["means"][arm][r]["error"] for r in ("p1", "p2"))
        assert abs(avg - expected) < 1e-7
        ax.plot(
            [t / 50 for t in ticks],
            errs,
            color=color,
            marker=marker,
            ms=5,
            lw=1.35,
            label=f"{label}  ·  mean {avg:.2f}",
        )
    ax.set(
        xlabel="Time within the serve (seconds)",
        ylabel="Body motion error (luma levels)",
        xlim=(0, 4.8),
        ylim=(0, None),
    )
    ax.grid(alpha=0.16)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.94)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.24, top=0.8)
    fig.suptitle(
        "How well does each stream preserve frame-to-frame body motion?",
        fontsize=13.7,
        weight="bold",
        x=0.035,
        y=0.974,
        ha="left",
    )
    fig.text(
        0.035,
        0.879,
        "Two player masks per instant. Each point compares the current frame with its predecessor using the same source optical flow.",
        fontsize=8.8,
    )
    fig.text(
        0.035,
        0.065,
        "Lower values mean closer agreement with the reference motion residual. Source-flow estimates and twelve instants limit interpretation.",
        fontsize=8.4,
    )
    fig.text(
        0.035,
        0.025,
        "PointStream 21.545 kbps with valid appearances (428.777 kbps cold), ordinary AV1 21.623 kbps, DCVC-UF 20.450 kbps.",
        fontsize=8.4,
    )
    for extension in ("pdf", "png"):
        fig.savefig(HERE / ("body-motion." + extension), dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
