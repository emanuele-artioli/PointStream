"""Compare the same player masks at two measured rate regimes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"


def read(study, file="result.json"):
    return json.loads((EXP / study / file).read_text())


def main():
    low = read("joint-player-machine-20261122")["means"]
    high = read("starved-player-segmentation-20261122")["means"]
    assert low["source"] == high["source"] and low["receiver_mask"] == high["receiver_mask"]
    bills = read("joint-trajectory-20261122")["streams"]["20"]
    binding = read("joint-machine-20261122", "binding.json")["costs_bytes"]
    assert bills["bytes"] == binding["joint"]
    cold = bills["cold_kbps"]
    pts = read("starved-stroke-20261122")["selected"]
    high_joint = next(
        r["optimistic_no_reference_kbps"]
        for r in read("starved-base-local-20261122", "base-receipt.json")["outputs"]
        if r["arm"] == "clean" and r["crf"] == 60
    )
    arms = [
        (
            "Ordinary AV1 RGB",
            "#223d6a",
            "o",
            [(binding["ordinary"] / 600, low["ordinary"]), (pts["plain"]["rate"], high["plain"])],
        ),
        ("DCVC-UF RGB", "#713490", "D", [(binding["neural"] / 600, low["neural"])]),
        ("ROI AV1 RGB", "#417a46", "X", [(pts["roi"]["rate"], high["roi"])]),
        (
            "PointStream RGB",
            "#bb530c",
            "^",
            [(bills["kbps"], low["joint"]), (high_joint, high["pointstream"])],
        ),
        (
            "PointStream receiver mask",
            "#087c87",
            "s",
            [(bills["kbps"], low["receiver_mask"]), (high_joint, high["receiver_mask"])],
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for label, color, marker, measurements in arms:
        ax.plot(
            [x for x, r in measurements],
            [r["all"]["mean"] for x, r in measurements],
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=7.5,
            label=label,
        )
    ceiling = low["source"]["all"]["mean"]
    ax.axhline(ceiling, color="#747474", linestyle=":", linewidth=1.1)
    ax.text(
        44, ceiling + 0.012, f"Source RGB  {ceiling:.3f}", fontsize=9, ha="center", color="#555"
    )
    ax.set(
        xlabel="Delivered bitrate with valid player references (kbps)",
        ylabel="Mean body-mask IoU (higher is better)",
        xlim=(17, 70),
        ylim=(0.25, 0.93),
        xticks=[20, 30, 40, 50, 60, 70],
    )
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.3)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.81)
    fig.suptitle(
        "Can an application recover the player silhouettes?",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=15.5,
        weight="bold",
    )
    fig.text(
        0.04,
        0.89,
        "One 50-fps serve. The same RGB segmentation model and 24 reviewed body masks throughout.",
        fontsize=9.3,
    )
    fig.text(
        0.04,
        0.065,
        f"PointStream cold start is {cold:.2f} kbps at the lower point and 469.34 kbps at the upper point.",
        fontsize=8.8,
    )
    fig.text(
        0.04,
        0.025,
        "The direct mask comes from player reconstruction. Line segments connect the two measured operating points.",
        fontsize=8.8,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("starved-masks." + ext), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
