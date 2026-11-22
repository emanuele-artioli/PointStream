"""Plot the complete delivery price of two rallies and the intervening broadcast."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"


def read(study, name):
    return json.loads((EXP / (study + "-20261122") / name).read_text())


def main():
    joint = read("cumulative-joint", "trajectory-detailed-receipt.json")
    fitted = read("cumulative-joint", "trajectory-package-receipt.json")
    native = read("cumulative-joint", "same-gap-control-package-receipt.json")
    reference = read("passage-same-gap", "control.json")
    continuous = read("passage-native", "result-20.json")
    gap = read("cumulative-joint", "gap-luma-score.json")
    parts = {p["name"]: p["bytes"] for p in joint["entries"]}
    assert joint["parsed_to_eof"] and native["all_sections_reparsed"]
    frames = sum(s["frames"] for s in joint["segments"])
    assert joint["packet_sha256"] == fitted["packet_sha256"]
    assert frames == native["frames"] == continuous["metrics"]["full"]["frames"]
    assert gap["source_sha256"] == continuous["source_sha256"]
    assert parts["fallback_ivf"] == reference["gap"]["bytes"]
    assert gap["sse"] == reference["gap"]["sse"]
    initial = read("cumulative-joint", "initial-playback-score.json")
    return_scores = read("cumulative-joint", "return-regions-result.json")
    total_samples = 255**2 * frames * math.prod(gap["support"])
    fitted_scores = read("cumulative-joint", "return-trajectory-score.json")
    assert fitted["packet_sha256"] == fitted_scores["packet_sha256"]
    ps_sse = initial["joint_sse"] + gap["sse"] + fitted_scores["return_luma_sse"]["trajectory"]
    av1_sse = (
        initial["control_display_sse"]
        + gap["sse"]
        + return_scores["all_frame_luma_sse"]["ordinary"]
    )
    ps_psnr, av1_psnr = (10 * math.log10(total_samples / sse) for sse in (ps_sse, av1_sse))

    categories = ["Appearance", "Player stream and base", "Ball", "Ordinary AV1", "Framing"]
    colors = ["#762b67", "#007499", "#cda33a", "#737e89", "#20242a"]
    joint_stages = [
        [
            parts["cache_initial"],
            parts["initial_wire"],
            parts["initial_ball"],
            0,
            parts["initial_manifest"],
        ],
        [0, 0, 0, parts["fallback_ivf"], 0],
        [
            parts["cache_return"],
            parts["return_wire"],
            parts["return_trajectory"],
            0,
            parts["return_manifest"],
        ],
        [0, 0, 0, 0, fitted["index_bytes"]],
    ]
    native_stages = [
        [0, 0, 0, reference[phase]["bytes"], 0] for phase in ("initial", "gap", "return")
    ] + [[0, 0, 0, 0, native["index_bytes"]]]
    assert sum(map(sum, joint_stages)) == fitted["total_bytes"]
    assert sum(map(sum, native_stages)) == native["total_bytes"]

    fig, ax = plt.subplots(figsize=(9, 5.3))
    fig.subplots_adjust(left=0.26, right=0.9, top=0.77, bottom=0.21)
    labels, ticks = [], []
    for i, name in enumerate(("First rally", "Intervening video", "Return rally", "Cut index")):
        for j, (mode, stages) in enumerate((("PointStream", joint_stages), ("AV1", native_stages))):
            y = 3 * i + j
            left = 0
            for cost, color in zip(stages[i], colors):
                if cost:
                    ax.barh(y, cost / 1000, left=left / 1000, height=0.76, color=color, linewidth=0)
                    left += cost
            ax.text(left / 1000 + 4, y, f"{left:,} B", va="center", fontsize=8)
            ticks.append(y)
            labels.append(f"{name}  ·  {mode}")
    ax.set(
        yticks=ticks,
        yticklabels=labels,
        xlabel="Transmitted kilobytes (1 kB = 1,000 bytes)",
        xlim=(0, 620),
    )
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "What two rallies cost across a 226-second broadcast",
        x=0.04,
        y=0.98,
        ha="left",
        weight="bold",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.90,
        f"PointStream {fitted['kbps']:.3f} kbps   •   AV1 with identical cutaway {native['kbps']:.3f} kbps   •   11,305 frames",
        fontsize=11,
    )
    fig.legend(
        [Patch(color=c) for c in colors],
        categories,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.875),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.04,
        0.075,
        f"On 12 return frames, DISTS in sender-defined regions was bodies {fitted_scores['body_means']['trajectory']['dists']:.3f} vs {fitted_scores['body_means']['ordinary']['dists']:.3f}, ball {fitted_scores['ball_means']['trajectory']['dists']:.3f} vs {fitted_scores['ball_means']['ordinary']['dists']:.3f}.",
        fontsize=10,
    )
    ctrl = continuous["metrics"]["full"]
    fig.text(
        0.04,
        0.031,
        f"Whole passage: {ps_psnr:.2f} vs {av1_psnr:.2f} dB with the same gap. Continuous AV1: {ctrl['luma_psnr']:.2f} dB at {ctrl['kbps']:.2f} kbps (480 × 272 luma).",
        fontsize=10,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("passage-bill." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
