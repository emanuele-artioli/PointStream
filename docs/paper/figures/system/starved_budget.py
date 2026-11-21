"""Contrast measured startup packets, measured native video, and reference-reuse scenarios."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"


def main():
    base = json.loads((EXP / "starved-base-local-20261122/base-receipt.json").read_text())
    controls = json.loads((EXP / "starved-stroke-20261122/result.json").read_text())["selected"]
    old = json.loads((EXP / "clean-court-grabcut-20261119/delivery-receipt.json").read_text())
    ladder = json.loads((EXP / "reference-ladder-20261122/result.json").read_text())
    plain, clean = [
        next(r for r in base["outputs"] if r["arm"] == a and r["crf"] == 60)
        for a in ["original", "clean"]
    ]
    keypoints = old["rows"][0]["components"]["keypoints"]["bytes"]
    refs = clean["reference_bytes"]
    tile = clean["files"]["ball.sidecar"]
    misc = clean["total_bytes"] - sum((clean["file_bytes"], keypoints, refs, tile))
    assert misc >= 0 and clean["files"]["receiver.manifest"] <= misc
    rungs = [r for r in ladder["probes"] if r["codec"] == "avif" and r["quality"] in [46, 65, 78]]
    rungs.sort(key=lambda r: r["quality"])
    assert [r["total_bytes"] for r in rungs] == [24463, 44179, 63915]
    # Delivered bytes per 4.8-second clip convert to decimal kbps by division by 600.
    common = [clean["file_bytes"], tile, keypoints, misc]
    rows = [
        ("Plain AV1 · same CRF and scale", [plain["file_bytes"], 0, 0, 0, 0]),
        ("Plain AV1 · 480 × 272", [controls["plain"]["file_bytes"], 0, 0, 0, 0]),
        ("ROI AV1 · 480 × 272", [controls["roi"]["file_bytes"], 0, 0, 0, 0]),
        ("PointStream · lossless start", common + [refs]),
        ("Valid cache · ideal floor", common + [0]),
    ] + [(f"AVIF Q{r['quality']} · modeled start", common + [r["total_bytes"]]) for r in rungs]
    fig, ax = plt.subplots(figsize=(9, 4.65))
    colors = ["#35778b", "#d48937", "#7a56a0", "#838789", "#25334c"]
    for i, (label, items) in enumerate(rows):
        left = 0
        for j, val in enumerate(items):
            ax.barh(
                i,
                val / 600,
                left=left,
                height=0.53,
                color=colors[j],
                hatch="//" if i >= 5 and j == 4 else None,
                edgecolor="white",
                lw=0.4,
            )
            left += val / 600
        ax.text(left + 4, i, f"{left:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [v[0] for v in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 515)
    ax.set_xlabel("Total bitrate over 4.8 s (kbps)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("What fits a low-rate player-and-ball stream?", x=0.04, ha="left", weight="bold")
    ax.set_title(
        "240 × 136 CRF 60 joint base · 50 fps · Nadal–Kyrgios · hatched images priced at measured AVIF sizes", fontsize=10, loc="left"
    )
    fig.subplots_adjust(left=0.29, right=0.98, top=0.8, bottom=0.25)
    labels = [
        "Base AV1",
        "Ball tiles + positions",
        "Player keypoints",
        "Packet overhead",
        "Player images",
    ]
    fig.legend(
        [Patch(color=c) for c in colors],
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"starved-budget.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
