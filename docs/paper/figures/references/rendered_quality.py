"""Compare three AVIF rungs against the lossless reference and author review."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "docs/experiments/common-reference-q78-20261122"


def main():
    q90_dir = ROOT / "docs/experiments/common-reference-q90-20261122"
    data = json.loads((q90_dir / "rendered-crop-metrics.json").read_text())
    q90 = json.loads((q90_dir / "image-preflight.json").read_text())
    assert sum(row["image_bytes"] for row in q90["rows"]) == 107276
    q65 = json.loads((ROOT / "docs/experiments/common-reference-20261122/prepared.json").read_text())
    q78 = json.loads((EXPERIMENT / "prepared.json").read_text())
    decision = json.loads((EXPERIMENT / "review-decision.json").read_text())
    assert data["reference_alignment_max_rgb_delta"] == [0, 0]
    assert sum(row["avif_bytes"] for row in q65["rows"]) == 44179
    assert sum(row["avif_bytes"] for row in q78["rows"]) == 63915
    assert decision["ratings"]["player_1"]["accept"] == {"A": False, "B": False, "C": True}
    assert decision["ratings"]["player_2"]["accept"] == {"A": True, "B": True, "C": True}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True)
    fig.subplots_adjust(left=.11, right=.96, top=.74, bottom=.26, wspace=.24)
    fig.suptitle("Reference bytes and generated-player crop fidelity", x=.04, y=.98,
                 ha="left", fontsize=16, weight="bold")
    fig.text(.04, .86, "One development rally  ·  three shared AVIF settings  ·  24 source-aligned frames", fontsize=10)
    for ax, idx in zip(axes, (0, 1)):
        unit = "nadal_kyrgios__000_at120__person_" + f"{idx:012d}"
        scores = data["metrics"][unit]
        ax.plot([44.179, 63.915, 107.276], [scores[key]["rgb_psnr_db"] for key in ("avif_q65", "avif_q78", "avif_q90")],
                color="#086f9b", marker="o", markersize=7, linewidth=1.6, label="Shared AVIF reference")
        ax.axhline(scores["lossless"]["rgb_psnr_db"], color="#a54822", linestyle="--", linewidth=1.4,
                   label="Lossless reference quality")
        for x, label, key in ((44.179, "Q65", "avif_q65"), (63.915, "Q78", "avif_q78"), (107.276, "Q90", "avif_q90")):
            ax.annotate(label, (x, scores[key]["rgb_psnr_db"]), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(f"Player {idx + 1}", loc="left", weight="bold")
        ax.set_xlim(35, 118)
        ax.set_ylim(18.45, 20.5)
        ax.set_xlabel("Two image payloads (kB)")
        ax.grid(axis="y", color="#e3e6e8")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Full-crop RGB PSNR (dB)  ↑")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .11),
               ncol=2, frameon=False, fontsize=9)
    fig.text(.04, .02, "Author review on this rally accepted lossless for both players and Q65/Q78 for Player 2.",
             fontsize=9)
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"rendered-reference-quality.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
