"""Draw the shared-setting JPEG, WebP and AVIF reference ladder."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parents[2] / "experiments/reference-ladder-20261122/result.json"
STYLES = {"jpeg": ("JPEG", "#4B5870", "s"), "webp": ("WebP", "#BA512D", "^"), "avif": ("AVIF", "#007F7C", "o")}


def draw():
    data = json.loads(RECEIPT.read_bytes())
    by_code = {(r["codec"], r["quality"]): r for r in data["probes"]}
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.35, 3.9))
    for i, ax in enumerate(axes):
        for cap in data["caps_bytes"]:
            ax.axvline(cap / 1000, linewidth=.7, color="#d1d4d6", linestyle=":", zorder=0)
        for codec, (label, color, marker) in STYLES.items():
            pts = [by_code[codec, s["quality"]] for s in data["selected"] if s["codec"] == codec]
            ax.plot([r["total_bytes"]/1000 for r in pts], [r["players"][i]["psnr_db"] for r in pts],
                    color=color, marker=marker, label=label, linewidth=1.6, markersize=5.5)
            if codec == "avif":
                for r in pts:
                    ax.annotate(f'Q{r["quality"]}', (r["total_bytes"]/1000, r["players"][i]["psnr_db"]),
                                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.set_title(f"Player {i+1}", loc="left", fontsize=12, weight="bold")
        ax.set_xticks([25,45,65])
        ax.set_xlim(20,70)
        ax.set_ylim(32.8,46.1)
        ax.grid(axis="y", linewidth=.55, color="#dcdfe1")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Reference RGB PSNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5,.005))
    fig.suptitle("Fidelity of the two transmitted player references", x=.09, y=.985, ha="left", fontsize=15, weight="bold")
    fig.text(.5,.155,"Total image payload for the pair (kB)",ha="center",fontsize=10)
    fig.subplots_adjust(left=.09,right=.99,top=.81,bottom=.32,wspace=.15)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"reference-ladder.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw()
