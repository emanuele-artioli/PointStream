"""Plot every preselected source tick so support gaps are visible at their true locations."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
JOBS = [
    ("match108", "Fritz–Draper", "25 fps"),
    ("match102", "Sinner–Shelton", "29.97 fps"),
    ("match120", "Gaston–Thiem", "60 fps"),
]


def main():
    output = HERE.parents[1] / "paper/figures/test-joint"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 3.3), sharex=True, layout="constrained")
    cmap = ListedColormap(["#eee9e3", "#176b6a"])
    for ax, (name, match, fps) in zip(axes, JOBS, strict=True):
        result = json.loads((HERE / f"{name}-result.json").read_text())
        q = json.loads((HERE / f"{name}-prices.json").read_text())
        near, far = result["selected_ids"]
        rows = []
        for pool, n in [
            (result["tracks"], near),
            (result["tracks"], far),
            (result["whole_frame_original"], far),
        ]:
            fit = np.ones(240, dtype=int)
            fit[pool[n]["missing"]] = 0
            rows.append(fit)
        ax.imshow(
            np.stack(rows),
            interpolation="nearest",
            aspect="auto",
            extent=(-0.5, 239.5, 2.5, -0.5),
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.hlines([0.5, 1.5], -0.5, 239.5, color="white", lw=1)
        ax.set(
            yticks=[0, 1, 2],
            yticklabels=["Near athlete", "Far athlete, crop", "Far athlete, whole frame"],
        )
        ax.set_title(f"{match}  ·  {fps}", loc="left", size=9.5, fontweight="medium")
        ax.set_title(f"Both fitted on {q['available_both']}/240 frames", loc="right", size=8.5)
        ax.tick_params(axis="both", length=0, labelsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks([0, 40, 80, 120, 160, 200, 239])
    axes[-1].set_xlabel("Native display frame", size=9)
    fig.legend(
        handles=[
            Patch(color="#176b6a", label="Pose fitted"),
            Patch(color="#eee9e3", label="Pose unavailable"),
        ],
        loc="outside upper center",
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output / "test-pose-support.pdf", bbox_inches="tight")
    fig.savefig(output / "test-pose-support.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
