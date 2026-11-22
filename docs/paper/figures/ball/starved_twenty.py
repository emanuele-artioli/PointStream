"""Build the private 20-kbps illustration from previously scored decoded crops."""
import argparse
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

EXP = Path(__file__).resolve().parents[3] / "experiments/starved-ball-twenty-20261122"


def build(archive, directory):
    receipt = json.loads((EXP / "result.json").read_text())
    scores = json.loads((EXP / "scores.json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["crops_sha256"] == scores["crops_sha256"]
    arms = [("source", "Source"), ("plain160Q61", "Ordinary AV1\n20.063 kbps"),
            ("pointstream", "PointStream\n19.965 kbps")]
    directory.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 6, figsize=(10.6, 5.35))
    fig.subplots_adjust(left=.145, right=.985, top=.80, bottom=.12, hspace=.08, wspace=.075)
    with zipfile.ZipFile(archive) as saved:
        for j, tick in enumerate(range(10, 231, 40)):
            for i, (key, label) in enumerate(arms):
                image = Image.open(BytesIO(saved.read(f"{tick:03d}/ball/{key}.png")))
                ax = axes[i, j]
                ax.imshow(image, interpolation="nearest")
                ax.set_axis_off()
                if i == 0:
                    ax.set_title(f"{tick / 50:.1f} s", fontsize=10)
                if j == 0:
                    ax.text(-.09, .5, label, transform=ax.transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("A ball transmitted at 20 kbps", x=.03, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.03, .865, "One 4.8-second serve, sampled every 0.8 seconds at a source-labeled location.", fontsize=10)
    fig.text(.03, .03, "All 12 labeled instants enter the score. Mean ball-crop DISTS improved from 0.4270 to 0.3643. Lower is better.", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"ball-at-20-kbps.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    build(args.archive, args.directory)
