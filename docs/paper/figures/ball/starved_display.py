"""Plot real display-fidelity comparisons and optionally render private source crops."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
RECORD = HERE.parents[2] / "experiments/starved-rally-tiles-20261122"


def figures(source_zip, private_output):
    rates = json.loads((RECORD / "render.json").read_text())
    scores = json.loads((RECORD / "scores.json").read_text())
    assert rates["crops_sha256"] == scores["crops_sha256"]
    fig, ax = plt.subplots(figsize=(7, 4.1))
    for arm, label, symbol, color in [
        ("plain", "Plain AV1 · 240 × 136", "o", "#236999"),
        ("tile", "AV1 + tracked tiles · 240 × 136 base", "s", "#b54b13"),
    ]:
        points = [
            (p[arm + "_kbps"], scores["means"][arm + str(p["target"])]["ball"])
            for p in rates["pairs"]
            if p["feasible"]
        ]
        ax.plot(*zip(*points), marker=symbol, color=color, label=label)
    if (RECORD / "strong-scores.json").exists():
        extra = json.loads((RECORD / "strong-scores.json").read_text())
        setup = json.loads((RECORD / "strong-result.json").read_text())
        assert extra["crops_sha256"] == setup["crop_sha256"]
        for mode, color, marker, label in [
            ("plain", "#56782c", "^", "Plain AV1 · 480 × 272"),
            ("roi", "#603987", "D", "Player and ball ROI · 480 × 272"),
        ]:
            pts = sorted(
                (r["rate"], extra["means"][a]["ball"])
                for a, r in setup["selected"].items()
                if a.startswith(mode)
            )
            if pts:
                ax.plot(*zip(*pts), marker=marker, color=color, label=label, linestyle="--", lw=1.2)
    ax.set(
        xlabel="Complete transmitted bitrate (kbps)", ylabel="Ball-crop DISTS  (lower is better)"
    )
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Ball appearance under a constrained bitrate", x=0.10, ha="left", weight="bold")
    ax.set_title(
        "One Nadal–Kyrgios rally · 12 labeled visible instants · 64 × 64 source crops",
        loc="left",
        fontsize=9,
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"starved-display.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if source_zip is None:
        return
    assert hashlib.sha256(source_zip.read_bytes()).hexdigest() == rates["crops_sha256"]
    private_output.mkdir(exist_ok=False, parents=True)
    fig, axes = plt.subplots(3, 3, figsize=(6.2, 5.6))
    with zipfile.ZipFile(source_zip) as z:
        for i, t in enumerate([10, 90, 170]):
            for j, (a, title) in enumerate(
                [
                    ("source", "Source"),
                    ("plain63", "AV1 · 60.62 kbps"),
                    ("tile63", "AV1 + tiles · 60.67 kbps"),
                ]
            ):
                im = Image.open(BytesIO(z.read(f"{t:03d}/ball/{a}.png")))
                axes[i, j].imshow(im, interpolation="nearest")
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(title, fontsize=10)
                if j == 0:
                    axes[i, j].set_ylabel(f"{t / 50:.2f} s", rotation=0, ha="right", va="center")
    fig.suptitle("Tracking the ball at 61 kbps", weight="bold")
    fig.text(
        0.5,
        0.02,
        "Same source coordinates in every column. First row has no transmitted tile.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.12, top=0.87, bottom=0.08, left=0.14, right=0.98)
    for ext in ["png", "pdf"]:
        fig.savefig(private_output / f"ball-at-61-kbps.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-zip", type=Path)
    cli.add_argument("--private-output", type=Path)
    args = cli.parse_args()
    if bool(args.source_zip) != bool(args.private_output):
        cli.error("Supply both private paths together")
    figures(args.source_zip, args.private_output)
