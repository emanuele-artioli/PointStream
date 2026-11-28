"""Render compact single-row 3-panel real-frame Figure 1 for ACM TOMM."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
INK = "#142b39"
MUTED = "#475965"
OBJECT = "#096b7e"
FALLBACK = "#526068"
JOINT = "#6cfff2"
BALL = "#ffe15f"
LIMBS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def build(frames_dir: Path, output_pdf: Path, output_png: Path | None = None) -> None:
    receipt = json.loads((HERE / "source-frame-receipt.json").read_text())
    signals = json.loads((HERE / "packet-signals.json").read_text())
    first = receipt["rows"][0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(10.8, 2.95), facecolor="white")

    pw = 0.300
    gap_w = 0.036
    x0s = [0.014, 0.014 + pw + gap_w, 0.014 + 2 * (pw + gap_w)]
    py = 0.225
    ph = 0.625

    def resolve(candidates: tuple[str, ...]) -> Path:
        for name in candidates:
            path = frames_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {frames_dir}")

    panels = [
        (
            x0s[0],
            ("first.png", "img-000.png"),
            "1  First court visit",
            "Full-Obj (cold) · 1.80 s",
            OBJECT,
            "273,927 B",
            "cache (242.6 KB) + poses, ball, court",
            None,
        ),
        (
            x0s[1],
            ("intervening_closeup.png", "img-002.png"),
            "2  Cutaway (216.5 s)",
            "Conv fallback · 24.48 s",
            FALLBACK,
            "540,921 B",
            "conventional AV1 · state retained",
            "Cache retained",
        ),
        (
            x0s[2],
            ("return.png", "img-003.png"),
            "3  Return to court",
            "Full-Obj (warm) · 223.50 s",
            OBJECT,
            "24,723 B",
            "cached ref + new poses, ball, court",
            "Appearance reused",
        ),
    ]

    for idx, (x0, img_names, title, badge, accent, cost, desc, pill) in enumerate(panels):
        fig.text(x0, py + ph + 0.055, title, fontsize=9.8, weight="bold", color=INK, va="bottom")
        fig.text(
            x0 + pw,
            py + ph + 0.057,
            badge,
            fontsize=8.0,
            weight="bold",
            color=accent,
            ha="right",
            va="bottom",
        )

        ax = fig.add_axes((x0, py, pw, ph))
        with Image.open(resolve(img_names)) as im:
            ax.imshow(im, extent=(0, 1920, 1080, 0), aspect="auto")
        ax.set(xlim=(0, 1920), ylim=(1080, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(accent if idx != 1 else "#8a98a0")
            spine.set_linewidth(1.1)

        if idx == 0:
            outline = [patheffects.Stroke(linewidth=2.2, foreground="#154651"), patheffects.Normal()]
            for points in signals["tracks"].values():
                joints = np.asarray(points)
                xy = joints[:, :2]
                visible = joints[:, 2] > 0
                for a, b in LIMBS:
                    if visible[a] and visible[b]:
                        ax.plot(*xy[[a, b]].T, color=JOINT, lw=1.1, path_effects=outline)
                ax.scatter(
                    *xy[visible].T,
                    s=4.5,
                    color=JOINT,
                    edgecolors="#164a53",
                    linewidths=0.3,
                    zorder=4,
                )
            trail = np.asarray(
                [[np.nan, np.nan] if pt is None else pt for pt in signals["tile_centers"]]
            )
            ax.plot(
                *trail.T,
                color=BALL,
                lw=1.25,
                path_effects=[
                    patheffects.Stroke(linewidth=2.4, foreground="#5c410c"),
                    patheffects.Normal(),
                ],
            )
            ax.scatter(
                *trail[-1], s=14, facecolor=BALL, edgecolor="#5c410c", linewidth=0.7, zorder=5
            )

            inset = ax.inset_axes((0.755, 0.56, 0.215, 0.37))
            with Image.open(resolve(("ball-inset.png", "img-001.png"))) as bim:
                inset.imshow(bim)
            inset.set(xticks=[], yticks=[])
            for spine in inset.spines.values():
                spine.set(color=BALL, linewidth=1.3)
            cx, cy = first["ball_center_xy"]
            ax.add_patch(
                Rectangle((cx - 32, cy - 32), 64, 64, fill=False, edgecolor=BALL, lw=0.9)
            )
            ax.annotate(
                "",
                xy=(cx + 32, cy - 32),
                xycoords="data",
                xytext=(0.755, 0.57),
                textcoords="axes fraction",
                arrowprops={"arrowstyle": "-", "color": "white", "lw": 0.85, "ls": ":"},
            )
        elif pill:
            ax.text(
                0.965,
                0.065,
                pill,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.6,
                weight="bold",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.25,rounding_size=0.15",
                    "facecolor": accent,
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

        fig.text(x0, py - 0.032, cost, fontsize=9.0, weight="bold", color=accent, va="top")
        fig.text(x0 + 0.078, py - 0.034, desc, fontsize=7.8, color=INK, va="top")

    for left_x in (x0s[0] + pw, x0s[1] + pw):
        arrow = FancyArrowPatch(
            (left_x + 0.005, py + ph / 2),
            (left_x + gap_w - 0.005, py + ph / 2),
            transform=fig.transFigure,
            arrowstyle="-|>,head_length=5,head_width=3.2",
            color="#64747d",
            linewidth=1.3,
        )
        fig.add_artist(arrow)

    fig.add_artist(
        plt.Line2D(
            [0.014, 0.986], [0.105, 0.105], transform=fig.transFigure, color="#c5d0d4", lw=0.75
        )
    )
    fig.text(
        0.014,
        0.042,
        "Total: 841,055 B over 226.1 s (29.76 kbps, incl. 1,484 B index)",
        fontsize=8.0,
        weight="bold",
        color=INK,
        va="center",
    )
    fig.text(
        0.986,
        0.042,
        "Cyan: 17-joint poses  ·  Yellow: 0.24 s ball trail  ·  Retrospective routing",
        fontsize=7.6,
        color=MUTED,
        ha="right",
        va="center",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf, dpi=220, metadata={"CreationDate": None}, bbox_inches="tight", pad_inches=0.05
    )
    if output_png is not None:
        fig.savefig(output_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()
    build(args.frames_dir, args.output_pdf, args.output_png)
