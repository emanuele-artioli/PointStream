"""Render real sender signals on the measured court, fallback and return passage."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
INK = "#142b39"
MUTED = "#475965"
OBJECT = "#096b7e"
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


def photo(fig, location, name, folder, row):
    path = folder / f"{name}.png"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["derivative_sha256"]
    ax = fig.add_axes(location)
    with Image.open(path) as im:
        assert im.size == (640, 360)
        ax.imshow(im)
    ax.set(xlim=(-0.5, 639.5), ylim=(359.5, -0.5))
    ax.axis("off")
    return ax


def build(folder, output):
    receipt = json.loads((HERE / "source-frame-receipt.json").read_text())
    signals = json.loads((HERE / "packet-signals.json").read_text())
    first, gap, returning = receipt["rows"]
    assert signals["packet_sha256"] == receipt["packet_sha256"]
    assert first["passage_seconds"] == signals["tick"] / 50
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11.4, 6.05), facecolor="white")
    fig.text(
        0.038,
        0.933,
        "What PointStream sends during a broadcast",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.038,
        0.88,
        "One measured passage  ·  226.1 seconds  ·  11,305 frames at 50 fps",
        fontsize=11,
        color=MUTED,
    )

    fig.text(0.038, 0.803, "1   First rally", fontsize=13, weight="bold", color=INK)
    fig.text(
        0.038,
        0.757,
        "Player poses and a tracked ball on the source frame at 1.80 s",
        fontsize=9.5,
        color=MUTED,
    )
    ax = photo(fig, [0.038, 0.224, 0.497, 0.514], "first", folder, first)
    outline = [patheffects.Stroke(linewidth=2.75, foreground="#154651"), patheffects.Normal()]
    for points in signals["tracks"].values():
        joints = np.asarray(points)
        xy = joints[:, :2] / 3
        visible = joints[:, 2] > 0
        for a, b in LIMBS:
            if visible[a] and visible[b]:
                ax.plot(*xy[[a, b]].T, color=JOINT, lw=1.25, path_effects=outline)
        ax.scatter(*xy[visible].T, s=5, color=JOINT, edgecolors="#164a53", linewidths=0.3, zorder=4)
    trail = (
        np.asarray(
            [[np.nan, np.nan] if point is None else point for point in signals["tile_centers"]]
        )
        / 3
    )
    ax.plot(
        *trail.T,
        color=BALL,
        lw=1.3,
        path_effects=[
            patheffects.Stroke(linewidth=2.8, foreground="#5c410c"),
            patheffects.Normal(),
        ],
    )
    ax.scatter(*trail[-1], s=18, facecolor=BALL, edgecolor="#5c410c", linewidth=0.8, zorder=5)
    # The inset is a magnified source crop from the independent tick-90 label.
    inset_path = folder / "ball-inset.png"
    assert hashlib.sha256(inset_path.read_bytes()).hexdigest() == first["ball_inset_sha256"]
    inset = ax.inset_axes([0.773, 0.63, 0.19, 0.30])
    with Image.open(inset_path) as im:
        inset.imshow(im)
    inset.set(xticks=[], yticks=[])
    for spine in inset.spines.values():
        spine.set(color=BALL, linewidth=1.4)
    cx, cy = np.array(first["ball_center_xy"]) / 3
    ax.add_patch(
        Rectangle((cx - 32 / 3, cy - 32 / 3), 64 / 3, 64 / 3, fill=False, edgecolor=BALL, lw=0.8)
    )
    ax.annotate(
        "",
        xy=(cx + 32 / 3, cy - 32 / 3),
        xycoords="data",
        xytext=(0.773, 0.635),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "white", "lw": 0.85, "ls": ":"},
    )

    fig.text(
        0.038,
        0.196,
        "Cyan shows transmitted joints. Yellow follows ball-tile centers over 0.24 s.",
        color=MUTED,
        fontsize=9,
    )
    fig.text(0.038, 0.151, "273,927 B", fontsize=11, color=OBJECT, weight="bold")
    fig.text(
        0.135,
        0.151,
        "Initial references, poses, ball tiles and court video",
        fontsize=9.2,
        color=INK,
    )

    for y, n, title, stem, row, detail, cost in [
        (
            0.79,
            2,
            "Conventional fallback",
            "intervening_closeup",
            gap,
            "AV1 video for all\n216.5 intervening seconds.",
            "540,921 B",
        ),
        (
            0.455,
            3,
            "Return to court",
            "return",
            returning,
            "Cached appearances, new\nplayer signals, fitted ball\ntrajectory and court video.",
            "24,723 B",
        ),
    ]:
        fig.text(0.566, y, f"{n}   {title}", fontsize=13, weight="bold", color=INK)
        photo(fig, [0.566, y - 0.212, 0.191, 0.193], stem, folder, row)
        fig.text(0.775, y - 0.073, detail, fontsize=9.5, va="top", color=INK, linespacing=1.35)
        fig.text(0.775, y - 0.197, cost, fontsize=11, weight="bold", color=OBJECT)
        fig.text(
            0.566, y - 0.245, f"Source at {row['passage_seconds']:.2f} s", fontsize=8.5, color=MUTED
        )
    fig.add_artist(
        plt.Line2D(
            [0.038, 0.963], [0.111, 0.111], transform=fig.transFigure, color="#bdc9cd", lw=0.75
        )
    )
    fig.text(
        0.038,
        0.055,
        "Total delivery   841,055 B  ·  29.759 kbps",
        weight="bold",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.475,
        0.055,
        "Includes the 1,484 B index. Routes were chosen retrospectively.",
        fontsize=9.2,
        color=MUTED,
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            output / f"measured-passage-intro.{ext}", dpi=180, bbox_inches="tight", pad_inches=0.15
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.private_dir, args.output_dir or args.private_dir)
