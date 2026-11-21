"""Render the scene-selective system diagram and an optional private source timeline."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ITEM = Path(__file__).resolve().parent
SOURCE_REVIEW = Path(
    "/Users/manu/Documents/PointStream-review/2026-11-18-continuous-return/source-only-review.mp4"
)
SOURCE_SHA = "0d20d883430ecfea33c474f7e7c7f42df6052437898d390af254148def687b50"
DARK, TEAL, GRAY, ORANGE = "#24343d", "#007d80", "#657887", "#bb712e"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def box(ax, x, y, width, height, text, color=GRAY, fill="white", fontsize=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=.02,rounding_size=.07",
            linewidth=1.2,
            edgecolor=color,
            facecolor=fill,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        linespacing=1.55,
    )


def arrow(ax, start, end, color=GRAY, connection="arc3", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color=color,
            linewidth=1.3,
            connectionstyle=connection,
            linestyle="--" if dashed else "-",
        )
    )


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder / (name + "." + suffix), bbox_inches="tight", dpi=190)
    plt.close(fig)


def architecture(output):
    fig, ax = plt.subplots(figsize=(14.6, 5.5))
    ax.set_xlim(-0.1, 15.2)
    ax.set_ylim(-0.1, 5.1)
    ax.axis("off")
    ax.text(
        0.05,
        4.85,
        "PointStream selects a representation for each shot",
        fontsize=18,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0.05,
        4.43,
        "System architecture · coding choices depend on scene content, receiver state and available bitrate",
        fontsize=11,
        color=GRAY,
    )
    ax.text(0.05, 3.95, "SERVER", fontsize=10, fontweight="bold", color=GRAY)
    ax.text(10.35, 3.95, "RECEIVER", fontsize=10, fontweight="bold", color=GRAY)
    box(ax, 0.05, 2.05, 1.25, 0.75, "Broadcast\nframes")
    box(ax, 1.65, 2.05, 1.25, 0.75, "Detect\nshot cuts")
    box(ax, 3.25, 2.05, 1.65, 0.75, "Scene and\nrate decision", TEAL, "#edf7f6")
    box(
        ax,
        5.45,
        2.85,
        2.45,
        0.95,
        "Foreground representation\nBackground recovery\nReference / update coding",
        TEAL,
        "#edf7f6",
        10,
    )
    box(ax, 5.45, 0.9, 2.45, 0.8, "Conventional\nfull-frame encoder")
    box(ax, 8.5, 2.05, 1.45, 0.75, "Timed\nbitstream")
    box(
        ax,
        10.5,
        2.85,
        2.45,
        0.95,
        "Decode points and pixels\nRender foreground layers\nUpdate background state",
        TEAL,
        "#edf7f6",
        10,
    )
    box(ax, 10.5, 0.9, 2.45, 0.8, "Conventional\nfull-frame decoder")
    box(ax, 13.5, 2.05, 1.45, 0.75, "Compose and\npresent frames", TEAL, "#edf7f6")
    for a, b in [((1.3, 2.425), (1.65, 2.425)), ((2.9, 2.425), (3.25, 2.425))]:
        arrow(ax, a, b)
    arrow(ax, (4.9, 2.62), (5.45, 3.3), TEAL)
    arrow(ax, (4.9, 2.22), (5.45, 1.3))
    ax.text(4.1, 3.42, "Eligible shot", fontsize=9.5, color=TEAL, ha="center")
    ax.text(4.15, 1.13, "Other or uncertain", fontsize=9.5, color=GRAY, ha="center")
    arrow(ax, (7.9, 3.3), (8.5, 2.63), TEAL)
    arrow(ax, (7.9, 1.3), (8.5, 2.22))
    arrow(ax, (9.95, 2.62), (10.5, 3.3), TEAL)
    arrow(ax, (9.95, 2.22), (10.5, 1.3))
    arrow(ax, (12.95, 3.3), (13.5, 2.62), TEAL)
    arrow(ax, (12.95, 1.3), (13.5, 2.22))
    ax.text(
        6.67,
        2.52,
        "Poses · appearance · ball locations\nOptional object pixels · background updates",
        ha="center",
        va="top",
        fontsize=8.5,
        color=TEAL,
        linespacing=1.5,
    )
    ax.text(
        11.72,
        2.52,
        "Retain valid appearance and court state\nacross conventional-coded shots",
        ha="center",
        va="top",
        fontsize=8.5,
        color=TEAL,
        linespacing=1.5,
    )
    ax.text(0.05, 0.28, "Tennis example", fontsize=10, fontweight="bold", color=DARK)
    ax.text(
        1.45,
        0.28,
        "Point views may use object coding. Close-ups, crowd shots and interludes follow the conventional path.",
        fontsize=10,
        color=DARK,
    )
    ax.text(
        0.05,
        -0.03,
        "All initialization, refresh, model updates and switching payloads contribute to the transmitted rate.",
        fontsize=9.5,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.03)
    save(fig, output, "architecture")


def private_timeline(output):
    if sha(SOURCE_REVIEW) != SOURCE_SHA:
        raise ValueError("Source-only review video identity changed")
    report_path = ROOT / "docs/experiments/continuous-return-20261118/results.json"
    report = json.loads(report_path.read_text())
    assert report["court"]["cuts"] == [100, 145] and report["interval"] == [493, 738]
    assert (
        report["sha256"]["/private/tmp/ps-continuous-20261118/Datasets/source-only-review.mp4"]
        == SOURCE_SHA
    )
    ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
    assert sha(ffmpeg) == report["sha256"]["/opt/homebrew/bin/ffmpeg"]
    output.mkdir(parents=True, exist_ok=True)
    # Select fixed mid-shot views from the already verified 245-frame presentation copy.
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(SOURCE_REVIEW),
        "-vf",
        r"select=eq(n\,50)+eq(n\,120)+eq(n\,195)",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    data = subprocess.check_output(command, timeout=30)
    frames = np.frombuffer(data, np.uint8).reshape(3, 540, 960, 3)
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.24, wspace=0.07)
    titles = ["Court view", "Close-up", "Return to court"]
    times = ["0.00–2.00 s", "2.00–2.90 s", "2.90–4.90 s"]
    routes = [
        "Assess eligibility and coding cost\nInitialize foreground and background state",
        "Encode the complete shot\nKeep valid object and court references",
        "Assess eligibility and coding cost\nUpdate or refresh the returning court",
    ]
    for i in range(3):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(frames[i])
        ax.axis("off")
        ax.set_title(titles[i], fontsize=12, fontweight="bold", pad=9)
        text = fig.add_subplot(grid[1, i])
        text.axis("off")
        text.text(0.5, 0.9, times[i], ha="center", fontsize=11, color=GRAY)
        text.text(
            0.5,
            0.52,
            routes[i],
            ha="center",
            va="top",
            fontsize=10,
            color=TEAL if i != 1 else GRAY,
            linespacing=1.6,
        )
    fig.suptitle(
        "One broadcast, several coding choices", x=0.035, ha="left", fontsize=18, fontweight="bold"
    )
    fig.text(
        0.035,
        0.88,
        "Source views from an archived 4.9-second passage · coding labels illustrate the architecture",
        fontsize=10,
        color=GRAY,
    )
    fig.text(
        0.035,
        0.055,
        "Cut boundaries and source views are recorded measurements. Object reconstruction and the complete mixed-codec stream remain to be evaluated.",
        fontsize=9,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.76, bottom=0.12)
    save(fig, output, "introduction-timeline-private")
    receipt = dict(
        source_presentation_path=str(SOURCE_REVIEW),
        source_presentation_sha256=SOURCE_SHA,
        source_broadcast_sha256="676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66",
        source_results_sha256=sha(report_path),
        presentation_shape=[245, 540, 960, 3],
        selected_ticks=[50, 120, 195],
        source_frame_indices=[543, 613, 688],
        cuts=[100, 145],
        fps=50,
        command=command,
        selected_rgb_sha256=[hashlib.sha256(frame).hexdigest() for frame in frames],
        role="source-verified display imagery with conceptual coding labels; private publication-rights review pending",
    )
    (output / "introduction-timeline-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output / "introduction-timeline-private.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ITEM)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args()
    architecture(args.output)
    if args.private_output is not None:
        private_timeline(args.private_output)
    print(args.output / "architecture.pdf")


if __name__ == "__main__":
    main()
