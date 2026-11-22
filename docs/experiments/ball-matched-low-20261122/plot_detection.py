"""Plot decoded-video detection and directly received coordinates at matched rates."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--refitted", type=Path, required=True)
    parser.add_argument("--paid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed, refitted, paid = [json.loads(path.read_text()) for path in
                             (args.fixed, args.refitted, args.paid)]
    if fixed["counts"] != {"visible": 261, "absent": 30, "uncertain": 1} or fixed["counts"] != refitted["counts"]:
        raise ValueError("Unexpected labels or denominators")
    rows = [
        ("AV1 · WASB on decoded video", fixed["detections"]["baseline"], fixed["baseline_bytes"], "#19639b"),
        ("AV1 + fixed streak · WASB", fixed["detections"]["candidate"], fixed["candidate_bytes"], "#cf740d"),
        ("AV1 + refitted streak · WASB", refitted["detections"]["candidate"], refitted["candidate_bytes"], "#b5640a"),
        ("AV1 + refitted streak · direct points", {"10": paid["arms"]["refitted"]["visible_hit10"],
          "false": paid["arms"]["refitted"]["absent_false"]}, paid["arms"]["refitted"]["packet_bytes"], "#087e6b"),
    ]
    if rows[2][2] != rows[3][2] or any(abs(row[2] / rows[0][2] - 1) > .005 for row in rows[1:]):
        raise ValueError("The packet rates are not matched")
    fig, ax = plt.subplots(figsize=(10, 5.1))
    fig.suptitle("Ball localization at nearly equal total rate", x=.04, ha="left", fontsize=16, weight="bold")
    fig.text(.04, .89, "One 292-frame clip · 261 visible and 30 absent frames · 10 px tolerance", fontsize=9.5)
    for index, (label, row, size, color) in enumerate(rows):
        y = len(rows) - 1 - index
        percent = 100 * row["10"] / 261
        rate = size * 240 / 292000
        ax.barh(y, percent, color=color, height=.4)
        ax.text(min(percent + 1, 101), y, f'{row["10"]}/261 · {percent:.1f}%', va="center", fontsize=10)
        ax.text(0, y - .29, f'{rate:.3f} kbps · {row["false"]}/30 false when absent',
                fontsize=9, color=color, va="center")
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])
    ax.set(xlim=(0, 115), ylim=(-.6, len(rows) - .5), xlabel="Visible frames with the ball correctly located (%)")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout(rect=(0,0,1,.85))
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
