"""Render the complete fixed cohort, including unpaired or excluded source clips."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".png"), dpi=190, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figures(d, output):
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "axes.labelsize": 8,
        }
    )
    rows = d["rows"]
    pos = np.arange(len(rows))[::-1]
    labels = [r["event"].replace(", ATP Cup", "") for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.44))
    fig.subplots_adjust(left=0.19, right=0.76, top=0.87, bottom=0.17)
    palette = ["#164f72", "#b4511e"]
    ax.axvline(0, color=".55", lw=0.8, zorder=0)
    for idx, (y, r) in enumerate(zip(pos, rows, strict=True)):
        if idx % 2 == 0:
            ax.axhspan(y - 0.48, y + 0.48, color="#f1f3f5", zorder=-2)
        if r.get("candidate_dists") is not None:
            a = r["ordinary_dists"] - r["candidate_dists"]
            b = r["strongest_dists"] - r["candidate_dists"]
            ax.plot([a, b], [y + 0.12, y - 0.12], color="#777", lw=0.7, zorder=1)
            ax.scatter(
                a, y + 0.12, s=25, marker="o", edgecolor=palette[0], facecolor=palette[0], zorder=2
            )
            ax.scatter(
                b,
                y - 0.12,
                s=27,
                marker="D",
                edgecolor=palette[1],
                facecolor="white",
                lw=1.2,
                zorder=3,
            )
        txt = "unavailable"
        if "candidate_kbps" in r:
            strongest = f"{r['strongest_kbps']:.1f}" if r.get("strongest_kbps") is not None else "—"
            txt = f"{r['candidate_kbps']:.1f} / {r['ordinary_kbps']:.1f} / {strongest}"
        if r.get("selection") == "sensitivity":
            txt += " *"
        if r.get("selection") == "unmatched":
            txt += " †"
        ax.text(1.02, y, txt, va="center", transform=ax.get_yaxis_transform(), fontsize=7)
    ax.set_yticks(pos, labels)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel("Decrease in ball-crop DISTS with PointStream (positive favors PointStream)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", alpha=0.16)
    ax.text(
        1.02,
        1.02,
        "Delivered kbps\nPS / AV1 / best AV1",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color=palette[0],
                ls="none",
                label="Rate-matched AV1 at the same resolution",
            ),
            Line2D(
                [],
                [],
                marker="D",
                mfc="white",
                color=palette[1],
                ls="none",
                label="Best affordable AV1 over two resolutions",
            ),
        ],
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0, 0.983),
        fontsize=7,
        handletextpad=0.25,
        borderpad=0,
        labelspacing=0.35,
    )
    notes = []
    if any(r.get("selection") == "sensitivity" for r in rows):
        notes.append("* Price difference between 0.5% and 2%.")
    if any(r.get("selection") == "unmatched" for r in rows):
        notes.append("† Price difference above 2%.")
    if notes:
        fig.text(0.19, 0, "  ".join(notes), fontsize=7)
    save(fig, output / "official-test-ball-quality")

    fig, ax = plt.subplots(figsize=(6.2, 3.34))
    fig.subplots_adjust(left=0.20, right=0.80, top=0.88, bottom=0.15)
    for idx, (y, r) in enumerate(zip(pos, rows, strict=True)):
        if idx % 2 == 0:
            ax.axhspan(y - 0.48, y + 0.48, color="#f1f3f5", zorder=-2)
        direct = r.get("direct")
        if direct and direct["visible"]:
            ax.scatter(
                100 * direct["hit10"] / direct["visible"],
                y,
                s=26,
                marker="o",
                color=palette[0],
                zorder=3,
            )
            text = f"{direct['hit10']}/{direct['visible']}"
            if r.get("source_qualified"):
                for offset, name, marker, color in (
                    (0.18, "ordinary", "s", "#b4511e"),
                    (-0.18, "candidate", "D", "#41784d"),
                ):
                    j = r["rgb"][name]
                    ax.scatter(
                        100 * j["hit10"] / j["visible"],
                        y + offset,
                        s=23,
                        marker=marker,
                        facecolor="white",
                        edgecolor=color,
                        lw=1.05,
                        zorder=2,
                    )
        else:
            text = "0 visible" if direct else "unavailable"
        ax.text(1.02, y, text, va="center", transform=ax.get_yaxis_transform(), fontsize=7)
    ax.set_yticks(pos, labels)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Visible marks localized within 10 source pixels (%)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", alpha=0.16)
    ax.text(
        1.02, 1.01, "Direct hits /\nvisible marks", va="bottom", transform=ax.transAxes, fontsize=7
    )
    keys = [
        ("o", palette[0], "Paid coordinates"),
        ("s", "#b4511e", "AV1 + detector"),
        ("D", "#41784d", "PointStream + detector"),
    ]
    ax.legend(
        handles=[
            Line2D([], [], ls="none", marker=m, color=c, mfc=c if i == 0 else "white", label=label)
            for i, (m, c, label) in enumerate(keys)
        ],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.00),
        fontsize=7,
        handletextpad=0.25,
        columnspacing=0.75,
        borderpad=0,
    )
    save(fig, output / "official-test-ball-localization")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    figures(json.loads(a.summary.read_text()), a.output)
