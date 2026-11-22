"""Compare the measured point decisions with author camera labels and the replay schedule."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments/passage-routing-20261122"


def main():
    d = json.loads((EXP / "routing.json").read_text())
    start = d["motion"][0]["start"]
    stop = d["motion"][-1]["end"]
    colors = {
        "point": "#126784",
        "interlude": "#e4b363",
        "other": "#c8ccd0",
        "court": "#487a54",
        "crowd": "#bb5a45",
        "unreviewed": "#eceef0",
        "ordinary AV1": "#c8ccd0",
        "object mode": "#126784",
        "non_court": "#bb5a45",
    }
    sources = [
        ("Historical point classifier", [(v["start"], v["end"], v["route"]) for v in d["motion"]]),
        ("Author camera labels", [(v["start"], v["end"], v["label"]) for v in d["human"]]),
    ]
    results = []
    for name in ("rallies", "crowd_point", "crowd_interlude"):
        p = EXP / (name + "-result.json")
        if p.exists():
            r = json.loads(p.read_text())
            for row in r["records"].values():
                results.extend((v["start"], v["end"], v["predicted"]) for v in row["parts"])
    if results:
        sources.append(("Geometry (tested spans only)", results))
    sources.append(
        (
            "Two-rally experiment",
            [
                (start, start + 240, "object mode"),
                (start + 240, stop - 240, "ordinary AV1"),
                (stop - 240, stop, "object mode"),
            ],
        )
    )
    fig, ax = plt.subplots(figsize=(10, 4.45))
    fig.subplots_adjust(left=0.265, right=0.965, bottom=0.34, top=0.83)
    for y, (_, spans) in enumerate(sources):
        for a, b, group in spans:
            ax.barh(
                y,
                (b - a) / 50,
                left=(a - start) / 50,
                color=colors[group],
                height=0.47,
                edgecolor="none",
                hatch="///" if group == "crowd" else None,
            )
    ax.set_ylim(len(sources) - 0.55, -0.45)
    ax.set_xlim(0, (stop - start) / 50)
    ax.set_xticks([0, 50, 100, 150, 200, 226.1], ["0", "50", "100", "150", "200", "226.1"])
    ax.set_yticks(range(len(sources)), [s[0] for s in sources])
    ax.set_xlabel("Elapsed broadcast time (seconds)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Which parts of the passage take the object path?",
        x=0.035,
        y=0.965,
        ha="left",
        weight="bold",
        fontsize=16,
    )
    handles = [
        Patch(facecolor=colors[k], label=v)
        for k, v in [
            ("point", "Point prediction or object mode"),
            ("interlude", "Interlude"),
            ("other", "Other or ordinary AV1"),
            ("court", "Court view"),
            ("crowd", "Crowd or noncourt"),
            ("unreviewed", "Unreviewed camera view"),
        ]
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=3,
        bbox_to_anchor=(0.027, 0.085),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.04,
        0.035,
        "Blank geometry spans were not evaluated; the crowd interlude contributes only its first 0.48 s.",
        fontsize=9,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("passage-routing." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
