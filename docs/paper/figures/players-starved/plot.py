"""Plot paired low-rate player results from the authenticated W9 audit."""
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / "docs/experiments/w9-inputs/w7-repaired-2h.json"
INPUT_SHA = "5245d83df6f8ee16ade28174f6f3c46a94b42ae4d32520c2165fa602e83e9d86"
GROUPS = (
    ("al", "Alcaraz–Hurkacz", "#0072B2", "o"),
    ("med", "Medvedev–Struff", "#D55E00", "s"),
    ("nad", "Nadal–Kyrgios", "#009E73", "^"),
)


def main():
    data = INPUT.read_bytes()
    assert hashlib.sha256(data).hexdigest() == INPUT_SHA
    result = json.loads(data)
    rows = result["tier1"]["windows"] + result["tier2"]["windows"]
    assert len(rows) == 9
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    summaries = []
    for prefix, label, color, marker in GROUPS:
        all_match = [row for row in rows if row["window"].startswith(prefix)]
        matched = [row for row in all_match if "dcvc" in row["at_hybrid_rate"]]
        assert len(all_match) == 3 and len(matched) == (2 if prefix == "med" else 3)
        for row in matched:
            x = row["rate_kbps"]
            ps = row["hybrid"]["dists"]["foreground"]
            anchor = row["at_hybrid_rate"]["dcvc"]["foreground_dists"]
            assert row["dcvc_bracket_kbps"][0] < x < row["dcvc_bracket_kbps"][1]
            ax.plot([x, x], [ps, anchor], color=color, alpha=.32, linewidth=1)
            ax.scatter(x, ps, c=color, marker=marker, s=35, alpha=.35, zorder=3)
            ax.scatter(x, anchor, facecolors="white", edgecolors=color, marker=marker,
                       s=35, alpha=.45, linewidths=1, zorder=3)
        rate = statistics.mean(row["rate_kbps"] for row in matched)
        ps_mean = statistics.mean(row["hybrid"]["dists"]["foreground"] for row in matched)
        dcvc_mean = statistics.mean(row["at_hybrid_rate"]["dcvc"]["foreground_dists"] for row in matched)
        ax.plot([rate, rate], [ps_mean, dcvc_mean], color=color, linewidth=1.5)
        ax.scatter(rate, ps_mean, c=color, marker=marker, s=100, edgecolors="white",
                   linewidths=.7, zorder=5)
        ax.scatter(rate, dcvc_mean, facecolors="white", edgecolors=color, marker=marker,
                   s=100, linewidths=2, zorder=5)
        summaries.append({"match": label, "windows": [r["window"] for r in matched],
                          "mean_modeled_kbps": rate, "pointstream": ps_mean,
                          "dcvc_interpolated": dcvc_mean, "difference": dcvc_mean - ps_mean})
    ax.set_xscale("log")
    ax.set_xticks([15, 20, 30, 40, 50, 60], ["15", "20", "30", "40", "50", "60"])
    ax.set_xlim(14, 65)
    ax.set_ylim(.21, .35)
    ax.set_xlabel("Total bitrate (kbps), with two-hour modeled reuse")
    ax.set_ylabel("Player-region DISTS (lower is better)")
    ax.grid(color="#dfe5e8", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    match_handles = [Line2D([], [], marker=marker, linestyle="", color=color, label=label, markersize=8)
                     for _, label, color, marker in GROUPS]
    match_legend = ax.legend(handles=match_handles, loc="upper left", frameon=False, fontsize=8.7)
    ax.add_artist(match_legend)
    style_handles = [Line2D([], [], marker="o", linestyle="", color="#303941",
                            label="PointStream scored output", markersize=7),
                     Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                            markeredgecolor="#303941", label="DCVC at the same rate, interpolated", markersize=7)]
    ax.legend(handles=style_handles, loc="lower right", frameon=False, fontsize=8.7)
    fig.suptitle("Player fidelity in the measured low-rate windows", x=.10, ha="left",
                 fontsize=16, weight="bold")
    fig.text(.10, .885, "Eight windows, three matches · large markers give each match's mean", fontsize=9.5)
    fig.text(.10, .025, "Each pair uses its own window and actual modeled rate; means span different rates.\n"
                       "80 sampled frames per window · excluded Med window has no DCVC bracket · rate omits some framing.",
             fontsize=8.5)
    fig.subplots_adjust(left=.10, right=.98, top=.79, bottom=.22)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"players-starved.{ext}", dpi=190, bbox_inches="tight")
    receipt = {"input": str(INPUT.relative_to(ROOT)), "input_sha256": INPUT_SHA,
               "metric": "player-region DISTS; lower is better",
               "rate": "2-hour modeled asset reuse; not operational lifetime",
               "aggregation": "equal windows within each match; equal means across 3 physical matches",
               "excluded": "med000 has no DCVC data",
               "matches": summaries,
               "equal_match_mean_difference": statistics.mean(r["difference"] for r in summaries)}
    (HERE / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
