"""Show how many active coded seconds a paid reference requires to meet a rate cap."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"


def load(p):
    return json.loads(p.read_text())


def main():
    receipt = load(EXP / "starved-tile-budget-20261122/result.json")
    row = receipt["rungs"][receipt["chosen"]]
    fixed = load(EXP / "starved-base-local-20261122/base-receipt.json")
    pair = next(r for r in fixed["outputs"] if r["arm"] == "clean" and r["crf"] == 60)
    assert pair["reference_bytes"] == 242317
    ladder = load(EXP / "reference-ladder-20261122/result.json")
    options = [
        (f"AVIF Q{r['quality']}", r["total_bytes"])
        for r in ladder["probes"]
        if r["codec"] == "avif" and r["quality"] in (46, 65, 78)
    ]
    options.sort(key=lambda r: r[1])
    options.append(("Lossless pair", pair["reference_bytes"]))
    fig, ax = plt.subplots(figsize=(8.1, 4.45))
    durations = np.geomspace(4.8, 300, 250)
    for (label, n), color, style in zip(
        options, ("#775399", "#b77328", "#0d817b", "#203e63"), (":", "-.", "--", "-"), strict=True
    ):
        ax.plot(
            durations,
            row["running"] + n * 8 / (1000 * durations),
            color=color,
            ls=style,
            label=f"{label} · {n:,} bytes",
        )
    ax.axhline(125, color="#555", lw=0.8, alpha=0.7)
    ax.text(6.5, 128, "125 kbps budget", fontsize=8, color="#414141")
    ax.axhline(row["running"], color="#888", lw=0.9, ls=":")
    ax.text(6.5, 54.8, "52.29 kbps running floor", fontsize=8, color="#414141")
    ax.set_xscale("log")
    ax.set_yscale("log")
    rates = [50, 65, 100, 125, 200, 300, 450]
    ax.set_ylim(48, 500)
    ax.set_yticks(rates, [str(n) for n in rates])
    ax.set_xticks([4.8, 10, 20, 40, 80, 160, 300], ["4.8", "10", "20", "40", "80", "160", "300"])
    ax.set_xlabel("Eligible seconds served by one reference pair")
    ax.set_ylabel("Modeled total bitrate (kbps)")
    ax.set_title(
        "How long must a player reference remain useful?", loc="left", weight="bold", pad=15
    )
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.22)
    fig.text(
        0.12,
        0.035,
        "Fixed running cost from one measured rally. Idle time between shots contributes no eligible seconds.",
        fontsize=8,
    )
    for form in ("png", "pdf"):
        fig.savefig(HERE / f"reuse-budget.{form}", dpi=180)
    plt.close(fig)
    values = {
        label: {
            str(cap): (8 * n / 1000 / (cap - row["running"]) if cap > row["running"] else None)
            for cap in [20, 50, 65, 125]
        }
        for label, n in options
    }
    (HERE / "reuse-duration.json").write_text(json.dumps(values, indent=2) + "\n")


if __name__ == "__main__":
    main()
