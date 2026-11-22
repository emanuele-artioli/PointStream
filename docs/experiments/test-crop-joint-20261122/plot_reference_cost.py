"""Standalone cost sensitivity for the prespecified startup formats on the held-out Shanghai shot."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

HERE = Path(__file__).resolve().parent


def main(out):
    r = json.loads((HERE / "match102-reference-cost.json").read_text())
    rows = {v["setting"]: v for v in r["rows"]}
    cases = [
        ("png", "PNG", "#272727", "--", "o"),
        ("webp", "WebP lossless", "#9b59a0", "-.", "s"),
        ("avif90", "AVIF Q90", "#2856a0", "-", "D"),
        ("avif78", "AVIF Q78", "#bd5a08", "-", "^"),
        ("avif65", "AVIF Q65", "#087555", "-", "v"),
    ]
    t = np.geomspace(12, 1100, 250)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.65, 3.45), layout="constrained")
    for key, label, color, style, mk in cases:
        row = rows[key]
        surcharge = 8 * row["init_bytes"] / 1000
        target = row["seconds_to_fit_20kbps"]
        assert abs(r["warm_run_kbps"] + surcharge / target - 20) < 0.0001
        ax.plot(
            t,
            r["warm_run_kbps"] + surcharge / t,
            color=color,
            linestyle=style,
            lw=1.7,
            label=f"{label}  ({target:.0f} s)",
        )
        ax.plot(target, 20, color=color, marker=mk, ms=5.5, linestyle="none", zorder=4)
    ax.axhline(20, lw=0.75, color="#555", ls=(0, (2, 2)), zorder=0)
    ax.set(
        xscale="log",
        xlim=(12, 1100),
        ylim=(16.8, 57),
        xlabel="Duration charged for one reference pair (s)",
        ylabel="Modeled total delivery rate (kbps)",
    )
    ticks = [15, 30, 60, 120, 300, 600, 1000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_yticks([20, 30, 40, 50])
    ax.legend(
        title="Reference  (time to reach 20 kbps)",
        fontsize=7.6,
        title_fontsize=8.2,
        loc="upper right",
        frameon=False,
        ncols=1,
        handlelength=2.2,
        columnspacing=1,
        labelspacing=0.36,
    )
    ax.grid(axis="y", lw=0.45, color="#ddd")
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"shanghai-startup-duration.{ext}", dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
