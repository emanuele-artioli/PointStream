"""Plot physical operating points and the measured low-rate player front on the held-out shot."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np

HERE = Path(__file__).resolve().parent
STYLE = {
    "PointStream": ("#146865", "o"),
    "DCVC-UF": ("#3b5ea4", "s"),
    "SVT-AV1": ("#ae5d1d", "^"),
    "SVT-AV1 ROI": ("#73417f", "D"),
}


def main(a):
    s = json.loads(a.data.read_text())
    assert (
        s["primary_joint"]
        and len([r for r in s["operating_points"] if r["codec"] == "PointStream"]) == 3
    )
    a.out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5.35, 3.5), layout="constrained")
    for name, (color, mk) in STYLE.items():
        allrows = sorted(
            [r for r in s["operating_points"] if r["codec"] == name and 8.5 < r["kbps"] < 37],
            key=lambda r: (r["bytes"], r["dists"]),
        )
        if not allrows:
            continue
        best = []
        for r in allrows:
            if not best or r["dists"] < best[-1]["dists"] - 1e-8:
                best.append(r)
        # A dominated field still represents a physical observation. All PointStream settings
        # are visible because they independently vary the low-resolution background price.
        ax.scatter(
            [r["kbps"] for r in allrows],
            [r["dists"] for r in allrows],
            marker=mk,
            s=17,
            color=color,
            alpha=0.42,
            linewidths=0.45,
        )
        ax.plot(
            [r["kbps"] for r in best],
            [r["dists"] for r in best],
            color=color,
            marker=mk,
            ms=4.5,
            lw=1.35,
            label=name,
        )
    cutoff = max(r["kbps"] for r in s["operating_points"] if r["codec"] == "PointStream")
    ax.axvline(cutoff, color="#626262", ls=(0, (2.1, 2.3)), lw=0.8, zorder=0)
    ax.set(
        xlabel="Total delivered rate with cached appearance (kbps)",
        ylabel="Player-region DISTS (lower is better)",
        xscale="log",
        xlim=(8.7, 36),
    )
    ticks = [10, 15, 20, 25, 30, 35]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.grid(axis="y", color="#d9dcdd", lw=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncols=2, loc="best")
    for fmt in ("pdf", "png"):
        fig.savefig(a.out / f"shanghai-player-rate.{fmt}", dpi=210)
    plt.close(fig)
    assert np.isfinite(cutoff)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
