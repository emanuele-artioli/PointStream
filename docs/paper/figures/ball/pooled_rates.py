"""Draw existing ball quality by total measured rate, equally weighted by match."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "docs/experiments/submission-ladder-ball-confirmation-analysis-20261115.json"
SHA = "e60ca013c6518a71f9059d029a3cf5bb90711edf6d262ad1ebb430887680a6d5"
SOURCES = {
    "match209_000": (64, 25, "o"),
    "match235_000": (154, 60, "s"),
    "match269_000": (64, 25, "^"),
}
FAMILIES = {
    "native_svt": ("SVT-AV1", "#19639b"),
    "roi_fitted": ("AV1 with fitted ROI", "#a96a04"),
    "native_dcvc": ("DCVC-UF", "#7d4f91"),
    "ptp": ("AV1 + tracked ball tiles", "#087e6b"),
}


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError("Confirmation analysis hash changed")
    source_rows = json.loads(raw)["sources"]
    if {r["source_id"] for r in source_rows} != set(SOURCES):
        raise ValueError("Unexpected match inventory")
    fig, ax = plt.subplots(figsize=(9.1, 5.2))
    domains = {}
    for family, (label, color) in FAMILIES.items():
        curves = []
        for source in source_rows:
            count, fps, marker = SOURCES[source["source_id"]]
            if source["frames"] != count:
                raise ValueError("Frame count changed")
            rows = [p for p in source["physical_curves"] if p["family"] == family]
            rows = sorted(rows, key=lambda p: p["bytes"])
            if len(rows) != 3:
                raise ValueError("Expected three actual operating points")
            x = np.array([p["bytes"] * 8 * fps / count / 1000 for p in rows])
            y = np.array([p["means"]["crop_dists"] for p in rows])
            if not np.all(np.diff(x) > 0):
                raise ValueError("Rates must be distinct")
            ax.scatter(x, y, marker=marker, facecolors="white", edgecolors=color,
                       linewidths=1.2, s=28, alpha=.7, zorder=3)
            curves.append((x, y))
        start, end = max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves)
        if start >= end:
            raise ValueError("No overlap across all three matches")
        rates = np.linspace(start, end, 120)
        averaged = np.mean([PchipInterpolator(x, y, extrapolate=False)(rates)
                            for x, y in curves], axis=0)
        ax.plot(rates, averaged, color=color, label=label, linewidth=2.8)
        domains[family] = [float(start), float(end)]
    ax.set(xlabel="Complete bitrate (kbps)", ylabel="Ball-crop DISTS  ↓", xlim=(0, 930))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Ball fidelity across three tennis matches", x=.08, ha="left",
                 fontsize=15, weight="bold")
    fig.text(.08, .885, "Lines average matches only where all three have measured support", fontsize=9)
    fig.text(.08, .04, "○ match 209     □ match 235     △ match 269  ·  32 visible labeled instants in total",
             fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .86))
    fig.savefig(HERE / "ball-three-match-rates.pdf", bbox_inches="tight")
    fig.savefig(HERE / "ball-three-match-rates.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(json.dumps({"overlap_kbps": domains}, indent=2))


if __name__ == "__main__":
    main()
