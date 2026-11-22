"""Plot the independently replayed two-shot reference-channel packet sizes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "docs/experiments/cross-shot-reference-20261122"


def main():
    png = json.loads((INPUT / "channel-png-audit.json").read_text())
    webp = json.loads((INPUT / "channel-webp-audit.json").read_text())
    for data in (png, webp):
        assert data["passed"] and data["corrupt_mapping_rejected"]
        assert data["observed_shot_seconds"] == .96
        assert data["separation_seconds_between_starts"] == 221.3
        assert data["total_reuse_bytes"] == data["initial_bytes"] + data["return_reuse_bytes"]
        assert data["total_fresh_bytes"] == data["initial_bytes"] + data["return_refresh_bytes"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=.26, right=.90, top=.74, bottom=.25)
    fig.suptitle("How many reference bytes does a return require?",
                 x=.02, y=.97, ha="left", fontsize=16, weight="bold")
    fig.text(.02, .86, "Two 24-frame shots  ·  two player references  ·  actual packet bytes", fontsize=10)
    rows = [("PNG · refresh", png, "return_refresh_bytes"),
            ("PNG · reuse", png, "return_reuse_bytes"),
            ("WebP · refresh", webp, "return_refresh_bytes"),
            ("WebP · reuse", webp, "return_reuse_bytes")]
    for index, (label, result, field) in enumerate(rows):
        initial = result["initial_bytes"] / 1000
        returning = result[field] / 1000
        ax.barh(index, initial, color="#27617a", height=.59, label="Initial references" if index == 0 else None)
        ax.barh(index, returning, left=initial, color="#e89936", height=.59,
                label="Return mapping or refresh" if index == 0 else None)
        amount = f"{initial + returning:.1f} kB"
        if field == "return_reuse_bytes":
            amount += f"  ({result[field]} B return)"
        ax.text(initial + returning + 8, index, amount, va="center", fontsize=10)
    ax.set_yticks(range(4), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 760)
    ax.set_xlabel("Delivered reference-channel bytes, thousands")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.50), ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"crossshot-reference.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
