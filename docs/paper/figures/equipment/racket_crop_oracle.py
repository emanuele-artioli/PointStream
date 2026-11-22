"""Plot the frozen single-clip source-pixel racket oracle."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
r = json.loads((ROOT / "docs/experiments/racket-crop-gate-20261122/full.json").read_text())
source = [v for row in r["labeled_results"] for v in row["source_match"]["iou"]]
oracle = [v for row in r["labeled_results"] for v in row["oracle_match"]["iou"]]
assert len(source) == len(oracle) == 6 and r["placements"] == 70 and not r["gate"]
base = json.loads(
    (ROOT / "docs/experiments/racket-source-oracle-20261122/decoded-full.json").read_text()
)
basevals = [v for row in base["rows"] if row["source"] == "match209" for v in row["assigned_iou"]]
assert len(basevals) == 6
vals = [source, basevals, oracle]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(8.4, 3.65))
y = np.arange(3)
h = 0.31
for delta, threshold, color in [(-h / 2, 0.5, "#08718a"), (h / 2, 0.3, "#bb6c27")]:
    count = [sum(v >= threshold for v in a) for a in vals]
    ax.barh(y + delta, count, height=h, color=color, label=f"IoU ≥ {threshold}")
    for i, n in enumerate(count):
        ax.text(n + 0.12, y[i] + delta, f"{n}/6", va="center", fontsize=10, weight="bold")
ax.set_yticks(y, ["Source", "Ordinary AV1", "Ideal source-pixel paste"])
ax.invert_yaxis()
ax.set_xlim(0, 7.2)
ax.set_xticks(range(7))
ax.set_xlabel("Official racket boxes localized (higher is better)")
ax.grid(axis="x", color="#dddddd")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", frameon=False)
ax.set_title(
    "A full-cadence racket crop feasibility gate", loc="left", fontsize=14, weight="bold", pad=12
)
fig.text(
    0.24,
    0.02,
    "Match 209 · 64 frames · six boxes on five labeled frames · paste uses uncoded source pixels",
    fontsize=9,
    color="#40535f",
)
fig.subplots_adjust(left=0.28, right=0.97, top=0.82, bottom=0.22)
fig.savefig(HERE / "racket-crop-oracle.pdf", bbox_inches="tight")
fig.savefig(HERE / "racket-crop-oracle.png", dpi=180, bbox_inches="tight")
