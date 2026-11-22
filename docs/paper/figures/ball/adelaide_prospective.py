"""Prospective Adelaide metadata localization at fixed predeclared bytes-only ladder."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
r = json.loads(
    (ROOT / "docs/experiments/ball-adelaide-prospective-20261122/machine-result.json").read_text()
)
results = {x["mode"]: x for x in r["results"]}
items = [
    ("Transmitted centers", results["metadata"], "direct_native_error", "#CC79A7"),
    ("WASB on decoded video", results["ordinary"], "native_error", "#D55E00"),
    ("WASB on source", results["source"], "native_error", "#777777"),
]
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
y = np.arange(len(items))
for i, (label, row, key, color) in enumerate(items):
    visible = [x for x in row["rows"] if x["visibility"] == 1]
    notvisible = [x for x in row["rows"] if x["visibility"] == 0]
    hits = sum(x[key] is not None and x[key] <= 10 for x in visible)
    presence = sum(
        x["direct_present"] if key == "direct_native_error" else x["detected"] for x in notvisible
    )
    ax.barh(i, hits, color=color, height=0.55)
    ax.text(hits + 0.1, i, f"{hits}/5 visible; {presence}/7 labeled not-visible", va="center")
ax.set(
    yticks=y,
    yticklabels=[x[0] for x in items],
    xlim=(0, 8.5),
    xticks=[0, 1, 2, 3, 4, 5],
    xlabel="Visible labeled frames localized within 10 native pixels",
    title="Prospective Adelaide: machine location\nAV1 40.806 kbps; AV1 + centers 41.213 kbps; same pixels",
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.2)
out = Path(__file__).with_name("adelaide-prospective-machine")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=180)
