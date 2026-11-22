"""Summarize the fixed source and decoded racket detector oracle."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "docs/experiments/racket-source-oracle-20261122"
inputs = [
    ("Source", "full.json"),
    ("Ordinary AV1", "decoded-full.json"),
    ("ROI AV1", "roi-full.json"),
    ("AV1 + ball tiles", "tracked-full.json"),
]
names = ["match209", "match235", "match269"]
counts = {}
rates = {}
for name in names:
    receipt = json.loads(
        (
            ROOT
            / f"docs/experiments/ball-confirmation-starved-20261122/racket{name[5:]}-result.json"
        ).read_text()
    )
    rates[name] = {r["mode"]: r["kbps"] for r in receipt["results"]}
for label, f in inputs:
    data = json.loads((RUN / f).read_text())
    counts[label] = {}
    for name in names:
        values = [v for row in data["rows"] if row["source"] == name for v in row["assigned_iou"]]
        counts[label][name] = (sum(v >= 0.5 for v in values), len(values))
assert [counts["Source"][n] for n in names] == [(6, 6), (4, 4), (4, 6)]
assert all(counts[label][n][0] == 0 for label, _ in inputs[1:] for n in names)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(9.8, 3.8))
ax.axis("off")
fig.text(
    0.08,
    0.92,
    "Can a detector find the racket at the source?",
    fontsize=17,
    weight="bold",
    color="#172b3a",
)
fig.text(
    0.08,
    0.845,
    "Frozen YOLO11x · confidence ≥ 0.25 · one-to-one box IoU ≥ 0.5",
    fontsize=10,
    color="#40535f",
)
mode = {"Ordinary AV1": "plain", "ROI AV1": "roi", "AV1 + ball tiles": "tracked"}
body = []
for label, _ in inputs:
    row = []
    for n in names:
        hit, total = counts[label][n]
        rate = "" if label == "Source" else f"\n{rates[n][mode[label]]:.2f} kbps"
        row.append(f"{hit}/{total}{rate}")
    body.append(row)
table = ax.table(
    cellText=body,
    rowLabels=[x[0] for x in inputs],
    colLabels=["Rotterdam · 209", "Washington · 235", "Indian Wells · 269"],
    cellLoc="center",
    rowLoc="right",
    bbox=[0.21, 0.18, 0.75, 0.69],
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#d4dde2")
    if row == 0:
        cell.set_facecolor("#e8f0f3")
        cell.set_text_props(weight="bold")
    elif col >= 0 and row == 1:
        cell.set_facecolor("#d5eef2")
        cell.set_text_props(weight="bold")
    elif col >= 0:
        cell.set_facecolor("#f5f7f8")
fig.text(
    0.08,
    0.12,
    "16 official boxes on 15 source-labeled frames · actual total rates shown for each decoded stream",
    fontsize=9,
)
fig.text(
    0.08,
    0.075,
    "Sparse-label feasibility; no racket encoder, grip assessment, or unlabeled-frame false-detection rate.",
    fontsize=9,
    color="#40535f",
)
fig.savefig(HERE / "racket-detection.pdf", bbox_inches="tight")
fig.savefig(HERE / "racket-detection.png", dpi=180, bbox_inches="tight")
