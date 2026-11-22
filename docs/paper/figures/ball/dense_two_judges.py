"""Two-detector matched-rate ball figure from authenticated numeric receipts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
mono = json.loads((ROOT / "docs/experiments/second-ball-judge-20261122/result.json").read_text())
wasb = json.loads(
    (ROOT / "docs/experiments/ball-matched-low-20261122/refitted-streak-detection.json").read_text()
)
track = json.loads(
    (ROOT / "docs/experiments/ball-matched-low-20261122/paid-track-scores.json").read_text()
)
assert mono["counts"] == {"1": 261, "0": 30, "-1": 1}
assert wasb["counts"] == {"visible": 261, "absent": 30, "uncertain": 1}
assert wasb["judged_rgb_hashes"]["baseline"] == mono["arms"]["baseline"]["rgb_sha256"]
assert wasb["judged_rgb_hashes"]["candidate"] == mono["arms"]["candidate"]["rgb_sha256"]
d = track["arms"]["refitted"]
assert (d["visible_hit10"], d["absent_false"], d["packet_bytes"]) == (240, 2, 77093)
labels = [
    "WASB · ordinary AV1",
    "WASB · fitted ball",
    "MonoTrack · ordinary AV1",
    "MonoTrack · fitted ball",
    "Direct transmitted position",
]
visible = [
    wasb["detections"]["baseline"]["10"],
    wasb["detections"]["candidate"]["10"],
    mono["arms"]["baseline"]["hits10"],
    mono["arms"]["candidate"]["hits10"],
    d["visible_hit10"],
]
false = [
    wasb["detections"]["baseline"]["false"],
    wasb["detections"]["candidate"]["false"],
    len(mono["arms"]["baseline"]["absent_false"]),
    len(mono["arms"]["candidate"]["absent_false"]),
    d["absent_false"],
]
colors = ["#686868", "#126b9b", "#686868", "#126b9b", "#a64b12"]
y = np.arange(len(labels))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 3.45), gridspec_kw={"width_ratios": [2.1, 1]})
for axis, values, limit, denominator in ((ax, visible, 261, 261), (bx, false, 30, 30)):
    axis.barh(y, values, color=colors, height=0.62)
    axis.set_xlim(0, limit * 1.18)
    axis.invert_yaxis()
    axis.grid(axis="x", color="#dddddd", linewidth=0.65)
    axis.set_axisbelow(True)
    for row, val in enumerate(values):
        axis.text(
            val + limit * 0.025,
            row,
            f"{val}/{denominator}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
ax.set_yticks(y, labels=labels)
bx.set_yticks(y, labels=[])
bx.tick_params(axis="y", length=0)
ax.set_title("Visible ball localized", loc="left", fontsize=11, fontweight="bold")
bx.set_title("False detections", loc="left", fontsize=11, fontweight="bold")
ax.set_xlabel("Frames within 10 px of label (higher is better)")
bx.set_xlabel("Ball absent (lower is better)")
ax.set_xticks([0, 65, 130, 195, 260])
bx.set_xticks([0, 10, 20, 30])
fig.text(
    0.14,
    0.01,
    "TrackNet game9 / Clip5 · 292 frames · ordinary 63.06 kbps; fitted 63.36 kbps",
    fontsize=9,
    color="#333333",
)
fig.subplots_adjust(left=0.25, right=0.98, top=0.87, bottom=0.23, wspace=0.28)
fig.savefig(HERE / "dense-two-judges.pdf", bbox_inches="tight")
fig.savefig(HERE / "dense-two-judges.png", dpi=180, bbox_inches="tight")
plt.close(fig)
