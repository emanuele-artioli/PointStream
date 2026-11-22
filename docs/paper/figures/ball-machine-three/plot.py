"""Frozen source-label counts, with one panel per match and no extrapolation."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT = HERE.parents[2] / "experiments" / "ball-machine-three-20261122"
HASHES = {
    "209": "0c6d19213639516402cf19ea6be22793675fb6ab2a7f53fbaae96359db8cca6b",
    "269": "1b15d63cbbf16ea841a55c806f653f054fff3b82c5f53ef054d404bafc5b618c",
    "235": "484b1a9a2ecf8baa799fadda902a2f1acb0bb84579337f8e1ac2056f21b2a52e",
}
NAMES = ["Source → WASB", "AV1 → WASB", "ROI → WASB", "Tiles → WASB", "Direct points"]
ARMS = ["source", "plain", "roi", "tracked", "tracked"]
COLORS = ["#53687c", "#a7b5c3", "#6f8e9e", "#c25631", "#236b51"]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.75), sharey=True, constrained_layout=True)
hidden = [0] * 5
hidden_n = 0
for ax, (match, checksum) in zip(axes, HASHES.items()):
    data = (INPUT / f"{match}-result.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("receipt hash changed")
    report = json.loads(data)
    count = sum(r["visibility"] == 1 for r in report["results"]["source"]["rows"])
    hidden_n += sum(r["visibility"] == 0 for r in report["results"]["source"]["rows"])
    for index, arm in enumerate(ARMS):
        rows = report["results"][arm]["rows"]
        success = sum(bool(r["direct_hit_10px"] if index == 4 else r["hit_10px"]) for r in rows if r["visibility"] == 1)
        hidden[index] += sum(bool(r["direct_present"] if index == 4 else r["detected"]) for r in rows if r["visibility"] == 0)
        ax.barh(index, success / count, color=COLORS[index], height=.68)
        ax.text(success / count + .035, index, f"{success}/{count}", va="center", fontsize=8)
    ax.set_title(f"Match {match}\n{count} visible labels", fontsize=10)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, .5, 1], ["0%", "50%", "100%"])
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(5), NAMES, fontsize=9)
axes[0].invert_yaxis()
for i, count in enumerate(hidden):
    axes[3].barh(i, count / hidden_n, color=COLORS[i], height=.68)
    axes[3].text(count / hidden_n + .035, i, f"{count}/{hidden_n}", va="center", fontsize=8)
axes[3].set_title("Labeled not visible\nthree matches pooled", fontsize=10)
axes[3].set_xlim(0, 1.25)
axes[3].set_xticks([0, .5, 1], ["0%", "50%", "100%"])
axes[3].grid(axis="x", alpha=.2)
axes[3].set_axisbelow(True)
fig.text(.5, -.04, "Left: hits within 10 native pixels of visible labels   ·   Right: position reported when labeled not visible", ha="center", fontsize=9)
fig.savefig(HERE / "ball-machine-three.pdf", bbox_inches="tight")
fig.savefig(HERE / "ball-machine-three.png", dpi=180, bbox_inches="tight")
