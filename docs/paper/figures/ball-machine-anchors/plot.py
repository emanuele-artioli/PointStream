"""Exact-packet machine detections against tested affordable AV1 controls."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "experiments"
HASHES = {
    "209": "48773b30a0e8e2c0059b3efeb1d9dd1cce125f5c24d1edfa1b2e2b9051a867b0",
    "269": "deb053733c1c259f2d154746ae61431714eada82ed0a07ea8db37e882881e0aa",
    "235": "7e3b1651ccde187860472183d76a2fd06bc6d9b714747a2864cb693a5095e1b9",
}
SOURCE_HASHES = {
    "209": "0c6d19213639516402cf19ea6be22793675fb6ab2a7f53fbaae96359db8cca6b",
    "269": "1b15d63cbbf16ea841a55c806f653f054fff3b82c5f53ef054d404bafc5b618c",
    "235": "484b1a9a2ecf8baa799fadda902a2f1acb0bb84579337f8e1ac2056f21b2a52e",
}
NAMES = [
    "Source → WASB",
    "640 AV1 → WASB",
    "640 ROI → WASB",
    "1280 AV1 → WASB",
    "1280 ROI → WASB",
    "Tiles → WASB",
    "Direct points",
]
ARMS = [
    (None, "source"),
    (640, "plain"),
    (640, "roi"),
    (1280, "plain"),
    (1280, "roi"),
    (320, "tracked"),
    (320, "tracked"),
]
COLORS = ["#53687c", "#7995aa", "#7995aa", "#345c79", "#345c79", "#c25631", "#236b51"]
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True, constrained_layout=True)
for ax, (match, digest) in zip(axes, HASHES.items()):
    raw = (ROOT / "ball-machine-anchors-20261122" / f"{match}-result.json").read_bytes()
    source_raw = (ROOT / "ball-machine-three-20261122" / f"{match}-result.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == digest
        and hashlib.sha256(source_raw).hexdigest() == SOURCE_HASHES[match]
    )
    data = json.loads(raw)
    source = json.loads(source_raw)["results"]["source"]
    count = sum(r["visibility"] == 1 for r in source["rows"])
    for i, (width, mode) in enumerate(ARMS):
        candidates = (
            [source]
            if width is None
            else [r for r in data["results"] if r["resolution"] == width and r["mode"] == mode]
        )
        if not candidates:
            ax.text(0.03, i, "none tested fit", va="center", fontsize=8, color="0.4")
            continue
        (arm,) = candidates
        rows = [r for r in arm["rows"] if r["visibility"] == 1]
        assert len(rows) == count
        hits = sum(r["direct_hit_10px"] if i == 6 else r["hit_10px"] for r in rows)
        ax.barh(i, hits / count, color=COLORS[i], height=0.64, hatch="//" if i in (2, 4) else None)
        rate = "" if width is None or i == 6 else f" · {arm['wire_kbps']:.1f}k"
        ax.text(hits / count + 0.035, i, f"{hits}/{count}{rate}", va="center", fontsize=7.5)
    event = {"209": "Rotterdam", "269": "Washington", "235": "Indian Wells"}[match]
    ax.set_title(f"{event} · {count} visible labels", fontsize=10)
    ax.set_xlim(0, 1.38)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(NAMES)), NAMES, fontsize=8.5)
axes[0].invert_yaxis()
fig.supxlabel("Ball location within 10 native source pixels; k = actual kbps", fontsize=9)
fig.savefig(HERE / "ball-machine-anchors.pdf")
fig.savefig(HERE / "ball-machine-anchors.png", dpi=180)
