"""Plot the measured cold-start bill for two comparable complete player-and-ball packets."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parents[2] / "experiments/clean-court-grabcut-20261119/delivery-receipt.json"
SHA256 = "347f0cc2080c3c02dbd01b8744ae24889bd95a4433dcb9dbf0c78d033d3405cb"


def components(row):
    files = row["files"]
    payload = row["components"]
    assert sum(entry["bytes"] for entry in files.values()) == row["total_bytes"]
    reference = payload["ref0"]["bytes"] + payload["ref1"]["bytes"]
    court = payload["base"]["bytes"]
    points = payload["keypoints"]["bytes"]
    ball = files["ball.sidecar"]["bytes"]
    framing = files["players.pscp"]["bytes"] - (reference + court + points)
    other = framing + files["receiver.manifest"]["bytes"]
    assert other >= 0 and sum((reference, court, points, ball, other)) == row["total_bytes"]
    return [reference, court, ball, points, other]


def main():
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    by_name = {row["name"]: row for row in json.loads(raw)["rows"]}
    rows = [by_name[name] for name in ("original", "grabcut")]
    values = [components(row) for row in rows]
    original, cleaned = values
    assert original[0] == cleaned[0] and original[2:] == cleaned[2:]
    fig, ax = plt.subplots(figsize=(8.9, 4.3))
    fig.subplots_adjust(left=.22, right=.86, bottom=.30, top=.79)
    fig.suptitle("Where the cold-start bytes go", x=.035, y=.98, ha="left", weight="bold", fontsize=17)
    fig.text(.035, .86, "Nadal–Kyrgios  ·  240 frames in 4.8 seconds  ·  both players and ball tiles", fontsize=10)
    names = ["Two lossless references", "Base video", "Ball sidecar", "Keypoints", "Framing and manifest"]
    colors = ["#324960", "#4b9283", "#d79835", "#82456c", "#c0c7cc"]
    for i, (row, parts) in enumerate(zip(rows, values)):
        offset = 0
        for label, color, count in zip(names, colors, parts):
            size = count / 1000
            ax.barh(i, size, left=offset, height=.49, color=color, label=label if i == 0 else None)
            if size > 90:
                ax.text(offset + size / 2, i, f"{size:.1f}", va="center", ha="center", color="white", weight="bold")
            offset += size
        ax.text(offset + 4, i, f"{offset:.1f} kB", va="center", fontsize=10)
    ax.set_yticks([0, 1], ["Original base", "Cleaned base"])
    ax.set_ylim(1.55, -.6)
    ax.set_xlim(0, 480)
    ax.set_xlabel("Delivered kilobytes (1 kB = 1,000 bytes)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5,.02),
               frameon=False, ncol=3, fontsize=9)
    for ext in ["pdf", "png"]:
        fig.savefig(HERE / f"byte-budget.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
