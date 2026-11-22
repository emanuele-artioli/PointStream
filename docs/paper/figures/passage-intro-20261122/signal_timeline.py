"""Draw the transmitted points and the verified return timeline without broadcast images."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]


def main():
    data = json.loads((HERE / "packet-signals.json").read_text())
    fig = plt.figure(figsize=(7.1, 2.82), layout="constrained")
    signal, route = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1]})
    signal.set(xlim=(610, 1180), ylim=(920, 165), aspect="equal")
    signal.set_xticks([])
    signal.set_yticks([])
    signal.set_facecolor("#f2f6f6")
    signal.set_title("One rally’s transmitted points", loc="left", fontsize=10, fontweight="bold")
    for track in data["tracks"].values():
        pts = np.asarray(track)
        for i, j in EDGES:
            if pts[i, 2] > 0 and pts[j, 2] > 0:
                signal.plot(pts[[i, j], 0], pts[[i, j], 1], color="#016c77", linewidth=1.5)
        valid = pts[:, 2] > 0
        signal.scatter(pts[valid, 0], pts[valid, 1], s=6, color="#004b53", zorder=4)
    xs, ys = np.array(data["tile_centers"]).T
    signal.plot(xs, ys, ".-", color="#bf6125", markersize=3)
    signal.annotate(
        "Ball track",
        (xs[-1], ys[-1]),
        (700, 505),
        arrowprops={"arrowstyle": "-", "color": "#8f4a1c"},
        fontsize=8,
    )
    signal.text(800, 297, "Far player", fontsize=8)
    signal.text(785, 770, "Near player", fontsize=8)
    for spine in signal.spines.values():
        spine.set_color("#a1adad")
    route.set(xlim=(0, 1), ylim=(0, 1))
    route.axis("off")
    route.set_title("One broadcast timeline", loc="left", fontsize=10, fontweight="bold")
    route.plot([0.055, 0.055], [0.17, 0.84], color="#738288", linewidth=1.6)
    for y, color, title, detail in [
        (0.84, "#016c77", "First court visit", "Send appearance, court and motion."),
        (0.50, "#56636b", "216.5-second cutaway", "Send conventional video. Retain state."),
        (
            0.17,
            "#016c77",
            "Return to the court",
            "Identify the cached players. Send\nnew court and motion data.",
        ),
    ]:
        route.scatter([0.055], [y], s=42, color=color, zorder=2)
        route.text(0.12, y + 0.025, title, fontsize=9, weight="bold", va="center")
        route.text(0.12, y - 0.04, detail, fontsize=8, va="top")
    fig.savefig(
        HERE / "signal-timeline.pdf",
        metadata={"CreationDate": None},
        bbox_inches="tight",
        pad_inches=0.04,
    )


if __name__ == "__main__":
    main()
