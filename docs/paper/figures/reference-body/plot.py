"""Draw a player-body rate–distortion curve from actual picture payloads and renditions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
CARDS = HERE.parents[2] / "experiments"
MODES = ("q65", "q78", "q90", "lossless")


def load(path):
    return json.loads(path.read_text())


def main():
    bodies = load(CARDS / "reference-body-rd-20261122/body-result.json")["summary"]["source_model"]
    temporal = load(CARDS / "reference-body-rd-20261122/temporal-result.json")["summary"]
    values = [
        sum(
            r["avif_bytes"] for r in load(CARDS / "common-reference-20261122/prepared.json")["rows"]
        ),
        sum(
            r["avif_bytes"]
            for r in load(CARDS / "common-reference-q78-20261122/prepared.json")["rows"]
        ),
        load(CARDS / "common-reference-q90-20261122/image-preflight.json")["total_image_bytes"],
        load(CARDS / "common-reference-q90-20261122/image-preflight.json")[
            "lossless_reference_bytes"
        ],
    ]
    base = load(CARDS / "joint-trajectory-20261122/result.json")["streams"]["20"]
    extra = base["reference_bytes"] - values[-1]
    assert extra == 2022 and base["kbps"] == 21.545
    bills = {}
    for mode, size in zip(MODES, values, strict=True):
        init = size + extra
        bills[mode] = dict(
            measured_image_bytes=size,
            comparison_initialization_bytes=init,
            body_dists=bodies[mode]["dists"],
            body_psnr=bodies[mode]["rgb_psnr"],
            motion_residual_error=temporal[mode],
            kbps_by_active_seconds={
                str(t): base["kbps"] + 8 * init / (1000 * t) for t in (4.8, 30, 120, 300)
            },
            active_seconds_at_33_kbps=8 * init / ((33 - base["kbps"]) * 1000),
        )
    (CARDS / "reference-body-rd-20261122/reference-rate-quality.json").write_text(
        json.dumps(
            dict(
                scope="Actual measured image bytes and 24-frame body quality on one development pair. "
                "Cost-duration scenarios use the 21.545 kbps measured running price, keep 2,022 other initialization bytes fixed, and assume valid reuse for the stated duration. "
                "Only the lossless full 240-frame stream is physically decoded here.",
                levels=bills,
            ),
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(figsize=(4.75, 2.75))
    fig.subplots_adjust(left=0.16, bottom=0.24, right=0.97, top=0.93)
    xs = [v / 1000 for v in values]
    ys = [bodies[m]["dists"] for m in MODES]
    ax.plot(xs[:3], ys[:3], color="#00787e", marker="o", markersize=5, linewidth=1.4)
    ax.scatter(xs[-1], ys[-1], color="#7a4290", marker="D", s=32, zorder=3)
    for i, (label, offset) in enumerate(
        zip(
            ("AVIF Q65", "Q78", "Q90", "Lossless WebP"),
            ((0, 8), (12, 6), (0, -13), (-2, 8)),
            strict=True,
        )
    ):
        ax.annotate(
            label,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=offset,
            ha="right" if i == 3 else "center",
            fontsize=8,
        )
    ax.set(
        xlabel="Two transmitted reference images (kB)",
        ylabel="Player-body DISTS  ↓",
        ylim=(0.1705, 0.1781),
        xlim=(12, 265),
    )
    ax.set_xticks([50, 100, 150, 200, 250])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color="#d7dce0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"reference-body-rd.{ext}", dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    main()
