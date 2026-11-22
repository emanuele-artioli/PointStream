"""Draw a player-body rate–distortion curve from actual picture payloads and renditions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
CARDS = HERE.parents[2] / "experiments"
MODES = ("jpeg40", "jpeg70", "q65", "q78", "q90", "lossless")


def load(path):
    return json.loads(path.read_text())


def main():
    bodies = load(CARDS / "reference-body-rd-20261122/body-result.json")["summary"]["source_model"]
    jpeg = load(CARDS / "reference-body-rd-20261122/jpeg-result.json")
    bodies.update(jpeg["summary"]["source_model"])
    temporal = load(CARDS / "reference-body-rd-20261122/jpeg-temporal-result.json")["summary"]
    values = [jpeg["init"][m]["total_image_bytes"] for m in ("jpeg40", "jpeg70")] + [
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
    for metric, field, ylabel in [
        ("rd", "body_dists", "Player-body DISTS  ↓"),
        ("motion", "motion_residual_error", "Motion-compensated body luma error  ↓"),
    ]:
        fig, ax = plt.subplots(figsize=(4.25, 2.6))
        fig.subplots_adjust(left=0.175, bottom=0.23, right=0.97, top=0.94)
        for levels, color, marker, label in [
            (("jpeg40", "jpeg70"), "#b34d00", "^", "JPEG"),
            (("q65", "q78", "q90"), "#00787e", "o", "AVIF"),
            (("lossless",), "#753c77", "D", "Lossless WebP"),
        ]:
            ax.plot(
                [bills[m]["measured_image_bytes"] / 1000 for m in levels],
                [bills[m][field] for m in levels],
                color=color,
                marker=marker,
                label=label,
                markersize=5,
                linewidth=1.15,
            )
        offsets = (
            [(5, -1), (-2, -13), (4, 5), (10, 5), (4, -12)]
            if metric == "rd"
            else [(-14, -1), (4, 1), (-11, -12), (4, 3), (3, 5)]
        )
        for m, text, offset in zip(MODES[:5], ["40", "70", "65", "78", "90"], offsets, strict=True):
            ax.annotate(
                text,
                (bills[m]["measured_image_bytes"] / 1000, bills[m][field]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(xlabel="Two transmitted reference images (kB)", ylabel=ylabel, xlim=(10, 264))
        ax.set_ylim((0.165, 0.181) if metric == "rd" else (7.3, 8.7))
        ax.set_xticks([50, 100, 150, 200, 250])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=1, handlelength=1.5)
        ax.grid(axis="y", color="#d7dce0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"reference-body-{metric}.{ext}", dpi=210)
        plt.close(fig)


if __name__ == "__main__":
    main()
