"""Compare paid machine readout and detection on frozen Shanghai displays."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
CARD = HERE.parents[2] / "experiments/shanghai-prospective-20261122"


def read(name):
    return json.loads((CARD / name).read_text())


def summarize(rows, direct=False):
    visible = [r for r in rows if r.get("visibility", r.get("visible")) == 1]
    hidden = [r for r in rows if r.get("visibility", r.get("visible")) == 0]
    assert len(visible) == 26 and len(hidden) == 23
    error = "direct_error_native_px" if direct else "error_native_px"
    emitted = "direct_present" if direct else "detected"
    hits = sum(r[error] is not None and r[error] <= 10 for r in visible)
    scores = [r["ball_crop_dists"] for r in visible if r["ball_crop_dists"] is not None]
    return hits, sum(r[emitted] for r in hidden), mean(scores) if scores else None


def main():
    frozen = read("primary-display-result.json")
    primary = read("primary-score-result.json")
    arms = {item["mode"]: item for item in primary["results"]}
    cohort = [(r["tick"], r["visibility"]) for r in arms["source"]["rows"]]
    assert primary["frames"] == frozen["frames"] == 240
    assert primary["rgb_sha256"] == frozen["rgb_sha256"]
    data = []

    def add(name, rate, rows, psnr=None, direct=False):
        assert [(r["tick"], r.get("visibility", r.get("visible"))) for r in rows] == cohort
        hits, emissions, dists = summarize(rows, direct)
        data.append(
            [
                name,
                "—" if rate is None else f"{rate:.2f}",
                f"{hits}/26",
                f"{emissions}/23",
                "—" if direct or dists is None else f"{dists:.3f}",
                "—" if psnr is None else f"{psnr:.2f}",
            ]
        )

    add("Source video", None, arms["source"]["rows"])
    for mode, name in [("ordinary", "AV1"), ("candidate", "PointStream video")]:
        rate = frozen[mode]["bytes"] / 1200
        add(name, rate, arms[mode]["rows"], arms[mode]["rgb_psnr"])
    add(
        "PointStream coordinates",
        frozen["candidate"]["bytes"] / 1200,
        arms["candidate"]["rows"],
        direct=True,
    )
    for scale, name in [("half", "DCVC-UF · 480 × 272"), ("full", "DCVC-UF · 960 × 544")]:
        arm = read(f"neural-score-{scale}-result.json")
        assert (
            arm["frames"] == primary["frames"] and arm["label_sha256"] == primary["labels_sha256"]
        )
        assert arm["judge_checkpoint_sha256"] == primary["wasb_checkpoint_sha256"]
        assert arm["source_rgb_sha256"] == frozen["rgb_sha256"]["source"]
        add(name, arm["kbps"], arm["rows"], arm["whole_rgb_psnr"])

    fig, ax = plt.subplots(figsize=(9.6, 4.05))
    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.25, top=0.79)
    table = ax.table(
        cellText=data,
        colLabels=[
            "Receiver input",
            "Rate\n(kbps)",
            "Visible ball\nlocalized ↑",
            "Not visible\nemissions ↓",
            "Ball crop\nDISTS ↓",
            "Full frame\nPSNR (dB) ↑",
        ],
        colWidths=[0.27, 0.10, 0.16, 0.17, 0.15, 0.15],
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("#e6eef2" if i == 0 else "#e4f2ef" if i == 4 else "white")
        if i == 0 or i == 4:
            cell.get_text().set_weight("bold")
        if j == 0:
            cell.get_text().set_ha("left")
    fig.text(
        0.035,
        0.935,
        "Can a machine locate the ball in a new Shanghai shot?",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.035,
        0.867,
        "9.6 seconds · 240 consecutive frames · 49 official labels · codec settings frozen before scoring",
        fontsize=10,
    )
    fig.text(
        0.035,
        0.17,
        "Video rows use the same WASB detector. A hit lies within 10 original pixels. The two PointStream rows use the same packet.",
        fontsize=9,
    )
    fig.text(
        0.035,
        0.108,
        "PointStream and AV1 differ by 0.37% in total bytes. Neural controls show their measured rates. The other 191 frames have no label.",
        fontsize=9,
    )
    fig.text(
        0.035,
        0.046,
        "WASB found 5/26 on source, below the 80% gate. Detector counts are diagnostic. Official labels judge the transmitted positions.",
        fontsize=9,
    )
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"shanghai-prospective-ball.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
