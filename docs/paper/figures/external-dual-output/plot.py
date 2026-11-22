"""Plot visual and machine outcomes of the same physical ball deliveries."""

import hashlib
import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[2] / "experiments/cohort-expansion-20261122/test-physical"


def load(name):
    return json.loads((RAW / name).read_text())


def hits(rows):
    positive = [r for r in rows if r["visibility"] == 1]
    negative = [r for r in rows if r["visibility"] == 0]
    assert len(positive) == 40 and len(negative) == 1
    return sum(r["present"] and r["error_native_pixels"] <= 10 for r in positive), sum(
        r["present"] for r in negative
    )


def main():
    names = ["match102-score.json", "match102-judge-portable.json", "match102-strongjudge.json"]
    scores, judge, strong = map(load, names)
    assert (
        len({v["source_sha256"] for v in [scores, judge, strong]}) == 1
        and len({v["labels_sha256"] for v in [scores, judge, strong]}) == 1
    )
    assert scores["best_affordable"]["sha256"] == strong["packet_sha256"]
    rows = []
    for name, visual, rate, locations in [
        (
            "AV1, best resolution",
            scores["best_affordable"]["ball_dists"],
            strong["kbps"],
            strong["detector"]["rows"],
        ),
        (
            "AV1, same resolution",
            mean(x["dists"] for x in scores["ordinary"]["rows"]),
            scores["ordinary_kbps"],
            judge["detections"]["ordinary"]["rows"],
        ),
        (
            "PointStream, detector",
            mean(x["dists"] for x in scores["candidate"]["rows"]),
            scores["candidate_kbps"],
            judge["detections"]["candidate"]["rows"],
        ),
        (
            "PointStream, direct",
            mean(x["dists"] for x in scores["candidate"]["rows"]),
            scores["candidate_kbps"],
            scores["direct"],
        ),
    ]:
        found, absent = hits(locations)
        rows.append(
            dict(
                name=name,
                distortion=visual,
                kbps=rate,
                within10=found,
                visible=40,
                absent_emissions=absent,
                absent_labels=1,
            )
        )
    assert [r["within10"] for r in rows] == [0, 0, 34, 36] and not any(
        r["absent_emissions"] for r in rows
    )
    ledger = dict(
        source=scores["source_sha256"],
        receipts={n: hashlib.sha256((RAW / n).read_bytes()).hexdigest() for n in names},
        rows=rows,
    )
    (HERE / "values.json").write_text(json.dumps(ledger, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 2.65))
    for i, (r, mark, col) in enumerate(
        zip(rows, ["s", "^", "o", "D"], ["#456882", "#456882", "#b04c21", "#b04c21"], strict=True)
    ):
        x = r["distortion"]
        y = r["within10"]
        ax.scatter(
            x,
            y,
            marker=mark,
            s=49,
            edgecolors=col,
            facecolors="white" if i in (1, 3) else col,
            linewidths=1.6,
            zorder=3,
        )
        offset = [(7, 8), (-7, 8), (-9, -16), (8, 3)][i]
        label = [
            f"Best AV1 resolution\n{r['kbps']:.2f} kbps",
            f"Same AV1 resolution\n{r['kbps']:.2f} kbps",
            "PointStream detector (34)",
            "Direct coordinates (36)",
        ][i]
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="right" if i in (1, 2) else "left",
            va="bottom" if i < 2 else "center",
            fontsize=8.5,
            color=col,
        )
    ax.plot([rows[2]["distortion"]] * 2, [34, 36], color="#b04c21", lw=1)
    ax.set_xlim(0.218, 0.275)
    ax.set_ylim(-2, 41)
    ax.set_ylabel("Balls localized out of 40")
    ax.set_xlabel("Ball-crop DISTS (lower is better)")
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(axis="y", color="#e3e5e4", lw=0.7)
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.22, top=0.955)
    fig.savefig(
        HERE / "quality-and-localization.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
