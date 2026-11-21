"""Rebuild ball evidence figures from verified PointStream experiment receipts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "docs/experiments"
PLAIN, TILE = "#4a6072", "#00796d"


def read(path):
    return json.loads((EXP / path).read_text())


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def polish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e6e9", linewidth=.7)
    ax.tick_params(length=0, pad=5)


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_pairs(ax, values, labels, decimals, label, bounds):
    for i, (a, b) in enumerate(values):
        y = len(values) - 1 - i
        ax.plot([a, b], [y, y], color="#bbc4ca", linewidth=2, zorder=1)
        ax.scatter([a, b], [y, y], color=[PLAIN, TILE], s=36, zorder=2)
        ax.annotate(f"{a:.{decimals}f}", (a, y), (0, 10), textcoords="offset points",
                    ha="center", color=PLAIN, fontsize=8)
        ax.annotate(f"{b:.{decimals}f}", (b, y), (0, -13), textcoords="offset points",
                    ha="center", color=TILE, fontsize=8)
    ax.set_yticks(range(len(labels)), list(reversed(labels)))
    ax.set_ylim(-.55, len(labels)-.45)
    ax.set_xlim(bounds)
    ax.set_xlabel(label)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    polish(ax)


def matched_fidelity():
    runs = read("matched-ball-rate-20261122/render-result.json")
    scores = read("matched-ball-rate-20261122/scores.json")["means"]["matched"]
    plain = read("clean-court-regional-screen-20261118/control-scores.json")["means"]["plain-48"]
    assert runs["order"] == ["plain48", "matched50_plus_tiles"]
    assert runs["baseline_total_bytes"] == 182055 and runs["candidate_total_bytes"] == 182226
    psnr = runs["rgb_psnr_12_reviewed"]

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.45), gridspec_kw={"width_ratios": [1.45, 1]})
    draw_pairs(ax[0], [(plain[k], scores[k]) for k in ("ball", "p1", "p2")],
               ["Ball crop", "Player 1", "Player 2"], 4, "DISTS   ↓", (.145, .26))
    draw_pairs(ax[1], [(psnr[0], psnr[1])], ["Whole frame"], 3, "RGB PSNR (dB)   ↑", (31.8, 32.45))
    ax[0].set_title("Object fidelity", loc="left", fontweight="bold", pad=9)
    ax[1].set_title("Full-frame fidelity", loc="left", fontweight="bold", pad=9)
    fig.text(.13, .97, "Ball tiles at nearly equal total rate", fontsize=13, fontweight="bold", va="top")
    fig.text(.13, .885, "One tennis rally · 12 source-reviewed frames · DISTS on body masks and a 64×64 ball crop",
             fontsize=8.5, va="top")
    fig.text(.13, .08, "●", color=PLAIN, fontsize=11)
    fig.text(.15, .08, f"Plain AV1  {runs['baseline_kbps']:.3f} kbps", fontsize=8.5)
    fig.text(.49, .08, "●", color=TILE, fontsize=11)
    fig.text(.51, .08, f"AV1 + tracked tiles  {runs['candidate_kbps']:.3f} kbps", fontsize=8.5)
    fig.subplots_adjust(left=.13, right=.98, top=.70, bottom=.26, wspace=.65)
    save(fig, "matched-rate-ball-fidelity")


def detection():
    rec = read("matched-ball-rate-20261122/historical-detection-replay.json")["result"]
    visible = rec["visible"]
    absent = rec["absent"]
    hits = rec["thresholds"]["10"]
    fa = rec["absent_pairs"]
    baseline = fa["both_false"] + fa["baseline_only_false"]
    candidate = fa["both_false"] + fa["candidate_only_false"]
    assert (visible, absent, hits["baseline_hits"], hits["candidate_hits"], baseline, candidate) == (261, 30, 5, 229, 3, 5)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8), gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, nums, denom, title in (
        (axes[0], (hits["baseline_hits"], hits["candidate_hits"]), visible, "Visible ball located"),
        (axes[1], (baseline, candidate), absent, "False detection when absent"),
    ):
        vals = [n * 100 / denom for n in nums]
        ax.barh([1, 0], vals, color=[PLAIN, TILE], height=.55)
        for y, n, val in zip((1, 0), nums, vals, strict=True):
            ax.text(val + 2, y, f"{n}/{denom}  ({val:.1f}%)", va="center", fontsize=8.5)
        ax.set_yticks([1, 0], ["Plain AV1", "AV1 + rendered streak"] if ax is axes[0] else ["", ""])
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
        ax.set_xlabel("Frames (%)")
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        polish(ax)
    fig.text(.23, .98, "Ball detection on a low-rate clip", fontsize=13, fontweight="bold", va="top")
    fig.text(.23, .89, "Independent WASB detector · 10-pixel localization tolerance · one clip",
             fontsize=8.5, va="top")
    rates = read("k5d-inputs/saved292-rescue-absence-analysis-root-acceptance-20261113.json")["accepted_findings"]["CRF62_baseline_streak_wire_kbps"]
    fig.text(.23, .075, f"Plain AV1 {rates[0]:.3f} kbps    ·    AV1 + streak {rates[1]:.3f} kbps", fontsize=8.5)
    fig.subplots_adjust(left=.23, right=.98, top=.70, bottom=.28, wspace=.42)
    save(fig, "ball-detection")


def three_matches():
    result = read("submission-ladder-ball-confirmation-analysis-20261115.json")["equal_source_summary"]["crop_dists"]
    assert result["source_count"] == 3 and result["status"] == "supported"
    sources = result["source_ids"]
    rates = [v * 100 for v in result["source_bd_rate"]]
    mean = result["mean_bd_rate"] * 100
    fig, ax = plt.subplots(figsize=(6.4, 2.85))
    ax.axvline(0, color="#606b74", linewidth=.8)
    ax.barh([3, 2, 1], rates, color="#88b9b2", height=.54)
    ax.scatter([mean], [0], marker="D", color=TILE, s=37, zorder=4)
    for y, rate in zip([3, 2, 1], rates, strict=True):
        ax.text(rate - 1.3, y, f"{rate:.1f}%", ha="right", va="center", fontsize=8.5)
    ax.text(mean - 1.3, 0, f"{mean:.1f}%", ha="right", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks([3, 2, 1, 0], sources + ["Equal-match average"])
    ax.set_xlim(-50, 3)
    ax.set_xlabel("Change in rate at equal ball-crop DISTS (%)   ↓")
    ax.set_title("Tracked ball pixels across three matches", loc="left", fontsize=12, fontweight="bold", pad=12)
    polish(ax)
    fig.text(.20, .035, "Three sources · 32 visible instants total · per-match curves and achievable anchor envelope",
             fontsize=7.5)
    fig.subplots_adjust(left=.25, right=.98, top=.82, bottom=.26)
    save(fig, "three-match-ball-bd-rate")


if __name__ == "__main__":
    setup()
    matched_fidelity()
    detection()
    three_matches()
