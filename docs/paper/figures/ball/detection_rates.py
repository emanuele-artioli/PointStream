"""Plot original WASB detection results using actual byte totals and frame counts."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "docs/experiments/k5d-inputs/svt-study4282-64160888-confirmation292-root-acceptance-20261113.json"
CHECK = "4922cb4f14c380aedf88f80be686f2f1c32755852b67f64f8976d8ebf798dc43"


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECK:
        raise ValueError("Source receipt changed")
    study = json.loads(raw)
    assert study["canonical_status"] == "COMPLETE/gatePASS/remoteexit0"
    scope = study["scientific_scope"]
    assert scope["full_frames"] == 292
    assert scope["label_counts"] == {"1": 261, "0": 30, "-1": 1}
    contexts = json.loads((HERE / "detection-rate-context.json").read_text())
    assert contexts["producer_job"] == study["job_id"]
    context_by_crf = {int(row["point"].split("-")[1]): row for row in contexts["point_context"]}
    points = sorted(study["quality_and_absence"]["points"],
                    key=lambda point: point["native_bytes"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for arm, key, label, color, marker in [
        ("baseline", "baseline_wire_bytes", "SVT-AV1", "#19639b", "o"),
        ("streak", "streak_wire_bytes", "SVT-AV1 + rendered streak", "#cf740d", "s"),
    ]:
        rates = [context_by_crf[p["crf"]][key] * 8 * 30 / (292 * 1000) for p in points]
        hits = [p["arms"][arm]["judges"]["wasb"]["hit10"] for p in points]
        false = [p["arms"][arm]["judges"]["wasb"]["false_on_absent"] for p in points]
        ax.plot(rates, [100 * h / 261 for h in hits], color=color, marker=marker,
                linewidth=2.2, markersize=7, label=label)
        for rate, hit, fp in zip(rates, hits, false):
            ax.annotate(f"{hit}/261; {fp}/30 false", (rate, 100 * hit / 261),
                        xytext=(0, 10 if arm == "streak" or hit < 20 else -20),
                        textcoords="offset points", ha="center", color=color, fontsize=8)
    ax.set(xlabel="Complete bitrate (kbps)",
           ylabel="Visible frames localized within 10 px (%)", ylim=(-4, 106), xlim=(35, 290))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(loc="center right", frameon=False)
    fig.suptitle("Ball detection at low bitrate", x=.08, ha="left", fontsize=15, weight="bold")
    fig.text(.08, .87, "One retrospective clip · WASB evaluates decoded RGB", fontsize=9)
    fig.text(.08, .035, "False detections use 30 absent frames. Paired rates differ by 0.4–1.6%; 1 uncertain frame is excluded.", fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,.84))
    fig.savefig(HERE / "ball-detection-rates.pdf", bbox_inches="tight")
    fig.savefig(HERE / "ball-detection-rates.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for arm, key, label, color, marker in [
        ("baseline", "baseline_wire_kbps", "SVT-AV1", "#19639b", "o"),
        ("streak", "streak_wire_kbps", "SVT-AV1 + rendered streak", "#cf740d", "s"),
    ]:
        rows = sorted(contexts["point_context"], key=lambda row: row[key])
        rates = [row[key] for row in rows]
        quality = [row[f"{arm}_mean_crop_dists"] for row in rows]
        ax.plot(rates, quality, color=color, marker=marker, linewidth=2.2,
                markersize=7, label=label)
        for rate, value in zip(rates, quality):
            ax.annotate(f"{rate:.1f} kbps · {value:.3f}", (rate, value),
                        xytext=(0, 9 if arm == "baseline" else -17),
                        textcoords="offset points", ha="center", color=color, fontsize=8)
    ax.set(xlabel="Complete bitrate (kbps)", ylabel="Ball-crop DISTS  ↓",
           ylim=(.06, .16), xlim=(35, 290))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Ball appearance at low bitrate", x=.08, ha="left", fontsize=15, weight="bold")
    fig.text(.08, .87, "One retrospective clip · 261 labeled visible frames", fontsize=9)
    fig.text(.08, .035, "Paired rates differ by 0.4–1.6%. The sender fits the streak to the source and base decode.", fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,.84))
    fig.savefig(HERE / "ball-streak-rates.pdf", bbox_inches="tight")
    fig.savefig(HERE / "ball-streak-rates.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
