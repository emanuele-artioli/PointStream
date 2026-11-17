"""Recompute archived point comparisons and draw descriptive paper evidence.

No reconstruction, metric inference, encoding, or network access. Outputs must
be written outside the checkout. Run from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.tennis.ball_confirmation_analysis import METRICS, REFERENCE, source_summary
from src.pointstream.evaluation import anchor_sensitivity

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "docs/experiments/submission-ladder-ball-confirmation-analysis-20261115.json"
AUDIT = ROOT / "docs/experiments/player-reference-pilot-20261117/packet-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual: object, expected: object, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key, value in expected.items():
            check(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected), path
        for i, value in enumerate(expected):
            check(actual[i], value, f"{path}/{i}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-11, abs_tol=1e-12
        ), path
    else:
        assert actual == expected, path


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data, audit = json.loads(ANALYSIS.read_text()), json.loads(AUDIT.read_text())
    groups = [
        ("reference", REFERENCE),
        ("native_svt", {"native_svt"}),
        ("native_dcvc", {"native_dcvc"}),
        ("roi_raw", {"roi_raw"}),
        ("roi_fitted", {"roi_fitted"}),
        ("streak", {"streak"}),
    ]
    rebuilt = []
    for source in data["sources"]:
        rows = source["physical_curves"]
        comparisons = {}
        for name, families in groups:
            comparisons[name] = {}
            for metric in METRICS:
                sign = 1 if metric == "rgb_psnr" else -1

                def points(family):
                    return [
                        (r["bytes"], sign * r["means"][metric])
                        for r in rows
                        if r["family"] in family
                    ]

                comparisons[name][metric] = anchor_sensitivity(points({"ptp"}), points(families))
        check(comparisons, source["comparisons"], source["source_id"])
        rebuilt.append({"source_id": source["source_id"], "comparisons": comparisons})
    summary = {metric: source_summary(rebuilt, metric) for metric in METRICS}
    check(summary, data["equal_source_summary"], "summary")
    assert all(
        len(s["bootstrap_resamples"]) == 27 for s in summary.values() if s["status"] == "supported"
    )
    assert audit["status"] == "PASS"
    variants = {v["quality"]: v for v in audit["variants"]}
    costs = [{"label": "PNG", "frame": 23547, "reference": 366161, "window": 2273}]
    for quality in (70, 40):
        roles = variants[quality]["role_bytes"]
        assert (
            roles["frame"] + roles["reference"] + roles["window"] + roles["match"]
            == variants[quality]["packet_bytes"]
        )
        costs.append(
            {"label": f"JPEG {quality}", **{k: roles[k] for k in ("frame", "reference", "window")}}
        )
    assert (
        costs[0]["frame"] + costs[0]["reference"] + costs[0]["window"]
        == audit["png_packet_unchanged_bytes"]
    )
    assert [(v["frame"], v["reference"], v["window"]) for v in costs] == [
        (23547, 366161, 2273),
        (23547, 65120, 2616),
        (23547, 41075, 2616),
    ]
    for item in costs:
        item["total_bytes"] = item["frame"] + item["reference"] + item["window"]
        item["kbps_4_8_seconds"] = item["total_bytes"] * 8 / 4800
    result = {
        "scope": "Recomputed saved scalar points; reconstructions not rescored.",
        "inputs_sha256": {
            str(ANALYSIS.relative_to(ROOT)): sha(ANALYSIS),
            str(AUDIT.relative_to(ROOT)): sha(AUDIT),
        },
        "generator_sha256": sha(Path(__file__)),
        "analysis_code_sha256": sha(ROOT / "experiments/tennis/ball_confirmation_analysis.py"),
        "anchor_code_sha256": sha(ROOT / "src/pointstream/evaluation.py"),
        "comparison_verification": "all six anchors × six metrics × three sources, including supports and geometry, agree at 1e-11 relative / 1e-12 absolute",
        "summary_verification": "all six metric summaries and ordered resamples agree at same tolerance",
        "equal_source_summary": summary,
        "role_costs": costs,
        "limitations": [
            "Three physical matches; percentile intervals are descriptive, not population significance.",
            "Full standard LPIPS and historical spatial-map mean are separate metrics.",
            "Short-window packet costs are not matched-quality evidence; no reference reuse assumed.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    metrics = ["crop_dists", "crop_lpips", "full_dists", "full_lpips_standard", "rgb_psnr"]
    labels = [
        "Ball crop DISTS",
        "Ball crop LPIPS",
        "Full DISTS",
        "Full LPIPS (standard)",
        "Full RGB PSNR",
    ]
    for y, metric in enumerate(metrics):
        s = summary[metric]
        if s["status"] != "supported":
            ax.text(0.02, y, "unsupported overlap", transform=ax.get_yaxis_transform(), va="center")
            continue
        for j, value in enumerate(s["source_bd_rate"]):
            ax.scatter(100 * value, y + (j - 1) * 0.12, s=25, color="#30769a", zorder=3)
        lo, hi = s["bootstrap_95_interval"]
        mean = s["mean_bd_rate"]
        ax.plot([100 * lo, 100 * hi], [y, y], color="#172e40", lw=2)
        ax.scatter(100 * mean, y, marker="D", s=38, color="#172e40", zorder=4)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("BD-rate vs. reference envelope (%)  •  lower is better")
    ax.set_title("Per-match effects and descriptive uncertainty", loc="left", weight="bold")
    ax.text(
        0,
        -0.2,
        "Dots: 3 matches  ·  Diamond: equal-match log mean  ·  Line: 27-resample percentile interval\nSaved score points; intervals describe only these matches.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "source-effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "source-effects.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    left = [0] * 3
    for key, label, color in [
        ("frame", "Ongoing frame data", "#397b9e"),
        ("reference", "Reference images", "#db9b39"),
        ("window", "Window framing", "#687a86"),
    ]:
        values = [v[key] / 1000 for v in costs]
        ax.barh([v["label"] for v in costs], values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    for y, item in enumerate(costs):
        ax.text(left[y] + 3, y, f"{item['total_bytes']:,} B", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.25)
    ax.set_xlabel("Actual packet bytes (thousands), 4.8-second window")
    ax.set_title("Frozen packet roles", loc="left", weight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "packet-roles.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "packet-roles.pdf", bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {"output": str(out), "verified": True, "summary_sha256": sha(out / "summary.json")}
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
