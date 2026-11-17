"""Recompute archived point comparisons and draw descriptive paper evidence.

No reconstruction, metric inference, encoding, or network access. Outputs must
be written outside the checkout. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.tennis.ball_confirmation_analysis import METRICS, REFERENCE, source_summary
from src.pointstream.evaluation import anchor_sensitivity

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "docs/experiments/submission-ladder-ball-confirmation-analysis-20261115.json"
AUDIT = ROOT / "docs/experiments/player-reference-pilot-20261117/packet-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ANALYSIS_SHA256 = "e60ca013c6518a71f9059d029a3cf5bb90711edf6d262ad1ebb430887680a6d5"
AUDIT_SHA256 = "202672cc26fe6f0256b97a8af7920080d891949320c7d74ad3a0158beb5ce636"


def authenticated_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"input SHA256 differs: {path}")
    return json.loads(raw)


def fresh_output(out: Path) -> None:
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("output must be outside the checkout")
    out.mkdir(parents=True, exist_ok=False)


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = authenticated_json(ANALYSIS, ANALYSIS_SHA256)
    audit = authenticated_json(AUDIT, AUDIT_SHA256)
    fresh_output(out)
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
        comparisons: dict[str, Any] = {}
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
    costs: list[dict[str, Any]] = [
        {"label": "PNG", "frame": 23547, "reference": 366161, "window": 2273}
    ]
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
        for j, value in enumerate(s["source_bd_rate"]):
            if value is not None:
                ax.scatter(100 * value, y + (j - 1) * 0.12, s=25, color="#30769a", zorder=3)
        if s["status"] != "supported":
            ax.text(
                0.38,
                y,
                "Pooled interval unsupported",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
            continue
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
        "Dots: available matches  ·  Diamond: equal-match log mean  ·  Line: 27-resample percentile interval\nSaved score points; intervals describe only these matches.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "source-effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "source-effects.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    left = [0.0] * 3
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


def player_figure(out: Path, report: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = "d09b054369e2c435d2347e0455a0b244af52f2af03d599becdb48110cbb1156b"
    data = authenticated_json(report, expected)
    windows = data["primary_ivf"]["scenarios"]["7200:+0.000:primary"]["windows"]
    names = ("al028", "al034", "al043", "med000", "med024", "med019")
    if set(windows) != set(names) or not windows["med000"]["dcvc_table_absent"]:
        raise ValueError("unexpected frozen windows or anchor coverage")
    rows = {
        name: {metric: windows[name]["metrics"][metric]["delta"] for metric in ("lpips", "dists")}
        for name in names
    }
    means = {
        group: {
            metric: math.fsum(rows[n][metric] for n in subset) / len(subset)
            for metric in ("lpips", "dists")
        }
        for group, subset in (("al", names[:3]), ("med", names[3:]))
    }
    fresh_output(out)
    result = {
        "report_sha256": expected,
        "generator_sha256": sha(Path(__file__)),
        "scenario": "7200:+0.000:primary",
        "window_delta_anchor_minus_hybrid": rows,
        "within_match_means": means,
        "scope": "Corrected modeled two-hour reuse; historical saved player scores; best available anchors. med000 DCVC absent; no complete codec or window-based inferential CI.",
    }
    (out / "player-effects.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, layout="constrained")
    for ax, metric in zip(axes, ("lpips", "dists")):
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(2.5, color="0.8", lw=0.7)
        for y, name in enumerate(names):
            ax.scatter(rows[name][metric], y, s=36, color="#30769a", zorder=3)
        for y, group in ((1, "al"), (4, "med")):
            ax.scatter(means[group][metric], y, marker="D", s=30, color="#db9b39", zorder=4)
        ax.set_title(metric.upper(), weight="bold")
        ax.set_xlabel("Player gap: anchor − hybrid")
        ax.set_yticks(range(6), names)
        ax.invert_yaxis()
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Window (two physical matches)")
    fig.suptitle("Historical player gaps: corrected modeled two-hour reuse", weight="bold")
    fig.text(
        0.05,
        -0.12,
        "Dots: windows  ·  Diamonds: within-match means  ·  Best available anchors\nmed000: DCVC missing; SVT anchor. No complete codec; no inferential intervals.",
        fontsize=8,
    )
    fig.savefig(out / "player-effects.png", dpi=220, bbox_inches="tight")
    fig.savefig(
        out / "player-effects.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--player-report", type=Path)
    args = parser.parse_args()
    if args.player_report:
        player_figure(args.output, args.player_report)
    else:
        main(args.output)
