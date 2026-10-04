"""Validate a complete CPU replay and produce compact paper data/figure.

Run only after the native job completes. This does not encode or decode video.
The raw per-frame report remains external; this copy retains identities and
derived equal-scene summaries, keeping region and whole-frame metrics distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def identity(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def summarize(raw: dict, report_identity: dict, historical: dict | None = None) -> dict:
    if (
        not raw.get("complete")
        or raw.get("smoke")
        or len(raw["points"]) != 96
        or len(raw["clips"]) != 8
    ):
        raise ValueError("a complete eight-scene, 96-stream non-smoke report is required")
    max_delta = max(abs(x["historical_mean_psnr_delta_db"]) for x in raw["points"])
    if max_delta > 1e-8:
        raise ValueError("fresh own-target scores do not reproduce the saved scores")
    if raw["controls"]["identical"]["whole"] != 0 or raw["controls"]["offset_8"]["whole"] != 64:
        raise ValueError("scoring controls failed")
    point_ids = [(x["clip"], x["codec"], x["target"], x["qp"]) for x in raw["points"]]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("duplicate registered operating point")
    for clip in raw["clips"]:
        if (
            not clip["rgb_window"]["all_luma_pixels_identical"]
            or clip["rgb_window"]["frames"] != 48
        ):
            raise ValueError("RGB input identity failed")
        sources = clip["sources"]
        for target in ["original", "plate"]:
            if sources["av1"][target]["luma_sha256"] != sources["vvc"][target]["luma_sha256"]:
                raise ValueError("AV1/VVC source targets differ")
        if any(x["outside_mask_changed_pixels"] for x in sources.values()):
            raise ValueError("removal changed unmasked pixels")
    compact = []
    regional_deltas = []
    for p in raw["points"]:
        scores: dict[str, dict[str, dict[str, Any]]] = {}
        for target, parts in p["scores"].items():
            if parts["frames"] != 48 or len(parts["per_frame"]) != 48:
                raise ValueError("incorrect all-frame denominator")
            scores[target] = {}
            for frame in parts["per_frame"]:
                fraction = frame["foreground_pixels"] / frame["frame_pixels"]
                combined = fraction * frame["foreground"] + (1 - fraction) * frame["background"]
                if not np.isclose(frame["whole"], combined, rtol=1e-10, atol=1e-10):
                    raise ValueError("regional pixel MSE does not reconstruct whole-frame MSE")
            for region in ["whole", "foreground", "background"]:
                if parts[region]["identical_frames"] or parts[region]["mean_frame_psnr_db"] is None:
                    raise ValueError(
                        "identical frame requires separate interpretation, never silently omit"
                    )
                dbs = [10 * np.log10(255**2 / frame[region]) for frame in parts["per_frame"]]
                if not np.isclose(
                    np.mean(dbs), parts[region]["mean_frame_psnr_db"], rtol=1e-10, atol=1e-10
                ):
                    raise ValueError("reported mean-frame PSNR arithmetic mismatch")
                scores[target][region] = {key: value for key, value in parts[region].items()}
        compact.append(
            {
                key: p[key]
                for key in [
                    "clip",
                    "codec",
                    "target",
                    "qp",
                    "stream",
                    "fresh_decode_luma_sha256",
                    "historical_mean_psnr_delta_db",
                ]
            }
            | {"scores": scores}
        )
        if p["target"] == "original":
            if historical is None:
                raise ValueError(
                    "historical foreground/background scores required for mask verification"
                )
            saved = historical["fg"][p["codec"]][p["clip"]]
            index = saved["original_curve"]["rates"].index(p["stream"]["bytes"])
            for region, key in [
                ("foreground", "original_fg_psnr"),
                ("background", "original_bg_psnr"),
            ]:
                delta = p["scores"]["original"][region]["mean_frame_psnr_db"] - saved[key][index]
                if abs(delta) > 1e-8:
                    raise ValueError(
                        "fresh original-region scores do not reproduce saved mask metrics"
                    )
                regional_deltas.append(abs(delta))
    midpoint: dict[str, Any] = {}
    for codec in ["av1", "vvc"]:
        midpoint[codec] = {}
        for target in ["original", "plate"]:
            selected = [
                p
                for p in compact
                if p["codec"] == codec and p["target"] == target and p["qp"] == 40
            ]
            if len(selected) != 8:
                raise ValueError("one registered QP40 point per scene required")
            midpoint[codec][target] = {}
            for reference in ["original", "plate"]:
                midpoint[codec][target][reference] = {}
                for region in ["whole", "foreground", "background"]:
                    values = [
                        p["scores"][reference][region]["mean_frame_psnr_db"] for p in selected
                    ]
                    midpoint[codec][target][reference][region] = {
                        "equal_scene_mean_db": float(np.mean(values)),
                        "scene_range_db": [float(min(values)), float(max(values))],
                        "per_scene_db": {p["clip"]: v for p, v in zip(selected, values)},
                    }
    support = {}
    for codec in ["av1", "vvc"]:
        rows = [c["comparisons"][codec]["original_target_unrestored_plate"] for c in raw["clips"]]
        support[codec] = {
            "no_common_support": sum(not x["has_overlap"] for x in rows),
            "nonmonotone_with_overlap": sum("ineligible_reason" in x for x in rows),
            "eligible_overlap": sum(x["bd_rate_percent"] is not None for x in rows),
            "per_scene": {c["name"]: c["comparisons"][codec] for c in raw["clips"]},
        }
    return {
        "schema": "pointstream.cpu_headroom_paper_summary.v1",
        "native_report": report_identity,
        "code_revision": raw["code_revision"],
        "worker": raw["worker"],
        "environment": raw["environment"],
        "policy": raw["policy"],
        "started_utc": raw["started_utc"],
        "finished_utc": raw["finished_utc"],
        "method": "QP40 summaries are equal-scene means of equal-frame mean Y-PSNR. Regional masks are fixed BP21 masks; neither regional dB nor their means are whole-frame PSNR. Source-cluster sensitivity uses fixed historical per-scene savings.",
        "integrity": {
            "streams": len(compact),
            "unique_windows": 8,
            "frames_per_stream": 48,
            "all_own_target_scores_reproduced": True,
            "max_own_target_mean_psnr_delta_db": max_delta,
            "max_original_region_psnr_delta_db": max(regional_deltas),
            "all_current_rgb_luma_pixels_verified": True,
            "all_unmasked_target_pixels_unchanged": True,
        },
        "controls": raw["controls"],
        "clips": raw["clips"],
        "points": compact,
        "qp40": midpoint,
        "support": support,
        "group_sensitivity": raw["group_sensitivity"],
    }


def render_figure(data: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    for ax, codec in zip(axes, ["av1", "vvc"]):
        rows = [
            p
            for p in data["points"]
            if p["codec"] == codec and p["target"] == "plate" and p["qp"] == 40
        ]
        for i, row in enumerate(rows):
            values = [
                row["scores"]["plate"]["whole"]["mean_frame_psnr_db"],
                row["scores"]["original"]["whole"]["mean_frame_psnr_db"],
                row["scores"]["original"]["foreground"]["mean_frame_psnr_db"],
            ]
            ax.plot(range(3), values, color="#718096", alpha=0.55, linewidth=0.8)
            ax.scatter(range(3), values, c=["#2455a6", "#b45309", "#9d174d"], s=13, zorder=3)
        means = [
            np.mean([row["scores"][target][region]["mean_frame_psnr_db"] for row in rows])
            for target, region in [
                ("plate", "whole"),
                ("original", "whole"),
                ("original", "foreground"),
            ]
        ]
        ax.plot(
            range(3),
            means,
            color="black",
            linewidth=2,
            marker="D",
            markersize=4,
            label="Equal-scene mean",
        )
        ax.set_title(codec.upper() + ": same decoded plate stream, QP40")
        ax.set_xticks(
            range(3),
            [
                "Plate target\nwhole frame",
                "Original target\nwhole frame",
                "Original target\nplayer mask",
            ],
        )
        ax.set_ylabel("Mean-frame Y-PSNR (dB)")
        ax.set_ylim(5, 46)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--figure", type=Path)
    p.add_argument("--historical", type=Path, required=True)
    args = p.parse_args()
    raw = json.loads(args.report.read_text())
    if identity(args.historical)["sha256"] != raw["historical_report"]["sha256"]:
        raise ValueError("historical reference report identity mismatch")
    data = summarize(raw, identity(args.report), json.loads(args.historical.read_text()))
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if args.figure:
        render_figure(data, args.figure)
    print(
        json.dumps(
            {
                "integrity": data["integrity"],
                "support": {
                    c: {k: v for k, v in s.items() if k != "per_scene"}
                    for c, s in data["support"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
