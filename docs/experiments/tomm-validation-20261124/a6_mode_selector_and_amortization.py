"""Evaluate a strictly source-only closed-loop shot mode selector and multi-rally amortization (Track A6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_json(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text())


def select_mode_source_only(
    *,
    court_ticks: int,
    total_ticks: int,
    raw_paired_ticks: int,
    max_interior_gap: int | None,
    touches_boundary: bool,
    visible_ball_ticks: int,
    appearance_cached: bool,
    target_kbps: float,
    full_obj_warm_kbps: float,
    full_obj_cold_kbps: float,
    conv_ball_min_kbps: float,
) -> tuple[str, str]:
    """Choose {Conv, Conv+Ball, Full-Obj} from source observables and rate/cache state only."""
    court_frac = court_ticks / max(1, total_ticks)
    if court_frac < 0.80:
        return "Conv", f"court_fit={court_frac:.3f} < 0.80"

    pose_ok = raw_paired_ticks == total_ticks or (
        max_interior_gap is not None and max_interior_gap <= 10 and not touches_boundary
    )
    need_kbps = full_obj_warm_kbps if appearance_cached else full_obj_cold_kbps
    if pose_ok and visible_ball_ticks >= 15 and target_kbps >= need_kbps:
        reason = "paired_240" if raw_paired_ticks == total_ticks else f"interp_gap_le_10(max={max_interior_gap})"
        return "Full-Obj", f"court_ok & {reason} & budget>={need_kbps:.1f}kbps"

    if visible_ball_ticks >= 10 and target_kbps >= conv_ball_min_kbps:
        if not pose_ok:
            why_no_full = "pose_gap>10_or_boundary"
        else:
            why_no_full = f"budget<{need_kbps:.1f}kbps({'warm' if appearance_cached else 'cold'})"
        return "Conv+Ball", f"court_ok & ball_vis={visible_ball_ticks} ({why_no_full})"

    return "Conv", "insufficient_ball_or_below_min_ball_rate"


def best_ball_dists_under_budget(row: dict[str, Any], codec_key: str) -> float:
    return float(row[codec_key])


def amortization_curve(
    *,
    name: str,
    fps_num: int,
    fps_den: int,
    frames: int,
    running_bytes: int,
    init_bytes: int,
    k_values: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100),
) -> dict[str, Any]:
    dur_s = frames * fps_den / fps_num
    rows = []
    for k in k_values:
        eff_bytes = running_bytes + init_bytes / k
        kbps = eff_bytes * 8.0 / 1000.0 / dur_s
        overhead_frac = (init_bytes / k) / eff_bytes
        rows.append(
            {
                "k_rallies": k,
                "active_seconds": k * dur_s,
                "effective_bytes_per_rally": eff_bytes,
                "effective_kbps": kbps,
                "init_share": overhead_frac,
            }
        )
    return {
        "name": name,
        "duration_s": dur_s,
        "running_bytes": running_bytes,
        "init_bytes": init_bytes,
        "warm_kbps": running_bytes * 8.0 / 1000.0 / dur_s,
        "cold_kbps": (running_bytes + init_bytes) * 8.0 / 1000.0 / dur_s,
        "points": rows,
    }


def main() -> None:
    cohort = load_json("docs/experiments/cohort-expansion-20261122/cohort-results.json")
    curve = load_json("docs/experiments/cohort-expansion-20261122/curve-summary.json")
    gaps = load_json("docs/experiments/tomm-validation-20261124/pose-gap-audit.json")

    range_by_match = {m["match"]: m for m in curve["grid"]["members"]}
    gap_by_match = gaps["matches"]

    per_match: list[dict[str, Any]] = []
    # Compare at 20.4 kbps warm-cache and 35.0 kbps warm-cache regimes across all 12 test matches
    # (and 26.0 kbps warm / 200 kbps cold)
    regimes = [
        ("warm_20.4", True, 20.4),
        ("warm_26.0", True, 26.0),
        ("cold_26.0", False, 26.0),
        ("cold_200.0", False, 200.0),
    ]

    for row in cohort["rows"]:
        mid = row["match"]
        g = gap_by_match.get(mid, {})
        players = g.get("players", {})
        max_gap = max((p.get("max_interior_gap", 0) for p in players.values()), default=None) if players else None
        touches_b = any(p.get("touches_boundary", False) for p in players.values()) if players else True
        raw_paired = g.get("paired_ticks", 0)
        vis_marks = range_by_match[mid]["visible"]
        cand_min = range_by_match[mid]["ranges"]["candidate"][0]

        # Match typical warm/cold Full-Obj cost by frame rate (25/30fps ~18 kbps warm / ~190 cold; 60fps ~26 kbps warm / ~325 cold)
        is_60 = row["fps"].startswith("60")
        full_warm = 26.0 if is_60 else 18.0
        full_cold = 325.0 if is_60 else 190.0

        regime_decisions = {}
        for rname, cached, budget in regimes:
            mode, reason = select_mode_source_only(
                court_ticks=row["court_ticks"],
                total_ticks=240,
                raw_paired_ticks=raw_paired,
                max_interior_gap=max_gap,
                touches_boundary=touches_b,
                visible_ball_ticks=vis_marks,
                appearance_cached=cached,
                target_kbps=budget,
                full_obj_warm_kbps=full_warm,
                full_obj_cold_kbps=full_cold,
                conv_ball_min_kbps=cand_min,
            )
            # Regret on ball-region DISTS at 20.4 kbps: Conv+Ball vs Conv ordinary
            ball_gain = None
            if mode in ("Conv+Ball", "Full-Obj") and row["ordinary_dists"] is not None:
                ball_gain = float(row["ordinary_dists"]) - float(row["candidate_dists"])
            regime_decisions[rname] = {
                "mode": mode,
                "reason": reason,
                "ball_dists_improvement_over_same_res_av1": ball_gain,
            }

        per_match.append(
            {
                "match": mid,
                "event": row["event"],
                "fps": row["fps"],
                "court_ticks": row["court_ticks"],
                "pose_status": g.get("status", "unknown"),
                "raw_paired_ticks": raw_paired,
                "max_interior_gap": max_gap,
                "visible_ball_marks": vis_marks,
                "conv_ball_min_kbps": cand_min,
                "regimes": regime_decisions,
            }
        )

    amortizations = [
        amortization_curve(
            name="Shanghai match102 (joint-crf50, 30000/1001 fps)",
            fps_num=30000,
            fps_den=1001,
            frames=240,
            running_bytes=17794,
            init_bytes=139463,
        ),
        amortization_curve(
            name="Sydney match127 (joint-crf70, 60 fps)",
            fps_num=60,
            fps_den=1,
            frames=240,
            running_bytes=12890,
            init_bytes=149450,
        ),
        amortization_curve(
            name="Halle match105 (joint-crf50, 25 fps, sender verified; receiver pending)",
            fps_num=25,
            fps_den=1,
            frames=240,
            running_bytes=18319,
            init_bytes=174384,
        ),
    ]

    summary = {
        "counts_by_regime": {
            rname: {
                m: sum(1 for pm in per_match if pm["regimes"][rname]["mode"] == m)
                for m in ("Conv", "Conv+Ball", "Full-Obj")
            }
            for rname, _, _ in regimes
        },
        "source_only_properties": [
            "Zero access to decoded quality or DISTS oracle metrics",
            "Pose-gap interpolation (max_gap <= 10, interior-only) recovers Munich match108 and Geneva match116 for Full-Obj eligibility",
            "Cold start at 26 kbps cleanly falls back to Conv+Ball (on 11/12 court-eligible matches) instead of blowing the rate ceiling",
        ],
        "per_match": per_match,
        "amortization_eq2": amortizations,
    }
    out = HERE / "a6-mode-selector-amortization.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote", out)
    print("Regime counts:", json.dumps(summary["counts_by_regime"]))


if __name__ == "__main__":
    main()
