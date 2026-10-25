"""CPU-only fixed-output side-information accounting; no new quality measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.background.g5 import envelope

DIAGNOSTIC_SHA = "e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f"
METRICS = ("lpips", "dists")
BASE_BYTES = 48_878_567


def curve(row: dict[str, Any], metric: str) -> list[tuple[float, float]]:
    points = [(float(a["kbps"]), -float(a[metric])) for a in row["anchor_points"].values()]
    if not points or any(not math.isfinite(r + q) or r < 0 for r, q in points):
        raise ValueError("Invalid measured anchors")
    return envelope(points)


def comparison(row: dict[str, Any], metric: str, rate: float) -> dict[str, Any]:
    """The original strict bracket-only comparison, including endpoint equality."""
    from experiments.tennis.k6b_confirm_validate import strict_won

    result = strict_won(row["anchor_points"], metric, rate, row["method"][metric])
    # An actual cheaper/no-worse point is a valid fixed-output dominance witness,
    # even when equal-rate interpolation is unbracketed. Keep won=None there.
    witnesses = [
        {"name": n, "kbps": a["kbps"], "error": a[metric]}
        for n, a in row["anchor_points"].items()
        if a["kbps"] <= rate and a[metric] <= row["method"][metric]
    ]
    result["observed_dominator"] = min(witnesses, key=lambda a: a["kbps"]) if witnesses else None
    return result


def headroom(row: dict[str, Any], metric: str) -> dict[str, Any]:
    hull = curve(row, metric)
    q = -float(row["method"][metric])
    rate = float(row["billed_kbps"])
    out: dict[str, Any] = {
        "baseline_kbps": rate,
        "method_error": -q,
        "measured_rate_bracket": [hull[0][0], hull[-1][0]],
        "baseline": comparison(row, metric, rate),
    }
    if q < hull[0][1]:
        return {
            **out,
            "kind": "left_censored_no_measured_win",
            "threshold_kbps": None,
            "headroom_upper_bound_kbps": hull[0][0] - rate,
        }
    if q > hull[-1][1]:
        return {
            **out,
            "kind": "right_censored_wins_through_bracket",
            "threshold_kbps": None,
            "headroom_lower_bound_kbps": hull[-1][0] - rate,
        }
    threshold = hull[0][0]
    for (r0, q0), (r1, q1) in zip(hull, hull[1:]):
        if q0 <= q <= q1:
            threshold = r0 + (r1 - r0) * (q - q0) / (q1 - q0)
            break
    return {
        **out,
        "kind": "measured_crossing",
        "threshold_kbps": threshold,
        "headroom_kbps": threshold - rate,
        "threshold_itself_wins": False,
    }


def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [r[m] for r in rows for m in METRICS]
    wins = sum(c["won"] is True for c in cells)
    resolved = sum(c["won"] is not None for c in cells)
    return {
        "wins": wins,
        "resolved": resolved,
        "denominator": 24,
        "bar_reached": wins >= 18,
        "maximum_possible_wins": wins + 24 - resolved,
        "fixed_output_achievability_upper_bound_wins": 24
        - sum(c["won"] is False or c.get("observed_dominator") is not None for c in cells),
        "by_metric": {m: sum(r[m]["won"] is True for r in rows) for m in METRICS},
        "by_match": {
            match: sum(r[m]["won"] is True for r in rows if r["match"] == match for m in METRICS)
            for match in ("alcaraz_hurkacz", "medvedev_struff")
        },
    }


def oracle(diagnostic: dict[str, Any]) -> dict[str, Any]:
    if len(diagnostic["rows"]) != 12 or not diagnostic["complete"]:
        raise ValueError("Require completed frozen twelve-window diagnostic")
    rows = [
        {"window": r["window"], "match": r["match"], **{m: headroom(r, m) for m in METRICS}}
        for r in diagnostic["rows"]
    ]
    original = [
        {
            "window": r["window"],
            "match": r["match"],
            **{m: comparison(r, m, r["billed_kbps"]) for m in METRICS},
        }
        for r in diagnostic["rows"]
    ]
    if any(
        a[m]["won"] != b["billed"][m]["won"]
        for a, b in zip(original, diagnostic["rows"])
        for m in METRICS
    ):
        raise ValueError("Original frozen decisions do not reproduce")
    return {
        "question": "fixed saved output: tolerated additional rate within measured anchor brackets",
        "rows": rows,
        "free_side_information_bound": tally(original),
        "limitations": diagnostic["limitations"],
        "provenance": diagnostic["provenance"],
    }


def accounting(diagnostic: dict[str, Any], transport: dict[str, Any]) -> dict[str, Any]:
    """Count real packet bytes, replacing the entropy bound exactly once."""
    costs = {r["window"]: r for r in transport["rows"]}
    if len(transport["rows"]) != 12 or set(costs) != {r["window"] for r in diagnostic["rows"]}:
        raise ValueError("Transport cohort differs")
    for c in costs.values():
        if not c["keypoints"]["equivalent"]:
            raise ValueError("Changed keypoints cannot inherit fixed-output quality")
        sizes = [
            c["keypoints"]["bytes"],
            *[c["references"][x + "_bytes"] for x in ("png", "jpeg70", "jpeg85", "jpeg95")],
            *[c["mattes"][x]["bytes"] for x in ("exact", "half_grid", "quarter_grid")],
        ]
        if any(not isinstance(n, int) or n <= 0 for n in sizes):
            raise ValueError("Transport sizes must be positive actual byte counts")
    bank = {
        match: sum(c["references"]["png_bytes"] for c in costs.values() if c["match"] == match)
        for match in ("alcaraz_hurkacz", "medvedev_struff")
    }
    scenarios = []
    # Every bank pays every distinct reference; no reference reuse is inferred from old scores.
    for duration in ("window_refresh", 60, 600, 7200):
        for matte in ("free_optimistic", "exact", "half_grid", "quarter_grid"):
            for appearance in ("png", "jpeg70", "jpeg85", "jpeg95"):
                rows = []
                for r in diagnostic["rows"]:
                    c = costs[r["window"]]
                    exact_inputs = (
                        appearance == "png" and matte == "exact" and c["keypoints"]["equivalent"]
                    )
                    image_key = appearance + "_bytes"
                    if duration == "window_refresh":
                        reference_rate = (
                            c["references"][image_key] * 8 / r["registered_seconds"] / 1000
                        )
                    else:
                        nbytes = sum(
                            v["references"][image_key]
                            for v in costs.values()
                            if v["match"] == r["match"]
                        )
                        reference_rate = nbytes * 8 / int(duration) / 1000
                    mask_bytes = 0 if matte == "free_optimistic" else c["mattes"][matte]["bytes"]
                    keypoint_rate = c["keypoints"]["bytes"] * 8 / r["registered_seconds"] / 1000
                    matte_rate = mask_bytes * 8 / r["registered_seconds"] / 1000
                    # Adapter+guider already charged over7200s in the original point.
                    adapter_rate = r["billed_kbps"] - r["keypoint_kbps"]
                    rate = keypoint_rate + adapter_rate + reference_rate + matte_rate
                    wire_reference_rate = (
                        c["references"][image_key] * 8 / r["whole_window_seconds"] / 1000
                        if duration == "window_refresh"
                        else reference_rate
                    )
                    rows.append(
                        {
                            "window": r["window"],
                            "match": r["match"],
                            "total_kbps": rate,
                            "components_kbps": {
                                "keypoints_and_metadata": keypoint_rate,
                                "adapter_guider_7200s": adapter_rate,
                                "reference": reference_rate,
                                "matte": matte_rate,
                            },
                            "whole_window_dynamic_stream_kbps": (
                                c["keypoints"]["bytes"] + mask_bytes
                            )
                            * 8
                            / r["whole_window_seconds"]
                            / 1000,
                            "whole_window_total_kbps": (c["keypoints"]["bytes"] + mask_bytes)
                            * 8
                            / r["whole_window_seconds"]
                            / 1000
                            + adapter_rate
                            + wire_reference_rate,
                            "bit_exact_conditioning": exact_inputs,
                            **{m: comparison(r, m, rate) for m in METRICS},
                            "residual_headroom_kbps": {
                                m: (h["headroom_kbps"] - (rate - r["billed_kbps"]))
                                if "headroom_kbps" in (h := headroom(r, m))
                                else None
                                for m in METRICS
                            },
                        }
                    )
                scenarios.append(
                    {
                        "reference": appearance,
                        "asset_amortization_seconds": duration,
                        "matte": matte,
                        "quality_status": "fixed-output bit-exact accounting"
                        if appearance == "png" and matte == "exact"
                        else "optimistic unchanged-quality counterfactual; not measured quality",
                        "rows": rows,
                        **tally(rows),
                    }
                )
    base = []
    for duration in (60, 600, 7200):
        rows = [
            {
                "window": r["window"],
                "match": r["match"],
                "total_kbps": r["billed_kbps"] + BASE_BYTES * 8 / duration / 1000,
                **{
                    m: comparison(r, m, r["billed_kbps"] + BASE_BYTES * 8 / duration / 1000)
                    for m in METRICS
                },
            }
            for r in diagnostic["rows"]
        ]
        base.append(
            {
                "duration_seconds": duration,
                "additional_base_kbps": BASE_BYTES * 8 / duration / 1000,
                "basis": "original optimistic side-information point, base charge only",
                "rows": rows,
                **tally(rows),
            }
        )
    ablations = []
    for label in (
        "actual_keypoints_free_reference_matte",
        "png_bank_free_keypoints_matte",
        "exact_matte_free_keypoints_reference",
    ):
        rows = []
        for r in diagnostic["rows"]:
            c = costs[r["window"]]
            rate = r["billed_kbps"] - r["keypoint_kbps"]
            if label.startswith("actual_keypoints"):
                rate += c["keypoints"]["bytes"] * 8 / r["registered_seconds"] / 1000
            elif label.startswith("png_bank"):
                rate += bank[r["match"]] * 8 / 7200 / 1000
            else:
                rate += c["mattes"]["exact"]["bytes"] * 8 / r["registered_seconds"] / 1000
            rows.append(
                {
                    "window": r["window"],
                    "match": r["match"],
                    "total_kbps": rate,
                    **{m: comparison(r, m, rate) for m in METRICS},
                }
            )
        ablations.append(
            {
                "name": label,
                "basis": "optimistic fixed-saved-output configuration-specific upper bound; omitted costs zero",
                "rows": rows,
                **tally(rows),
            }
        )
    reuse = []
    for r in diagnostic["rows"]:
        c = costs[r["window"]]
        dynamic = (
            c["keypoints"]["bytes"] * 8 / r["registered_seconds"] / 1000
            + r["billed_kbps"]
            - r["keypoint_kbps"]
        )
        for m in METRICS:
            h = headroom(r, m)
            remaining = None if h["threshold_kbps"] is None else h["threshold_kbps"] - dynamic
            minimum = (
                bank[r["match"]] * 8 / remaining / 1000
                if remaining is not None and remaining > 0
                else None
            )
            reuse.append(
                {
                    "window": r["window"],
                    "metric": m,
                    "remaining_for_reference_kbps": remaining,
                    "reference_bank_amortization_must_exceed_seconds": minimum,
                    "matte": "free optimistic",
                    "adapter_seconds": 7200,
                }
            )
    finite = sorted(
        c["reference_bank_amortization_must_exceed_seconds"]
        for c in reuse
        if c["reference_bank_amortization_must_exceed_seconds"] is not None
    )
    return {
        "oracle": oracle(diagnostic),
        "scenarios": scenarios,
        "png_bank_bytes_by_match": bank,
        "base_sensitivities": base,
        "component_ablations": ablations,
        "free_matte_png_reference_reuse": {
            "cells": reuse,
            "finite_cells": len(finite),
            "18_cell_duration_must_exceed_seconds": finite[17] if len(finite) >= 18 else None,
            "quality_scope": "accounting assumption on the saved reference bank; future reference reuse quality unmeasured",
        },
        "transport_provenance": transport["provenance"],
        "scope": "rate evidence on saved outputs; no new quality or GPU work",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--transport", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = a.diagnostic.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIAGNOSTIC_SHA:
        raise ValueError("Diagnostic identity changed")
    d = json.loads(raw)
    out = accounting(d, json.loads(a.transport.read_text())) if a.transport else oracle(d)
    with a.out.open("x") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(json.dumps({"output": str(a.out), "rows": len(d["rows"]), "gpu_hours": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
