"""Recompute the frozen 30-clip task/fidelity benchmark from retained wire artifacts.

All population and confidence labels are retrospective. No extrapolation, no
fabricated unsupported pass, and no strong absence claim after the source false
positive calibration failure.
"""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_analysis as analysis, k5d_encoder as encoder
from experiments.tennis.k5d_fleet import EXPECTED, EXPOSURE_SHA, SOURCE, SOURCE_SHA

LOW = (("dcvc", 0), ("svt", 62))
HISTORICAL_EVIDENCE = SOURCE.parent / "source-evidence-reuse-20261104.json"
HISTORICAL_SHA = "d840b8f1daaf59150a4475967957d70ce1226e1fd647e7d142790cf685f0804f"
HISTORICAL_FAILURES = frozenset({"tracknet/game10/Clip11", "tracknet/game10/Clip8", "tracknet/game9/Clip7"})
SPANS = {"frame_psnr": .1, "crop_lpips": .005, "crop_dists": .005, "full_lpips": .005, "full_dists": .005}


def metric_value(rows: list[dict[str, Any]], metric: str) -> float:
    if metric == "frame_psnr":
        if all(type(r.get("sse")) is int and type(r.get("pixels")) is int for r in rows):
            sse, count = sum(r["sse"] for r in rows), sum(r["pixels"] for r in rows)
            if count <= 0 or sse < 0:
                raise ValueError("invalid pooled squared error")
            mse = sse / count
        else:
            # Frames have identical geometry. A 100 dB capped frame loses its
            # true MSE; substitute the cap's maximum MSE and label this bound.
            frame_psnr = [r.get("frame_psnr") for r in rows]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v > 100 for v in frame_psnr):
                raise ValueError("invalid saved frame PSNR")
            mse = sum(255**2 * 10**(-float(v) / 10) for v in frame_psnr if v is not None) / len(frame_psnr)
        return 100.0 if not mse else min(100., 10 * math.log10(255**2 / mse))
    values = [float(r[metric]) for r in rows if r[metric] is not None]
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("missing or nonfinite crop metric")
    return sum(values) / len(values)


def points(report: dict[str, Any], metric: str, candidate_family: str | None = None,
           anchor_family: str | None = None) -> list[analysis.Point]:
    fps = Fraction(report["fps"])
    duration = Fraction(report["frames"], 1) / fps
    rows = []
    for p in report["points"]:
        for arm, variant, size in (("baseline", "native", p["native_bytes"]), ("streak", "streak", p["wire_bytes"])):
            family = candidate_family if variant == "streak" else anchor_family
            if family is not None and p["codec"] != family:
                continue
            rows.append(analysis.Point(p["codec"], variant, str(p["point"]), size, duration,
                                       report["frames"], metric, metric_value(p[arm]["metrics"], metric)))
    return rows


def interpolate_native_false(native: list[dict[str, Any]], rate: float) -> float | None:
    """Piecewise log-rate interpolation of observed false counts; no extrapolation."""
    by_rate: dict[float, float] = {}
    for item in native:
        r, value = item["rate"], item["false"]
        # At ties use the smaller count, conservatively increasing excess false.
        by_rate[r] = min(by_rate.get(r, value), value)
    rows = sorted(by_rate.items())
    if not rows or not rows[0][0] <= rate <= rows[-1][0]:
        return None
    for r, v in rows:
        if r == rate:
            return v
    for (r0, v0), (r1, v1) in zip(rows, rows[1:]):
        if r0 < rate < r1:
            f = math.log(rate / r0) / math.log(r1 / r0)
            return v0 + f * (v1 - v0)
    return None


def historical_psnr_guard(fidelity: dict[str, Any], exact_sse: bool, has_cap: bool) -> bool | None:
    """Original +1% BD-rate guard: DCVC streak versus DCVC native only."""
    value = fidelity["frame_psnr"]["dcvc"]["dcvc"]["bd_rate"]
    if (not isinstance(value, (float, int)) or isinstance(value, bool)
            or not math.isfinite(value) or (has_cap and not exact_sse)):
        return None
    return bool(value <= .01)


def task_source_eligibility(clip: str, current: bool) -> tuple[bool, bool, bool]:
    """Prior registered source failures remain excluded even after a later pass."""
    if not isinstance(current, bool):
        raise ValueError("invalid current source qualification")
    historical = clip not in HISTORICAL_FAILURES
    return current, historical, current and historical


def summarize_clip(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    encoder.validate(directory)
    report = json.loads((directory / "pilot.json").read_text())
    source = json.loads((directory / "source-qualification.json").read_text())
    ids = source["identities"]
    if (ids["manifest"]["sha256"] != SOURCE_SHA or ids["exposure"]["sha256"] != EXPOSURE_SHA
            or ids["archive"]["sha256"] != EXPECTED["archive"]):
        raise ValueError("source qualification identity mismatch")
    if report["clip"] != expected["clip_id"] or report["frames"] != expected["label_rows"] or not report["full_metrics"]:
        raise ValueError("clip identity or full frame/metric scope differs")
    labels = [(r["label_kind"], r["label_x"], r["label_y"]) for r in source["rows"]]
    counts = {str(i): sum(row[0] == i for row in labels) for i in (-1, 0, 1)}
    visibility = expected["visibility_counts"]
    if counts != {"-1": visibility["3"], "0": visibility["0"], "1": visibility["1"] + visibility["2"]}:
        raise ValueError("registered label coverage mismatch")
    duration = Fraction(report["frames"], 1) / Fraction(report["fps"])
    fidelity: dict[str, Any] = {}
    for metric, span in SPANS.items():
        fidelity[metric] = {}
        for candidate in (None, "dcvc", "svt"):
            comparisons = {}
            for anchor_family in (None, "dcvc", "svt"):
                comparisons[anchor_family or "pooled"] = analysis.compare(
                    points(report, metric, candidate, anchor_family), report["frames"], duration, metric, "streak", span)
            fidelity[metric][candidate or "pooled"] = comparisons
    metric_rows = [row for point in report["points"] for arm in ("baseline", "streak") for row in point[arm]["metrics"]]
    exact_sse = all(type(row.get("sse")) is int and type(row.get("pixels")) is int for row in metric_rows)
    has_cap = any(row["frame_psnr"] >= 100 for row in metric_rows)
    psnr_method = ("exact pooled integer SSE" if exact_sse else
                   "derived from saved rounded per-frame PSNR; 100 dB cap gives an MSE upper bound" if has_cap else
                   "derived from saved rounded per-frame PSNR")
    psnr_guard = historical_psnr_guard(fidelity, exact_sse, has_cap)
    whole = []
    for p in report["points"]:
        for arm in ("baseline", "streak"):
            rows = p[arm]["metrics"]
            whole.append({"codec": p["codec"], "setting": p["point"], "arm": arm,
                          "bytes": p["native_bytes"] if arm == "baseline" else p["wire_bytes"],
                          "frame_psnr": metric_value(rows, "frame_psnr"),
                          "full_lpips": sum(r["full_lpips"] for r in rows) / len(rows),
                          "full_dists": sum(r["full_dists"] for r in rows) / len(rows)})
    native = []
    for p in report["points"]:
        scores = p["baseline"]["judges"]["wasb"]["scores"]
        native.append({"rate": p["native_bytes"] * 8 / float(duration) / 1000,
                       "recall": scores["recall10"], "false": scores["false_on_absent"]})
    if any(x["recall"] is None for x in native):
        raise ValueError("clip has no visible labels")
    # Existing achievable native envelope, with quality interpolation in log rate.
    hull = analysis.envelope(analysis._unique_rates([(x["rate"], x["recall"]) for x in native]))
    support = (min(x["rate"] for x in native), max(x["rate"] for x in native))
    low = []
    for codec, setting in LOW:
        p = next(x for x in report["points"] if (x["codec"], x["point"]) == (codec, setting))
        rate = p["wire_bytes"] * 8 / float(duration) / 1000
        scores = p["streak"]["judges"]["wasb"]["scores"]
        anchor = analysis.quality_at(hull, rate, support)
        baseline_false = interpolate_native_false(native, rate)
        low.append({"codec": codec, "setting": setting, "candidate_bytes": p["wire_bytes"], "rate_kbps": rate,
                    "candidate_recall10": scores["recall10"], "native_recall10": anchor,
                    "recall_gain": None if anchor is None else scores["recall10"] - anchor,
                    "candidate_false": scores["false_on_absent"], "native_false_at_rate": baseline_false})
    supported = all(x["recall_gain"] is not None for x in low)
    mean_gain = sum(x["recall_gain"] for x in low) / 2 if supported else None
    current_qualified, historical_qualified, qualified = task_source_eligibility(report["clip"], source["qualified"])
    return {"clip_id": report["clip"], "game": report["clip"].split("/")[1], "frames": len(labels),
            "visible": counts["1"], "absent": counts["0"], "unknown": counts["-1"],
            "source_qualified": qualified, "current_source_qualified": current_qualified,
            "historical_source_qualified": historical_qualified, "source_scores": source["scores"]["wasb"],
            "task_status": "source_failed" if not qualified else "unsupported" if not supported else "evaluated",
            "low_points": low, "mean_recall_gain": mean_gain,
            "task_pass": bool(qualified and mean_gain is not None and mean_gain >= .10),
            "dcvc_vs_dcvc_psnr_bd_rate_guard_at_most_plus_one_percent": psnr_guard,
            "historical_psnr_guard_scope": "DCVC streak vs DCVC native; original four windows failed; pointwise guard is separate",
            "psnr_method": psnr_method,
            "full_frame_quality": whole, "fidelity": fidelity}


def summarize(selection: dict[str, Any], directories: list[Path]) -> dict[str, Any]:
    if encoder.oracle.sha(HISTORICAL_EVIDENCE) != HISTORICAL_SHA:
        raise ValueError("historical source evidence hash mismatch")
    historical = json.loads(HISTORICAL_EVIDENCE.read_text())["source_job"]["per_clip_gate"]["failing_clips"]
    if {row["clip_id"] for row in historical} != HISTORICAL_FAILURES:
        raise ValueError("historical source exclusion list differs")
    clips = selection["tracknet"]["clips"]
    ids = [c["clip_id"] for c in clips]
    if len(ids) != 30 or len(set(ids)) != 30 or len(directories) != 30:
        raise ValueError("exactly 30 distinct registered clip outputs required")
    reports = {}
    for path in directories:
        report = json.loads((path / "pilot.json").read_text())
        clip = report["clip"]
        if clip not in ids or clip in reports:
            raise ValueError("unregistered or duplicate clip")
        reports[clip] = path
    if set(reports) != set(ids):
        raise ValueError("missing registered clips")
    result = [summarize_clip(reports[c["clip_id"]], c) for c in clips]
    totals = {name: sum(c[name] for c in result) for name in ("frames", "visible", "absent", "unknown")}
    if totals != {"frames": 5675, "visible": 5472, "absent": 197, "unknown": 6}:
        raise ValueError("registered population coverage mismatch")
    source_false = sum(c["source_scores"]["false_on_absent"] for c in result)
    source_false_rate = source_false / totals["absent"]
    passes = sum(c["task_pass"] for c in result)
    unmatched = [(c["clip_id"], x["codec"], x["setting"]) for c in result if c["absent"] for x in c["low_points"]
                 if x["native_false_at_rate"] is None]
    setting_excess: dict[str, Any] = {}
    for codec, setting in LOW:
        rows = [x for c in result if c["absent"] for x in c["low_points"]
                if (x["codec"], x["setting"]) == (codec, setting)]
        complete = all(x["native_false_at_rate"] is not None for x in rows)
        candidate = sum(x["candidate_false"] for x in rows)
        anchor = sum(x["native_false_at_rate"] for x in rows) if complete else None
        setting_excess[f"{codec}-{setting}"] = {"candidate_false": candidate, "native_false_at_rate": anchor,
                                               "absent_frames": totals["absent"],
                                               "excess_per_absent": None if anchor is None else (candidate - anchor) / totals["absent"]}
    excess = None if unmatched else sum(x["excess_per_absent"] for x in setting_excess.values()) / 2
    absence_pass = excess is not None and excess <= .05
    games = {}
    for game in ("game8", "game9", "game10"):
        members = [c for c in result if c["game"] == game]
        games[game] = {"clips": len(members), "source_qualified": sum(c["source_qualified"] for c in members),
                       "task_passes": sum(c["task_pass"] for c in members),
                       "absent_frames": sum(c["absent"] for c in members)}
    return {"scope": "retrospective selection-exposed benchmark; no independent-match inference",
            "task_endpoint_scope": "new 30-clip descriptive target, separate from the old 6/8 whole-contract rule",
            "fidelity_comparison_layout": "metric -> candidate codec family -> native anchor family; pooled uses both families",
            "coverage": totals, "source_false": source_false, "source_false_per_negative": source_false_rate,
            "strong_absence_claim_allowed": False,
            "strong_absence_claim_reason": "prior frozen source negative calibration failed: 10/197 > 0.05",
            "qualified_clips": sum(c["source_qualified"] for c in result),
            "current_source_qualified_clips": sum(c["current_source_qualified"] for c in result),
            "historical_source_evidence_sha256": HISTORICAL_SHA, "passing_clips": passes,
            "required_passing_clips": 23, "unmatched_absence_rates": unmatched,
            "pooled_mean_excess_false_per_absent": excess, "false_excess_by_setting": setting_excess,
            "false_limit_pass": absence_pass,
            "task_endpoint_pass": passes >= 23 and absence_pass,
            "false_interpolation": "linear in log native rate; tied native rates take minimum false count; two settings equally weighted",
            "games": games, "clips": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-output", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = summarize(json.loads(SOURCE.read_text()), args.clip_output)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
