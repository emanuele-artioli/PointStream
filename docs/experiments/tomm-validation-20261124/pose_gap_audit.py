"""Audit pose-tracking gaps and interpolation potential across official TEST coverage clips."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "pose-gap-audit.json"

CROP_RECEIPTS = {
    "match108": ROOT / "docs/experiments/test-crop-joint-20261122/match108-result.json",
    "match102": ROOT / "docs/experiments/test-crop-joint-20261122/match102-result.json",
    "match120": ROOT / "docs/experiments/test-crop-joint-20261122/match120-result.json",
    "match127": ROOT / "docs/experiments/test-joint-extension-20261122/match127-crop-result.json",
    "match105": ROOT / "docs/experiments/test-joint-extension-20261122/match105-crop-result.json",
    "match116": ROOT / "docs/experiments/test-joint-extension-20261122/match116-crop-result.json",
}


def contiguous_spans(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    spans: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((start, prev))
            start = prev = idx
    spans.append((start, prev))
    return spans


def summarize_missing(missing: list[int], total_frames: int) -> dict[str, Any]:
    spans = contiguous_spans(missing)
    lengths = [end - start + 1 for start, end in spans]
    touches_boundary = any(start == 0 or end == total_frames - 1 for start, end in spans)
    interior_lengths = [
        end - start + 1
        for start, end in spans
        if start > 0 and end < total_frames - 1
    ]
    return {
        "missing_count": len(missing),
        "gap_spans": [[s, e, e - s + 1] for s, e in spans],
        "gap_lengths": lengths,
        "max_gap": max(lengths, default=0),
        "touches_boundary": touches_boundary,
        "max_interior_gap": max(interior_lengths, default=0),
        "gaps_le_3": sum(1 for length in lengths if length <= 3),
        "gaps_le_5": sum(1 for length in lengths if length <= 5),
        "gaps_le_10": sum(1 for length in lengths if length <= 10),
        "gaps_gt_10": sum(1 for length in lengths if length > 10),
    }


def main() -> None:
    coverage_path = ROOT / "docs/experiments/official-test-player-coverage-20261122/coverage-summary.json"
    coverage = json.loads(coverage_path.read_text())
    thresholds = [3, 5, 10, 15, 20]

    matches: dict[str, Any] = {}
    for row in coverage["rows"]:
        match_id = row["match"]
        entry: dict[str, Any] = {
            "event": row["event"],
            "status": row["status"],
            "court_ticks": row["court_ticks"],
            "paired_ticks": row["paired_ticks"],
            "full": row["full"],
        }
        receipt_path = CROP_RECEIPTS.get(match_id)
        if receipt_path and receipt_path.exists():
            data = json.loads(receipt_path.read_text())
            total_frames = int(data["frames"])
            per_player: dict[str, Any] = {}
            paired_missing: set[int] = set()
            for pid in data["selected_ids"]:
                tinfo = data["tracks"][pid]
                missing = [int(t) for t in tinfo.get("missing", [])]
                paired_missing.update(missing)
                per_player[pid] = {
                    "median_height_px": data["median_height_px"][pid],
                    "fitted": tinfo["fitted"],
                    "mask": tinfo["mask"],
                    "setting": tinfo["setting"],
                    **summarize_missing(missing, total_frames),
                }
            paired_summary = summarize_missing(sorted(paired_missing), total_frames)
            recoverable_at = {}
            for k in thresholds:
                spans = contiguous_spans(sorted(paired_missing))
                # Linear interpolation requires endpoints on both sides (interior gaps <= k).
                unrecoverable = sum(
                    e - s + 1
                    for s, e in spans
                    if s == 0 or e == total_frames - 1 or (e - s + 1) > k
                )
                recoverable_at[str(k)] = {
                    "covered_ticks": total_frames - unrecoverable,
                    "reaches_full_240": unrecoverable == 0,
                }
            entry["players"] = per_player
            entry["paired_missing"] = paired_summary
            entry["interpolation_thresholds"] = recoverable_at
        matches[match_id] = entry

    measured_ids = [m for m, v in matches.items() if v["status"] == "measured"]
    threshold_totals = {
        str(k): {
            "full_240_matches": [
                m for m in measured_ids if matches[m]["interpolation_thresholds"][str(k)]["reaches_full_240"]
            ],
            "count_full_240": sum(
                1 for m in measured_ids if matches[m]["interpolation_thresholds"][str(k)]["reaches_full_240"]
            ),
            "total_paired_ticks": sum(
                matches[m]["interpolation_thresholds"][str(k)]["covered_ticks"] for m in measured_ids
            ),
        }
        for k in thresholds
    }

    summary = {
        "enrolled_total": len(coverage["rows"]),
        "measured_count": len(measured_ids),
        "unmeasured_count": sum(1 for v in matches.values() if v["status"] == "unmeasured"),
        "ordinary_routed_count": sum(1 for v in matches.values() if v["status"] == "ordinary"),
        "raw_full_240_count": sum(1 for m in measured_ids if matches[m]["full"]),
        "threshold_totals": threshold_totals,
        "matches": matches,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"threshold_totals": threshold_totals}, indent=2))
    for m in measured_ids:
        info = matches[m]
        pm = info["paired_missing"]
        heights = {pid: p["median_height_px"] for pid, p in info["players"].items()}
        print(
            f"{m} ({info['event']}): paired={info['paired_ticks']}/240, "
            f"max_gap={pm['max_gap']}, interior_max={pm['max_interior_gap']}, "
            f"spans={pm['gap_spans']}, heights={heights}"
        )


if __name__ == "__main__":
    main()
