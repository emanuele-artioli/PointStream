"""BD-rate between a scene ladder's PointStream arm and its anchor arm.

`ladder_scenes.py` records both arms per rung; this integrates them. Kept
separate so a long ladder run never has to be repeated to re-read its own
result, and so a curve can be refused *after* the encodes rather than instead
of them.

**It refuses more often than it reports, on purpose.** A BD-rate is only
meaningful between two curves that overlap in quality and that each vary with
the knob being swept, so this checks before integrating:

- the plate must move across rungs (`plans/BP31-findings.md` §8: a run whose
  plate was byte-identical at all five rungs produced a smooth, monotone and
  entirely fictional curve);
- both arms must be monotone in rate and quality;
- `compare_rd_curves` enforces the overlap span itself and raises rather than
  extrapolating.

The bound is `outputs/bp31-ladder/bounds-before-run.json`, written before the
first encode: **[-20%, +150%]**, and a *negative* result — PointStream winning —
is an alarm to be checked, not a headline.

Run::

    python -m experiments.tier.ladder_scenes_compare <ladder-json> [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.components.metrics.bd_rate import (
    InsufficientOverlapError,
    RDCurve,
    compare_rd_curves,
)

#: From the bounds file, duplicated here so the check runs beside the number.
BD_RATE_BAND: tuple[float, float] = (-20.0, 150.0)


def _curve(points: list[tuple[int, float]], label: str) -> RDCurve:
    ordered = sorted(points, key=lambda item: item[0])
    return RDCurve(
        rates=tuple(float(rate) for rate, _ in ordered),
        qualities=tuple(float(quality) for _, quality in ordered),
        label=label,
    )


def compare(report: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in report.get("rows", []) if row.get("pointstream") and row.get("anchor")]
    out: dict[str, Any] = {
        "video": report.get("video"),
        "scenes": report.get("scenes"),
        "method": (report.get("background") or {}).get("method"),
        "plate_knob": report.get("plate_knob"),
        "n_rungs": len(rows),
        "refusals": [],
    }
    if len(rows) < 4:
        out["refusals"].append(f"{len(rows)} usable rungs; a BD-rate needs at least 4.")
        return out

    plates = [int(row["pointstream"]["parts"]["panorama"]) for row in rows]
    if len(set(plates)) == 1:
        out["refusals"].append(
            f"the plate is {plates[0]} B at every rung. This curve swept the residual "
            "against a frozen background and is not a payload ladder (findings §8)."
        )
        return out

    stream_points = [(int(r["pointstream"]["coded_bytes"]), float(r["pointstream"]["psnr_y_dB"])) for r in rows]
    anchor_points = [(int(r["anchor"]["joint_bytes"]), float(r["anchor"]["psnr_y_dB"])) for r in rows]
    out["pointstream_curve"] = stream_points
    out["anchor_curve"] = anchor_points

    try:
        bd_rate = compare_rd_curves(
            _curve(anchor_points, "anchor"), _curve(stream_points, "pointstream")
        )
    except InsufficientOverlapError as exc:
        out["refusals"].append(f"no readable overlap: {exc}")
        return out

    # `BDComparison.bd_rate` is a FRACTION ("+1.168" is +116.8%), not a
    # percentage. AGENTS.md records a bound that fired against a correct result
    # because it had been derived in the wrong units; the band below is written
    # in percent, so the conversion happens here, once, next to the comparison.
    value = float(bd_rate.bd_rate) * 100.0
    out["bd_rate_percent"] = round(value, 2)
    out["bd_quality_dB"] = round(float(bd_rate.bd_quality), 3)
    out["overlap_dB"] = [round(x, 2) for x in bd_rate.overlap]
    out["overlap_fraction"] = round(float(bd_rate.overlap_fraction), 3)
    low, high = BD_RATE_BAND
    if value < low:
        out["alarm"] = (
            f"BD-rate {value:.2f}% is below the pre-written floor of {low}%. PointStream "
            "winning is an ALARM, not a triumph. Before reporting it check: the anchor got "
            "the same scenes concatenated (joint_over_separate < 1.0), quality came from "
            "delivered_frames, and the decode named an explicit -c:v."
        )
    elif value > high:
        out["alarm"] = (
            f"BD-rate {value:.2f}% is above the pre-written ceiling of {high}%, which would "
            "mean the plate levers made things worse than the fresh-plate baseline. Suspect "
            "the wiring before the content."
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args(argv)

    for path in args.reports:
        report = json.loads(Path(path).read_text())
        result = compare(report)
        print(f"\n=== {Path(path).name} ({result.get('method')}, knob={result.get('plate_knob')})")
        for refusal in result["refusals"]:
            print(f"  REFUSED: {refusal}")
        if "bd_rate_percent" in result:
            print(f"  BD-rate vs anchor: {result['bd_rate_percent']:+.2f}%")
        if "alarm" in result:
            print(f"  ! {result['alarm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
