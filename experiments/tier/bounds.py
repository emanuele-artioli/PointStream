"""Check a finished tier report against the bounds written before it ran.

Separate from `run.py` on purpose: the bounds were written first, the run
happened second, and the comparison is a third step that can be re-run without
re-running anything expensive. A band that fires is an alarm to investigate, not
a finding to report — and the alarm has to be legible without re-reading two
JSON files side by side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "bp23-tier"

#: Tier name -> the key its delivered-PSNR band lives under in the bounds file.
_DELIVERED_KEYS = {
    "fast": "fast_coarse",
    "balanced": "balanced_medium",
    "quality": "quality_fine",
}
_RESIDUAL_KEYS = {
    "fast": "fast_coarse_fraction_of_source",
    "balanced": "balanced_medium_fraction_of_source",
    "quality": "quality_fine_fraction_of_source",
}


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    low: float | None
    high: float | None
    inside: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "value": self.value,
            "band": [self.low, self.high],
            "inside_band": self.inside,
            "note": self.note,
        }


def _number(value: Any) -> float | None:
    """JSON carries infinities as strings, because JSON has no infinity."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.lstrip("-") == "inf":
        return float(value)
    return None


def _scoped(report: dict[str, Any], metric: str, role: str) -> float | None:
    for item in report.get("scoped", []):
        if item["metric"] == metric and item["role"] == role:
            return _number(item["value"])
    return None


def _band(value: float | None, low: float, high: float, name: str, note: str = "") -> Check:
    inside = value is not None and low <= value <= high
    return Check(name=name, value=value, low=low, high=high, inside=inside, note=note)


def check(report_path: Path, bounds_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bounds = json.loads(bounds_path.read_text(encoding="utf-8"))["bounds"]
    runs = {item["tier"]: item for item in report["runs"]}
    source_bytes = float(report["clip"]["source_bytes"])

    checks: list[Check] = []

    control = runs.get("all-off (control)")
    if control is not None:
        checks.append(
            Check(
                name="all-off control is the source",
                value=1.0 if control.get("source_bit_identical") else 0.0,
                low=1.0,
                high=1.0,
                inside=bool(control.get("source_bit_identical")),
                note="1 = bit-identical to the source clip",
            )
        )

    for tier in ("fast", "balanced", "quality"):
        item = runs.get(tier)
        if item is None:
            continue
        recon = item["reconstruction_quality"]
        delivered = item["delivered_quality"]

        band = bounds["unaided_whole_frame_psnr_dB"]
        checks.append(
            _band(
                _number(recon["pixel_psnr_dB"]),
                band["low"],
                band["high"],
                f"{tier}: unaided whole-frame PSNR (pooled-MSE convention)",
                band["basis"],
            )
        )
        band = bounds["unaided_object_region_psnr_dB"]
        checks.append(
            _band(
                _scoped(recon, "psnr", "object"),
                band["low"],
                band["high"],
                f"{tier}: unaided object-region PSNR",
                band["basis"],
            )
        )
        band = bounds["unaided_background_region_psnr_dB"]
        checks.append(
            _band(
                _scoped(recon, "psnr", "background"),
                band["low"],
                band["high"],
                f"{tier}: unaided background-region PSNR",
                band["basis"],
            )
        )
        band = bounds["delivered_whole_frame_psnr_dB"][_DELIVERED_KEYS[tier]]
        checks.append(
            _band(
                _number(delivered["pixel_psnr_dB"]),
                band["low"],
                band["high"],
                f"{tier}: delivered whole-frame PSNR",
            )
        )
        band = bounds["residual_payload_bytes"][_RESIDUAL_KEYS[tier]]
        residual = float(item["sizes_bytes"]["residual"])
        checks.append(
            _band(
                residual / source_bytes,
                band["low"],
                band["high"],
                f"{tier}: residual payload as a fraction of source pixels",
            )
        )
        band = bounds["wall_clock_per_tier"]
        checks.append(
            _band(
                float(item["wall_clock_seconds"]) / 60.0,
                band["low_minutes"],
                band["high_minutes"],
                f"{tier}: wall clock (minutes)",
            )
        )
        checks.append(
            Check(
                name=f"{tier}: encoder/client symmetry with no generator in the path",
                value=1.0 if item["encoder_client_symmetry"]["bit_identical"] else 0.0,
                low=1.0,
                high=1.0,
                inside=bool(item["encoder_client_symmetry"]["bit_identical"]),
                note=bounds["symmetry_encoder_vs_client"]["basis"],
            )
        )
        requested = set(item["config"]["requested_metrics"])
        reported = set(item["metrics_reported"])
        checks.append(
            Check(
                name=f"{tier}: every requested metric was reported",
                value=float(len(requested - reported)),
                low=0.0,
                high=0.0,
                inside=requested <= reported,
                note=f"requested {sorted(requested)}, reported {sorted(reported)}",
            )
        )
        calls = item["disabled_stage_calls"]
        checks.append(
            Check(
                name=f"{tier}: stages switched off were never invoked",
                value=float(sum(calls.values())),
                low=0.0,
                high=0.0,
                inside=all(count == 0 for count in calls.values()),
                note=f"{calls}",
            )
        )

    ladder = [
        (tier, _number(runs[tier]["delivered_quality"]["pixel_psnr_dB"]))
        for tier in ("fast", "balanced", "quality")
        if tier in runs
    ]
    psnrs = [value for _tier, value in ladder if value is not None]
    checks.append(
        Check(
            name="pre-registered ordering: delivered PSNR rises with the tier",
            value=None,
            low=None,
            high=None,
            inside=psnrs == sorted(psnrs),
            note=f"{ladder}",
        )
    )
    bytes_ladder = [
        (tier, int(runs[tier]["sizes_bytes"]["residual"]))
        for tier in ("fast", "balanced", "quality")
        if tier in runs
    ]
    counts = [value for _tier, value in bytes_ladder]
    checks.append(
        Check(
            name="pre-registered ordering: residual payload rises with the tier",
            value=None,
            low=None,
            high=None,
            inside=counts == sorted(counts),
            note=f"{bytes_ladder}",
        )
    )

    fired = [item.as_dict() for item in checks if not item.inside]
    return {
        "report": str(report_path),
        "bounds": str(bounds_path),
        "n_checks": len(checks),
        "n_alarms": len(fired),
        "alarms": fired,
        "all_checks": [item.as_dict() for item in checks],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(OUT_DIR / "report.json"))
    parser.add_argument("--bounds", default=str(OUT_DIR / "bounds-before-run.json"))
    parser.add_argument("--out", default=str(OUT_DIR / "bounds-check.json"))
    args = parser.parse_args(argv)

    outcome = check(Path(args.report), Path(args.bounds))
    Path(args.out).write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(f"{outcome['n_checks']} checks, {outcome['n_alarms']} alarms")
    for alarm in outcome["alarms"]:
        print(f"  ALARM {alarm['check']}: {alarm['value']} outside {alarm['band']}")
    return 0


__all__ = ["Check", "check", "main"]
