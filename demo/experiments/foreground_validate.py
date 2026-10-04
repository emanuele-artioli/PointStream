"""Smoke-stage validator. Exit zero is not acceptance by itself."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from experiments.jobs.monitor import write_json


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_report(report: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "diagnostic_not_citable": report.get("citable") is False and report.get("label") == "diagnostic",
        "schema": isinstance(report.get("schema"), str) and report["schema"].startswith("pointstream.foreground."),
        "stage": report.get("stage") in {"smoke", "full", "local"},
        "substantive": False,
    }
    kind = report.get("kind")
    if kind == "packet":
        methods = report.get("methods") or {}
        checks["three_methods"] = set(methods) == {"raw", "zlib", "delta_zlib"}
        checks["file_bytes"] = all(
            _finite(item.get("bytes")) and item.get("bytes") == item.get("file_bytes") and item["bytes"] > 0
            for item in methods.values()
        )
        checks["same_codes"] = report.get("codes_identical") is True
        checks["substantive"] = all(checks[key] for key in ("three_methods", "file_bytes", "same_codes"))
    elif kind == "audit":
        checks["pending_or_reviewed"] = report.get("pixel_candidates", 0) > 0
        checks["no_fabricated_joint_scores"] = report.get("fabricated_joint_scores") is False
        checks["substantive"] = checks["pending_or_reviewed"] and checks["no_fabricated_joint_scores"]
    elif kind == "fit":
        checks["within_step_cap"] = _finite(report.get("max_steps_run", -1)) and report["max_steps_run"] <= 120
        checks["gate_recorded"] = "fit_gate" in report
        checks["substantive"] = checks["within_step_cap"] and checks["gate_recorded"]
    elif kind == "profile":
        checks["rates_present"] = "frames_per_s" in report and "crops_per_s" in report
        checks["provider_explicit"] = report.get("provider") in {"cuda", "cpu_fallback", "cpu", "mixed", "blocked", "unprofiled"}
        checks["substantive"] = checks["rates_present"] and checks["provider_explicit"]
    elif kind == "compare":
        checks["eight_frame_cut"] = report.get("frame_count") == 8 and report.get("start_index") == 120
        checks["codes_or_blocked"] = report.get("codes_identical") is True or report.get("status") == "blocked"
        checks["substantive"] = checks["eight_frame_cut"] and checks["codes_or_blocked"]
    else:
        checks["known_kind"] = False
    checks["passed_gate"] = all(value is True for key, value in checks.items() if key != "passed_gate")
    return checks


def main() -> int:
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    report_path = stage_dir / "report.json"
    destination = Path(os.environ["PS_VALIDATION_PATH"])
    if not report_path.is_file():
        write_json(destination, {"passed": False, "checks": {"report_present": False}})
        return 1
    report = json.loads(report_path.read_text())
    checks = validate_report(report)
    passed = checks.get("passed_gate") is True and checks.get("substantive") is True
    write_json(destination, {"passed": passed, "checks": checks, "citable": False})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
