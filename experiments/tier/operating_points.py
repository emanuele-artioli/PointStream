"""Load the frozen operating-point, colour, restoration and live-test contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_ID = "pointstream.campaign_operating_points.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_operating_points.json"


def load_operating_points(path: Path | None = None) -> dict[str, Any]:
    data = json.loads((path or DEFAULT_PATH).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"unsupported operating-point schema: {data.get('schema')!r}")
    for key in (
        "display_endpoints",
        "learning_crop",
        "colour_paths",
        "restoration",
        "quality_policy",
        "latency_policy",
        "sustained_live_test",
        "timestamp_sampling",
        "common_timebase_scoring",
    ):
        if key not in data:
            raise ValueError(f"operating-point contract missing {key}")
    live = data["sustained_live_test"]
    min_s = live.get("min_duration_seconds")
    if not isinstance(min_s, (int, float)) or isinstance(min_s, bool) or min_s < 30:
        raise ValueError("sustained_live_test.min_duration_seconds must be >= 30")
    if live.get("smoke_check_seconds") is None:
        raise ValueError("sustained_live_test.smoke_check_seconds is required")
    floor = (data.get("quality_policy") or {}).get("practical_quality_floor")
    if isinstance(floor, dict) and floor.get("vmaf") == 20.0 and floor.get("role") != "diagnostic_exclusion":
        raise ValueError("VMAF 20 is a diagnostic exclusion, not a practical quality floor")
    return data
