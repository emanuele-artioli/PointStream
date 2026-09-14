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
    ):
        if key not in data:
            raise ValueError(f"operating-point contract missing {key}")
    return data
