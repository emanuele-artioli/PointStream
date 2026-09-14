"""Versioned independent-match count for confirmation and development pilots.

Gate B historically hardcoded six matches. The September 2026 campaign allows a
smaller prospective count with explicit small-sample limits. Callers must load
this policy rather than embedding a literal six.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

SCHEMA_ID = "pointstream.confirmation_source_count.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_source_count_policy.json"

Stage = Literal["development_pilot", "confirmation"]

_LABELS = (
    (1, "case_study"),
    (2, "restricted_confirmation"),
    (3, "small_sample_confirmation"),
    (6, "preferred_confirmation"),
)


def confirmation_label(n: int) -> str:
    """Map an independent-match count to the campaign claim label."""
    if n < 1:
        return "insufficient"
    chosen = "case_study"
    for threshold, label in _LABELS:
        if n >= threshold:
            chosen = label
    return chosen


def load_source_count_policy(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the versioned source-count policy manifest."""
    policy_path = path or DEFAULT_POLICY_PATH
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"unsupported source-count schema: {data.get('schema')!r}")
    for stage in ("development_pilot", "confirmation"):
        block = data.get(stage) or {}
        n = block.get("required_matches")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{stage}.required_matches must be a positive int")
    preferred = data.get("preferred_matches")
    if not isinstance(preferred, int) or preferred < 1:
        raise ValueError("preferred_matches must be a positive int")
    return data


def load_required_matches(stage: Stage, path: Path | None = None) -> int:
    """Return the fail-closed independent-match requirement for a stage."""
    policy = load_source_count_policy(path)
    return int(policy[stage]["required_matches"])
