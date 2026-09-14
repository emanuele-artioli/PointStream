"""Versioned independent-match count for confirmation and development pilots.

Gate B historically hardcoded six matches. The September 2026 campaign allows a
smaller prospective count with explicit small-sample limits. Callers must load
this policy rather than embedding a literal six.

Confirmation ingest rejects pending/unaccepted policy, boolean counts, and
unknown stages. Exposed Gate B sources must not be used as a fresh-holdout
fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

SCHEMA_ID = "pointstream.confirmation_source_count.v1"
POLICY_ID = "evaluation_20260914_source_count.e01r"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_source_count_policy.json"

UNACCEPTED_STATUSES = frozenset(
    {
        "proposed_pending_coordinator_acceptance",
        "pending",
        "unaccepted",
        "draft",
    }
)

Stage = Literal["development_pilot", "confirmation"]
STAGES: tuple[Stage, ...] = ("development_pilot", "confirmation")

_LABELS = (
    (1, "case_study"),
    (2, "restricted_confirmation"),
    (3, "small_sample_confirmation"),
    (6, "preferred_confirmation"),
)


def confirmation_label(n: int) -> str:
    """Map an independent-match count to the campaign claim label."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return "insufficient"
    chosen = "case_study"
    for threshold, label in _LABELS:
        if n >= threshold:
            chosen = label
    return chosen


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive int, not {value!r}")
    return value


def load_source_count_policy(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the versioned source-count policy manifest."""
    policy_path = path or DEFAULT_POLICY_PATH
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"unsupported source-count schema: {data.get('schema')!r}")
    for stage in STAGES:
        block = data.get(stage) or {}
        _positive_int(block.get("required_matches"), f"{stage}.required_matches")
    _positive_int(data.get("preferred_matches"), "preferred_matches")
    fallback = (data.get("confirmation") or {}).get("fallback_if_unacquired") or {}
    if fallback.get("uses_exposed_sources") is True:
        raise ValueError(
            "fallback_if_unacquired.uses_exposed_sources cannot be true; "
            "exposed Gate B sources stay development or historical observations"
        )
    return data


def policy_identity(path: Path | None = None) -> dict[str, Any]:
    """Identity to pin on experiment provenance."""
    policy_path = path or DEFAULT_POLICY_PATH
    payload = policy_path.read_bytes()
    data = json.loads(payload.decode("utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "policy_id": data.get("policy_id") or POLICY_ID,
        "path": str(policy_path.relative_to(REPO_ROOT)) if policy_path.is_relative_to(REPO_ROOT) else str(policy_path),
        "sha256": digest,
        "status": data.get("status"),
        "confirmation_required_matches": (data.get("confirmation") or {}).get("required_matches"),
        "preferred_matches": data.get("preferred_matches"),
        "contract_revision": data.get("contract_revision"),
    }


def load_required_matches(stage: str, path: Path | None = None) -> int:
    """Return the fail-closed independent-match requirement for a stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown source-count stage {stage!r}; expected one of {STAGES}")
    policy = load_source_count_policy(path)
    if stage == "confirmation":
        status = policy.get("status")
        if status in UNACCEPTED_STATUSES or not status:
            raise ValueError(
                "confirmation rejects pending/unaccepted source-count policy; "
                f"status={status!r}"
            )
    return _positive_int(policy[stage]["required_matches"], f"{stage}.required_matches")
