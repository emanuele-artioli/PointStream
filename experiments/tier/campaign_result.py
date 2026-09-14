"""Shared campaign result contract: claim eligibility, timing refs, plot ingest.

E03/E08 own the plots. This module only validates records and emits exclusion
reasons. A missing timing stratum never silently drops an otherwise valid RD row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

CLAIM_SCOPES: Final[tuple[str, ...]] = (
    "rd",
    "runtime",
    "standalone_transport",
    "generalization",
)

SCHEMA_ID = "pointstream.campaign_result.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_RECORDS_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_result_contract.example.json"

REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "artifact_id",
    "artifact_path",
    "artifact_sha256",
    "code_revision",
    "source_ids",
    "frame_ids",
    "operating_point_id",
    "claim_eligibility",
    "controls",
)

REQUIRED_CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "standalone_decode",
    "metric_calibration",
    "wire_ledger",
)


def _as_dict(payload: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(payload, Path):
        return json.loads(payload.read_text(encoding="utf-8"))
    return dict(payload)


def validate_campaign_record(record: Mapping[str, Any]) -> list[str]:
    """Return blockers; empty means the record is structurally usable."""
    blockers: list[str] = []
    if record.get("schema") != SCHEMA_ID:
        blockers.append(f"schema must be {SCHEMA_ID}")
    for field in REQUIRED_RECORD_FIELDS:
        if field not in record:
            blockers.append(f"missing field {field}")
    elig = record.get("claim_eligibility")
    if not isinstance(elig, dict):
        blockers.append("claim_eligibility must be an object")
        return blockers
    exclusions = elig.get("exclusions")
    if exclusions is None:
        exclusions = []
    if not isinstance(exclusions, list):
        blockers.append("claim_eligibility.exclusions must be a list")
        exclusions = []
    for claim in CLAIM_SCOPES:
        if claim not in elig:
            blockers.append(f"claim_eligibility missing {claim}")
            continue
        flag = elig[claim]
        if not isinstance(flag, bool):
            blockers.append(f"claim_eligibility.{claim} must be bool")
            continue
        if flag is False:
            matching = [
                item
                for item in exclusions
                if isinstance(item, dict) and item.get("claim") == claim and item.get("reason")
            ]
            if not matching:
                blockers.append(f"ineligible claim {claim} requires an exclusion reason")
        if flag is True and claim == "runtime":
            ref = record.get("timing_evidence")
            if not isinstance(ref, dict) or not ref.get("timing_evidence_id"):
                blockers.append("runtime eligibility requires timing_evidence.timing_evidence_id")
    controls = record.get("controls")
    if isinstance(controls, dict):
        for field in REQUIRED_CONTROL_FIELDS:
            if field not in controls:
                blockers.append(f"controls missing {field}")
    return blockers


def ingest_for_claim(
    records: Iterable[Mapping[str, Any]],
    claim: str,
) -> dict[str, Any]:
    """Keep eligible rows; every drop carries an explicit reason."""
    if claim not in CLAIM_SCOPES:
        raise ValueError(f"unknown claim {claim!r}; expected one of {CLAIM_SCOPES}")
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        artifact_id = str(record.get("artifact_id") or "")
        structure = validate_campaign_record(record)
        if structure:
            excluded.append(
                {
                    "artifact_id": artifact_id,
                    "reason": "; ".join(structure),
                }
            )
            continue
        elig = record["claim_eligibility"]
        if elig.get(claim) is True:
            kept.append(record)
            continue
        reasons = [
            item["reason"]
            for item in elig.get("exclusions") or []
            if isinstance(item, dict) and item.get("claim") == claim and item.get("reason")
        ]
        excluded.append(
            {
                "artifact_id": artifact_id,
                "reason": reasons[0] if reasons else f"claim {claim} not eligible",
            }
        )
    return {"claim": claim, "kept": kept, "excluded": excluded, "n_kept": len(kept), "n_excluded": len(excluded)}


def load_example_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the checked-in example records used by tests and E02/E03 adapters."""
    payload = _as_dict(path or EXAMPLE_RECORDS_PATH)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("example contract must contain records[]")
    return [dict(item) for item in records]
