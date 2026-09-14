"""Shared campaign result contract: claim eligibility, timing refs, plot ingest.

E03/E08 own the plots. This module validates records and emits exclusion
reasons. Structural checks are not claim eligibility. Production ingest also
checks identities, provenance class, finite evidence, and control types.

A missing timing stratum never silently drops an otherwise valid RD row.
An unrun or placeholder timing row never certifies runtime.
Full-trajectory coverage is claim ``trajectory``, not ``generalization``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Mapping

CLAIM_SCOPES: Final[tuple[str, ...]] = (
    "rd",
    "runtime",
    "standalone_transport",
    "trajectory",
    "generalization",
)

RECORD_CLASSES: Final[tuple[str, ...]] = (
    "synthetic_example",
    "pending_run",
    "historical_observation",
    "recoverable_evidence",
    "validated_claim",
)

INGEST_PURPOSES: Final[tuple[str, ...]] = (
    "structure",
    "diagnostic",
    "validated",
)

SCHEMA_ID = "pointstream.campaign_result.v1"
CONTRACT_REVISION = "e01r-20260914"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_RECORDS_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_result_contract.example.json"
PRODUCER_EXAMPLE_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_e02_producer.example.json"

REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "contract_revision",
    "record_class",
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

CONTROL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "verified",
        "reconciled",
        "present",
        "inherited_named_artifact",
        "unverified",
        "unverified_on_this_artifact",
        "failed",
        "not_this_row",
        "not_applicable",
        "partial",
        "recorded_but_legacy_ssim_ceiling_alarm",
        "present_in_later_recovery_not_this_file",
        "parts_recorded_reconciliation_audited_later",
        "matrix_local",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_IDS = frozenset({"", "pending", "unassigned", "placeholder", "example"})
_ZERO_SHA256 = "0" * 64

Purpose = Literal["structure", "diagnostic", "validated"]


def _as_dict(payload: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(payload, Path):
        return json.loads(payload.read_text(encoding="utf-8"))
    return dict(payload)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in _PLACEHOLDER_IDS


def _meaningful_source_ids(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(_nonempty_str(item) for item in value)


def _meaningful_frame_ids(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if "native_timestamps" in value:
        stamps = value["native_timestamps"]
        if not isinstance(stamps, list) or not stamps:
            return False
        return all(isinstance(item, (int, float)) for item in stamps)
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return False
    start = value.get("start", 0)
    if start is not None and (isinstance(start, bool) or not isinstance(start, int) or start < 0):
        return False
    return True


def _sha256_ok(value: Any, *, allow_zero: bool) -> bool:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        return False
    if value == _ZERO_SHA256 and not allow_zero:
        return False
    return True


def _control_ok(value: Any) -> bool:
    if isinstance(value, dict):
        status = value.get("status")
        return isinstance(status, str) and status in CONTROL_STATUSES
    return isinstance(value, str) and value in CONTROL_STATUSES


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _rd_arms(record: Mapping[str, Any]) -> dict[str, bool]:
    elig = record.get("claim_eligibility") or {}
    arms = elig.get("rd_arms")
    if isinstance(arms, dict):
        return {str(name): bool(flag) for name, flag in arms.items()}
    evidence = record.get("evidence") or {}
    metrics = evidence.get("metrics") or {}
    bytes_block = evidence.get("bytes") or {}
    derived: dict[str, bool] = {}
    for name, raw in metrics.items():
        derived[str(name)] = _finite_number(raw)
    total = bytes_block.get("total")
    derived["bytes"] = _finite_number(total)
    return derived


def _class_allows_purpose(record_class: str, purpose: Purpose) -> bool:
    if purpose == "structure":
        return True
    if record_class in {"synthetic_example", "pending_run"}:
        return False
    if purpose == "diagnostic":
        return record_class in {
            "historical_observation",
            "recoverable_evidence",
            "validated_claim",
        }
    return record_class == "validated_claim"


def _timing_verified(record: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    ref = record.get("timing_evidence")
    if not isinstance(ref, dict):
        return ["runtime eligibility requires timing_evidence object"]
    timing_id = ref.get("timing_evidence_id")
    if not _nonempty_str(timing_id):
        blockers.append("runtime eligibility requires timing_evidence.timing_evidence_id")
    elif "not-yet-run" in str(timing_id) or "gpuX" in str(timing_id):
        blockers.append("runtime eligibility rejects placeholder timing_evidence_id")
    host = ref.get("host")
    if not _nonempty_str(host) or str(host) == "unassigned":
        blockers.append("runtime eligibility requires a named host stratum")
    path = str(record.get("artifact_path") or "")
    if "not-yet-run" in path:
        blockers.append("runtime eligibility rejects unrun artifact paths")
    n_repeats = ref.get("n_repeats", ref.get("sample_count"))
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        blockers.append("runtime eligibility requires a positive integer repeat count")
    return blockers


def _validated_rd_blockers(record: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        blockers.append("validated RD requires evidence.metrics and evidence.bytes")
        return blockers
    arms = _rd_arms(record)
    salvageable = [name for name, flag in arms.items() if flag]
    if not salvageable:
        blockers.append("validated RD requires at least one finite metric or byte arm")
    if not arms.get("bytes"):
        blockers.append("validated RD requires a finite byte/rate arm")
    quality_arms = [name for name in salvageable if name != "bytes"]
    if not quality_arms:
        blockers.append("validated RD requires at least one finite quality metric arm")
    controls = record.get("controls") or {}
    ledger = controls.get("wire_ledger")
    calib = controls.get("metric_calibration")
    decode = controls.get("standalone_decode")
    if isinstance(ledger, str) and ledger in {"unverified", "unverified_on_this_artifact", "failed"}:
        blockers.append("validated RD requires a verified or reconciled wire ledger")
    if isinstance(calib, str) and calib in {"unverified", "unverified_on_this_artifact", "failed"}:
        blockers.append("validated RD requires verified or inherited metric calibration")
    if isinstance(decode, str) and decode in {"unverified", "unverified_on_this_artifact", "failed"}:
        blockers.append("validated RD requires verified standalone decode of scored output")
    return blockers


def validate_campaign_record(
    record: Mapping[str, Any],
    *,
    purpose: Purpose = "structure",
) -> list[str]:
    """Return blockers. ``purpose='structure'`` is schema-only.

    Production callers must use ``purpose='validated'`` or ``'diagnostic'``.
    """
    if purpose not in INGEST_PURPOSES:
        raise ValueError(f"unknown ingest purpose {purpose!r}")
    blockers: list[str] = []
    if record.get("schema") != SCHEMA_ID:
        blockers.append(f"schema must be {SCHEMA_ID}")
    if record.get("contract_revision") != CONTRACT_REVISION:
        blockers.append(f"contract_revision must be {CONTRACT_REVISION}")
    record_class = record.get("record_class")
    if record_class not in RECORD_CLASSES:
        blockers.append("record_class must be a known evidence class")
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
    controls = record.get("controls")
    if purpose == "structure":
        if isinstance(controls, dict):
            for field in REQUIRED_CONTROL_FIELDS:
                if field not in controls:
                    blockers.append(f"controls missing {field}")
        return blockers

    allow_zero = record_class in {"synthetic_example", "pending_run"}
    if not _nonempty_str(record.get("artifact_id")):
        blockers.append("artifact_id must be a meaningful identity")
    if not _nonempty_str(record.get("artifact_path")):
        blockers.append("artifact_path must be a meaningful identity")
    if not _sha256_ok(record.get("artifact_sha256"), allow_zero=allow_zero):
        blockers.append("artifact_sha256 must be a 64-char hex digest")
    if not _nonempty_str(record.get("code_revision")):
        blockers.append("code_revision must be a meaningful identity")
    if not _meaningful_source_ids(record.get("source_ids")):
        blockers.append("source_ids must be a non-empty list of source identities")
    if not _meaningful_frame_ids(record.get("frame_ids")):
        blockers.append("frame_ids must identify a positive native span or timestamp list")
    if not _nonempty_str(record.get("operating_point_id")):
        blockers.append("operating_point_id must be a meaningful identity")
    if not isinstance(controls, dict):
        blockers.append("controls must be an object with typed statuses")
    else:
        for field in REQUIRED_CONTROL_FIELDS:
            if field not in controls:
                blockers.append(f"controls missing {field}")
            elif not _control_ok(controls[field]):
                blockers.append(f"controls.{field} must be a known status or {{status: ...}}")

    if not _class_allows_purpose(str(record_class), purpose):
        blockers.append(f"record_class {record_class} is not eligible for {purpose} ingest")

    if elig.get("runtime") is True:
        blockers.extend(_timing_verified(record))
    if elig.get("rd") is True and purpose == "validated":
        blockers.extend(_validated_rd_blockers(record))
    if elig.get("generalization") is True:
        if purpose == "validated" and record_class != "validated_claim":
            blockers.append("generalization requires validated independent-source evidence")
        frozen = record.get("frozen_procedure")
        independent = record.get("independent_match_ids")
        if not frozen:
            blockers.append("generalization requires frozen_procedure evidence")
        if not isinstance(independent, list) or len(independent) < 2:
            blockers.append("generalization requires at least two independent_match_ids")
    if elig.get("trajectory") is True:
        coverage = record.get("trajectory_coverage")
        if coverage != "full_visible_track":
            blockers.append("trajectory claim requires trajectory_coverage=full_visible_track")
    return blockers


def ingest_for_claim(
    records: Iterable[Mapping[str, Any]],
    claim: str,
    *,
    purpose: Purpose = "validated",
) -> dict[str, Any]:
    """Keep eligible rows; every drop carries an explicit reason.

    Default ``purpose='validated'`` is production plot ingest. Diagnostic
    plots should pass ``purpose='diagnostic'`` so historical rows stay
    labelled rather than entering validated RD curves. Structural
    ``validate_campaign_record`` alone never certifies a claim.
    """
    if claim not in CLAIM_SCOPES:
        raise ValueError(f"unknown claim {claim!r}; expected one of {CLAIM_SCOPES}")
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        artifact_id = str(record.get("artifact_id") or "")
        structure = validate_campaign_record(record, purpose=purpose)
        if structure:
            excluded.append({"artifact_id": artifact_id, "reason": "; ".join(structure)})
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
    return {
        "claim": claim,
        "purpose": purpose,
        "kept": kept,
        "excluded": excluded,
        "n_kept": len(kept),
        "n_excluded": len(excluded),
    }


def campaign_record_from_generation_adapter(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Translate E02 adapter output into campaign_result.v1 without editing E02.

    ``rd_claim`` maps to ``rd``. ``speed_claim`` maps to ``runtime``.
    ``temporal_continuity`` is not ``generalization`` and is not automatically
    ``trajectory``. Single-scene readiness does not require generalization.
    """
    if raw.get("schema") == SCHEMA_ID:
        return dict(raw)

    ckpt = raw.get("checkpoint_identity") or {}
    metrics = raw.get("metrics") or {}
    timing = raw.get("timing_evidence") or {}
    adapter_elig = raw.get("claim_eligibility") or {}
    controls_in = raw.get("controls") or {}
    exclusions_in = list(raw.get("exclusion_reasons") or [])

    source_ids = raw.get("source_ids")
    if not isinstance(source_ids, list):
        clip = raw.get("clip_id") or raw.get("scene_id")
        source_ids = [clip] if _nonempty_str(clip) else []

    frame_ids = raw.get("frame_ids")
    if not isinstance(frame_ids, dict):
        count = raw.get("frame_count")
        frame_ids = {"start": 0, "count": count} if isinstance(count, int) and count > 0 else {}

    rd_requested = bool(adapter_elig.get("rd_claim") or adapter_elig.get("rd"))
    runtime_requested = bool(adapter_elig.get("speed_claim") or adapter_elig.get("runtime"))
    shuffled_tested = bool(controls_in.get("conditioned_vs_shuffled_tested"))
    deterministic = bool(controls_in.get("same_seed_determinism_tested"))
    defaulted = controls_in.get("conditioned_vs_shuffled_tested", True) is True and not shuffled_tested

    bytes_total = metrics.get("total_bytes", metrics.get("residual_bytes"))
    quality = None
    for key in ("psnr_mean", "vmaf_mean", "ssim_mean"):
        if _finite_number(metrics.get(key)):
            quality = metrics[key]
            break
    rd_ok = (
        rd_requested
        and shuffled_tested
        and deterministic
        and not defaulted
        and _finite_number(bytes_total)
        and quality is not None
    )
    runtime_ok = (
        runtime_requested
        and _nonempty_str(timing.get("timing_evidence_id"))
        and _nonempty_str(timing.get("host") or timing.get("profiling_strata"))
        and str(timing.get("host") or "") != "unassigned"
        and not isinstance(timing.get("n_repeats"), bool)
        and isinstance(timing.get("n_repeats"), int)
        and int(timing["n_repeats"]) >= 1
        and _finite_number(timing.get("measured_client_seconds"))
    )
    trajectory_ok = raw.get("trajectory_coverage") == "full_visible_track"
    independent = raw.get("independent_match_ids")
    generalization_ok = bool(raw.get("frozen_procedure")) and isinstance(independent, list) and len(independent) >= 2

    exclusions: list[dict[str, str]] = []
    for reason in exclusions_in:
        if isinstance(reason, str) and reason:
            exclusions.append({"claim": "rd", "reason": reason})
    if not rd_ok:
        exclusions.append(
            {
                "claim": "rd",
                "reason": "E02 adapter row lacks verified controls and finite rate/quality evidence",
            }
        )
    if not runtime_ok:
        exclusions.append({"claim": "runtime", "reason": "adapter timing is missing a named measured stratum"})
    if not bool(adapter_elig.get("standalone_decode")):
        exclusions.append({"claim": "standalone_transport", "reason": "adapter did not prove standalone decode"})
    if not trajectory_ok:
        exclusions.append(
            {
                "claim": "trajectory",
                "reason": "full-trajectory coverage is not automatic; set trajectory_coverage=full_visible_track",
            }
        )
    if not generalization_ok:
        exclusions.append(
            {
                "claim": "generalization",
                "reason": "independent frozen-procedure sources required; single-scene trajectory is not generalization",
            }
        )

    record_class = "recoverable_evidence" if rd_ok else "pending_run"
    sha = ckpt.get("checkpoint_sha256")
    artifact_sha = sha if isinstance(sha, str) and _SHA256_RE.fullmatch(sha) else _ZERO_SHA256
    run_id = str(raw.get("run_id") or "")
    return {
        "schema": SCHEMA_ID,
        "contract_revision": CONTRACT_REVISION,
        "record_class": record_class,
        "artifact_id": run_id or "missing_generation_run_id",
        "artifact_path": str(raw.get("artifact_path") or f"generation/{run_id or 'unspecified'}"),
        "artifact_sha256": artifact_sha,
        "code_revision": str(raw.get("code_revision") or ckpt.get("config_identity") or "unspecified"),
        "source_ids": source_ids,
        "frame_ids": frame_ids,
        "operating_point_id": str(raw.get("operating_point_id") or "display_low"),
        "independent_match_ids": independent if isinstance(independent, list) else [],
        "frozen_procedure": bool(raw.get("frozen_procedure")),
        "trajectory_coverage": raw.get("trajectory_coverage") or "unspecified",
        "claim_eligibility": {
            "rd": rd_ok,
            "runtime": runtime_ok,
            "standalone_transport": bool(adapter_elig.get("standalone_decode")) and rd_ok,
            "trajectory": trajectory_ok,
            "generalization": generalization_ok,
            "rd_arms": {
                "psnr_y": _finite_number(metrics.get("psnr_mean")),
                "ssim": _finite_number(metrics.get("ssim_mean")),
                "vmaf": _finite_number(metrics.get("vmaf_mean")),
                "bytes": _finite_number(bytes_total),
            },
            "exclusions": exclusions,
        },
        "timing_evidence": {
            "timing_evidence_id": timing.get("timing_evidence_id"),
            "host": timing.get("profiling_strata") or timing.get("host"),
            "n_repeats": timing.get("n_repeats") or 0,
            "measured_client_seconds": timing.get("measured_client_seconds"),
        },
        "controls": {
            "standalone_decode": "verified" if adapter_elig.get("standalone_decode") and rd_ok else "unverified",
            "metric_calibration": "unverified",
            "wire_ledger": "unverified",
            "conditioned_vs_shuffled": "verified" if shuffled_tested else "unverified",
        },
        "evidence": {
            "metrics": {
                "psnr_y": metrics.get("psnr_mean"),
                "ssim": metrics.get("ssim_mean"),
                "vmaf": metrics.get("vmaf_mean"),
            },
            "bytes": {"total": bytes_total, "residual": metrics.get("residual_bytes")},
        },
        "producer": "e02_generation_adapter",
        "backend_name": raw.get("backend_name"),
        "checkpoint_identity": ckpt,
    }


def load_example_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the checked-in example records used by tests and E02/E03 adapters."""
    payload = _as_dict(path or EXAMPLE_RECORDS_PATH)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("example contract must contain records[]")
    return [dict(item) for item in records]


def load_producer_example_records(path: Path | None = None) -> list[dict[str, Any]]:
    payload = _as_dict(path or PRODUCER_EXAMPLE_PATH)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("producer example must contain records[]")
    return [dict(item) for item in records]
