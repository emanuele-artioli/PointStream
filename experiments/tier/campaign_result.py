"""Shared campaign result contract: claim eligibility, timing refs, plot ingest.

E03/E08 own the plots. This module validates records and emits exclusion
reasons. Structural checks are not claim eligibility. Production ingest also
checks identities, provenance class, finite evidence, and control types.

A missing timing stratum never silently drops an otherwise valid RD row.
An unrun or placeholder timing row never certifies runtime.
Full-trajectory coverage is claim ``trajectory``, not ``generalization``.
"""

from __future__ import annotations

import hashlib
import json
import math
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
CONTRACT_REVISION = "e03a-20260915"
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
_SUCCESS_BY_CONTROL: Final[dict[str, frozenset[str]]] = {
    "standalone_decode": frozenset({"verified"}),
    "metric_calibration": frozenset({"verified", "inherited_named_artifact"}),
    "wire_ledger": frozenset({"verified", "reconciled"}),
}
_SSIM_NAMES = frozenset({"ssim", "ssim_mean"})
_VMAF_NAMES = frozenset({"vmaf", "vmaf_mean"})
_PSNR_NAMES = frozenset({"psnr_y", "psnr_mean", "psnr"})
_KNOWN_QUALITY_NAMES = _SSIM_NAMES | _VMAF_NAMES | _PSNR_NAMES

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


def _control_status(value: Any) -> str | None:
    if isinstance(value, dict):
        status = value.get("status")
    else:
        status = value
    return status if isinstance(status, str) else None


def _control_ok(value: Any) -> bool:
    status = _control_status(value)
    return status in CONTROL_STATUSES


def _control_success(value: Any, field: str) -> bool:
    status = _control_status(value)
    allowed = _SUCCESS_BY_CONTROL.get(field, frozenset())
    return status in allowed


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_rate(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _quality_ok(name: str, value: Any) -> bool:
    if not _finite_number(value):
        return False
    key = name.lower()
    number = float(value)
    if key in _SSIM_NAMES:
        return 0.0 <= number <= 1.0
    if key in _VMAF_NAMES:
        return 0.0 <= number <= 100.0
    if key in _PSNR_NAMES:
        return number > 0.0
    return True


def _canonical_quality_name(name: str) -> str | None:
    key = name.lower()
    if key in _PSNR_NAMES:
        return "psnr_y"
    if key in _SSIM_NAMES:
        return "ssim"
    if key in _VMAF_NAMES:
        return "vmaf"
    return None


def _code_revision_str(value: Any) -> str:
    if _nonempty_str(value):
        return str(value).strip()
    if isinstance(value, dict):
        commit = value.get("commit")
        if _nonempty_str(commit):
            suffix = ""
            if value.get("dirty"):
                digest = value.get("diff_sha256")
                suffix = f"+dirty:{digest}" if _nonempty_str(digest) else "+dirty"
            return f"{commit}{suffix}"
    return ""


def _evidence_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    metrics = evidence.get("metrics") if isinstance(evidence, dict) else None
    if isinstance(metrics, dict) and metrics:
        return dict(metrics)
    raw_metrics = record.get("metrics")
    if isinstance(raw_metrics, dict):
        return dict(raw_metrics)
    scores = record.get("scores")
    if isinstance(scores, dict):
        return dict(scores)
    return {}


def _evidence_bytes_total(record: Mapping[str, Any]) -> Any:
    evidence = record.get("evidence") or {}
    if isinstance(evidence, dict):
        bytes_block = evidence.get("bytes") or {}
        if isinstance(bytes_block, dict) and bytes_block.get("total") is not None:
            return bytes_block.get("total")
    raw_metrics = record.get("metrics")
    metrics: dict[str, Any] = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
    if metrics.get("total_bytes") is not None:
        return metrics.get("total_bytes")
    raw_parts = record.get("parts")
    parts: dict[str, Any] = dict(raw_parts) if isinstance(raw_parts, dict) else {}
    if parts.get("transport_total") is not None:
        return parts.get("transport_total")
    return record.get("coded_bytes")


def _rd_arms(record: Mapping[str, Any]) -> dict[str, bool]:
    """Arm presence from measured values, not caller flags."""
    metrics = _evidence_metrics(record)
    derived: dict[str, bool] = {}
    for name, raw in metrics.items():
        key = str(name)
        canonical = _canonical_quality_name(key)
        if canonical is not None:
            derived[canonical] = derived.get(canonical, False) or _quality_ok(key, raw)
    derived["bytes"] = _positive_rate(_evidence_bytes_total(record))
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
    measured = ref.get("measured_client_seconds")
    if not _positive_rate(measured):
        blockers.append("runtime eligibility requires positive finite measured_client_seconds")
    encoder = ref.get("encoder_seconds")
    if encoder is not None and not _positive_rate(encoder):
        blockers.append("runtime encoder_seconds, when present, must be positive and finite")
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
        blockers.append("validated RD requires a positive finite byte/rate arm")
    quality_arms = [name for name in salvageable if name != "bytes"]
    if not quality_arms:
        blockers.append("validated RD requires at least one domain-valid quality metric arm")
    declared = (record.get("claim_eligibility") or {}).get("rd_arms")
    if not isinstance(declared, dict):
        blockers.append("validated RD requires an explicit rd_arms object")
    else:
        unknown = sorted(
            str(name)
            for name in declared
            if _canonical_quality_name(str(name)) is None and name != "bytes"
        )
        if unknown:
            blockers.append(f"validated RD rd_arms contains unknown metrics: {unknown}")
        normalized_declared: dict[str, bool] = {}
        for raw_name, flag in declared.items():
            name = "bytes" if raw_name == "bytes" else _canonical_quality_name(str(raw_name))
            if name is None:
                continue
            if not isinstance(flag, bool):
                blockers.append(f"validated RD arm {raw_name} must be an explicit bool")
                continue
            if name in normalized_declared and normalized_declared[name] is not flag:
                blockers.append(f"validated RD arm aliases for {name} disagree")
            normalized_declared[name] = flag
        if normalized_declared.get("bytes") is not True:
            blockers.append("validated RD requires explicit rd_arms.bytes=true")
        elif not arms.get("bytes"):
            blockers.append("validated RD arm bytes=true lacks positive finite evidence")
        quality_declared = [
            name for name, flag in normalized_declared.items() if name != "bytes" and flag
        ]
        if not quality_declared:
            blockers.append("validated RD requires at least one known quality arm=true")
        for name in quality_declared:
            if not arms.get(name):
                blockers.append(
                    f"validated RD arm {name}=true lacks matching domain-valid evidence"
                )
    controls = record.get("controls") or {}
    if not isinstance(controls, dict):
        blockers.append("validated RD requires typed control evidence")
        return blockers
    if not _control_success(controls.get("wire_ledger"), "wire_ledger"):
        blockers.append("validated RD requires a verified or reconciled wire ledger")
    if not _control_success(controls.get("metric_calibration"), "metric_calibration"):
        blockers.append("validated RD requires verified or inherited metric calibration")
    if not _control_success(controls.get("standalone_decode"), "standalone_decode"):
        blockers.append("validated RD requires verified standalone decode of scored output")
    return blockers


def _validated_standalone_blockers(record: Mapping[str, Any]) -> list[str]:
    controls = record.get("controls") or {}
    if not _control_success(
        controls.get("standalone_decode") if isinstance(controls, dict) else None,
        "standalone_decode",
    ):
        return ["standalone_transport requires verified standalone decode"]
    blockers: list[str] = []
    if not _control_success(controls.get("wire_ledger"), "wire_ledger"):
        blockers.append("standalone_transport requires a verified or reconciled wire ledger")
    if not _positive_rate(_evidence_bytes_total(record)):
        blockers.append("standalone_transport requires positive finite transport bytes")
    return blockers


def _validated_trajectory_blockers(record: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if record.get("trajectory_coverage") != "full_visible_track":
        blockers.append("trajectory claim requires trajectory_coverage=full_visible_track")
    frame_ids = record.get("frame_ids")
    if not isinstance(frame_ids, dict) or not isinstance(frame_ids.get("count"), int) or frame_ids["count"] < 2:
        blockers.append("trajectory claim requires at least two identified frames")
    controls = record.get("controls") or {}
    conditioned = controls.get("conditioned_vs_shuffled") if isinstance(controls, dict) else None
    if _control_status(conditioned) != "verified":
        blockers.append("trajectory claim requires a verified conditioned-vs-shuffled control")
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
    if not _code_revision_str(record.get("code_revision")):
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
    if elig.get("standalone_transport") is True and purpose == "validated":
        blockers.extend(_validated_standalone_blockers(record))
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
        blockers.extend(_validated_trajectory_blockers(record))
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


def _adapter_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(raw.get("metrics") or {})
    scores = raw.get("scores")
    if isinstance(scores, dict):
        if metrics.get("psnr_mean") is None:
            metrics["psnr_mean"] = scores.get("psnr_y", scores.get("psnr_mean"))
        if metrics.get("ssim_mean") is None:
            metrics["ssim_mean"] = scores.get("ssim", scores.get("ssim_mean"))
        if metrics.get("vmaf_mean") is None:
            metrics["vmaf_mean"] = scores.get("vmaf", scores.get("vmaf_mean"))
    raw_parts = raw.get("parts")
    parts: dict[str, Any] = dict(raw_parts) if isinstance(raw_parts, dict) else {}
    if metrics.get("total_bytes") is None:
        metrics["total_bytes"] = parts.get("transport_total", raw.get("coded_bytes"))
    if metrics.get("residual_bytes") is None:
        metrics["residual_bytes"] = parts.get("residual")
    return metrics


def _adapter_timing(raw: Mapping[str, Any]) -> dict[str, Any]:
    timing = raw.get("timing_evidence")
    if not isinstance(timing, dict):
        timing = {}
    nested = raw.get("timing")
    if isinstance(nested, dict):
        if timing.get("measured_client_seconds") is None:
            timing["measured_client_seconds"] = nested.get("client_seconds")
        if timing.get("encoder_seconds") is None:
            timing["encoder_seconds"] = nested.get("encoder_seconds")
    return timing


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(raw: Mapping[str, Any]) -> str:
    declared = raw.get("artifact_sha256")
    path_value = raw.get("artifact_path")
    path = Path(str(path_value)) if _nonempty_str(path_value) else None
    if path is not None and path.is_file():
        return _sha256_file(path)
    if isinstance(declared, str) and _SHA256_RE.fullmatch(declared):
        return declared
    return _ZERO_SHA256


def _map_campaign_control(value: Any, field: str) -> str:
    status = _control_status(value)
    if status in CONTROL_STATUSES:
        if status in _SUCCESS_BY_CONTROL.get(field, frozenset()) or status in {
            "unverified",
            "unverified_on_this_artifact",
            "failed",
            "not_this_row",
            "not_applicable",
            "partial",
        }:
            return status
        return "unverified"
    return "unverified"


def _wire_status_from_producer(raw: Mapping[str, Any], controls_in: Mapping[str, Any]) -> str:
    mapped = _map_campaign_control(controls_in.get("wire_ledger"), "wire_ledger")
    if mapped in _SUCCESS_BY_CONTROL["wire_ledger"]:
        return mapped
    recon = raw.get("wire_reconciliation")
    if isinstance(recon, dict) and recon.get("matched") is True and recon.get("verdict") == "matched":
        wire = recon.get("wire_bytes")
        total = recon.get("transport_total")
        if _positive_rate(wire) and wire == total:
            return "reconciled"
    return mapped if mapped in CONTROL_STATUSES else "unverified"


def campaign_record_from_generation_adapter(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Translate producer-shaped generation output into campaign_result.v1.

    Translation lives here so ``src.runner`` does not import ``experiments``.
    Current-revision records pass through. Older adapter JSON is remapped.
    """
    if (
        raw.get("schema") == SCHEMA_ID
        and raw.get("contract_revision") == CONTRACT_REVISION
        and raw.get("record_class") in RECORD_CLASSES
    ):
        return dict(raw)

    ckpt = raw.get("checkpoint_identity") or {}
    if not isinstance(ckpt, dict):
        ckpt = {}
    metrics = _adapter_metrics(raw)
    timing = _adapter_timing(raw)
    adapter_elig = raw.get("claim_eligibility") or {}
    if not isinstance(adapter_elig, dict):
        adapter_elig = {}
    controls_in = raw.get("controls") or {}
    if not isinstance(controls_in, dict):
        controls_in = {}
    gen_controls = raw.get("generator_controls") or {}
    if isinstance(gen_controls, dict):
        for key in (
            "conditioned_vs_shuffled_tested",
            "same_seed_determinism_tested",
            "conditioning_sensitive",
            "same_seed_deterministic",
        ):
            controls_in.setdefault(key, gen_controls.get(key))
    exclusions_in = list(raw.get("exclusion_reasons") or [])
    if isinstance(adapter_elig.get("exclusions"), list):
        exclusions_in.extend(adapter_elig["exclusions"])

    source_ids = raw.get("source_ids")
    if not isinstance(source_ids, list):
        video = raw.get("video")
        scene = raw.get("scene")
        if _nonempty_str(video) and _nonempty_str(scene):
            source_ids = [f"{video}_{scene}"]
        else:
            clip = raw.get("clip_id") or raw.get("scene_id")
            source_ids = [clip] if _nonempty_str(clip) else []

    frame_ids = raw.get("frame_ids")
    if not isinstance(frame_ids, dict):
        count = raw.get("frame_count") or raw.get("frames")
        fps = raw.get("fps")
        frame_ids = {}
        if isinstance(count, int) and count > 0:
            frame_ids["start"] = 0
            frame_ids["count"] = count
        if (
            isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and math.isfinite(fps)
            and float(fps) > 0
        ):
            frame_ids["fps"] = float(fps)

    shape = raw.get("delivered_shape") or raw.get("operating_resolution")
    operating_point = raw.get("operating_point_id")
    if not _nonempty_str(operating_point):
        operating_point = "display_native"

    bytes_total = metrics.get("total_bytes")
    psnr = metrics.get("psnr_mean")
    ssim = metrics.get("ssim_mean")
    vmaf = metrics.get("vmaf_mean")
    rd_measured = _positive_rate(bytes_total) and (
        _quality_ok("psnr_y", psnr) or _quality_ok("ssim", ssim) or _quality_ok("vmaf", vmaf)
    )

    decode_status = _map_campaign_control(controls_in.get("standalone_decode"), "standalone_decode")
    calib_status = _map_campaign_control(
        controls_in.get("metric_calibration"), "metric_calibration"
    )
    ledger_status = _wire_status_from_producer(raw, controls_in)
    decode_ok = _control_success(decode_status, "standalone_decode")
    calib_ok = _control_success(calib_status, "metric_calibration")
    ledger_ok = _control_success(ledger_status, "wire_ledger")
    blank_tested = bool(controls_in.get("no_conditioning_tested"))
    generation_on = bool(raw.get("generation_on", raw.get("backend_name")))
    model_free = bool(raw.get("model_free"))
    raw_deployment = raw.get("deployment")
    deployment: dict[str, Any] = dict(raw_deployment) if isinstance(raw_deployment, dict) else {}
    if model_free:
        deployment_ok = True
        blank_ok = True
    else:
        mode = str(deployment.get("mode") or "")
        if mode == "shared":
            deployment_ok = bool(deployment.get("receiver_availability")) and bool(
                deployment.get("amortization_policy")
            ) and deployment.get("storage_bytes") is not None
        elif mode == "per_video":
            charged = deployment.get("charged_bytes", deployment.get("storage_bytes"))
            deployment_ok = _positive_rate(charged)
        else:
            deployment_ok = False
        blank_ok = (not generation_on) or blank_tested
    rd_ok = bool(rd_measured and decode_ok and calib_ok and ledger_ok and deployment_ok and blank_ok)

    n_repeats = timing.get("n_repeats", timing.get("sample_count"))
    host = timing.get("host") or timing.get("profiling_strata")
    runtime_ok = (
        _nonempty_str(timing.get("timing_evidence_id"))
        and _nonempty_str(host)
        and str(host) != "unassigned"
        and "not-yet-run" not in str(timing.get("timing_evidence_id"))
        and isinstance(n_repeats, int)
        and not isinstance(n_repeats, bool)
        and n_repeats >= 1
        and _finite_number(timing.get("measured_client_seconds"))
        and float(timing["measured_client_seconds"]) > 0
    )
    conditioning_ok = bool(
        controls_in.get("conditioned_vs_shuffled_tested")
        and controls_in.get("conditioning_sensitive")
    )
    transport_ok = decode_ok and ledger_ok and _positive_rate(bytes_total)
    trajectory_ok = (
        raw.get("trajectory_coverage") == "full_visible_track"
        and isinstance(frame_ids.get("count"), int)
        and frame_ids["count"] >= 2
        and conditioning_ok
    )
    independent = raw.get("independent_match_ids")
    generalization_ok = (
        bool(raw.get("frozen_procedure")) and isinstance(independent, list) and len(independent) >= 2
    )

    exclusions: list[dict[str, str]] = []
    for reason in exclusions_in:
        if isinstance(reason, str) and reason:
            exclusions.append({"claim": "rd", "reason": reason})
        elif isinstance(reason, dict) and reason.get("claim") and reason.get("reason"):
            exclusions.append({"claim": str(reason["claim"]), "reason": str(reason["reason"])})
    if not deployment_ok:
        exclusions.append(
            {
                "claim": "rd",
                "reason": "undeclared model deployment cost; checkpoint digest is not delivered weights",
            }
        )
    if not blank_ok:
        exclusions.append(
            {"claim": "rd", "reason": "missing blank/no-conditioning control"}
        )
    if rd_measured and not (decode_ok and calib_ok and ledger_ok):
        exclusions.append(
            {
                "claim": "rd",
                "reason": "measured rate/quality present; decode/calibration/ledger not verified",
            }
        )
    if not rd_measured:
        exclusions.append(
            {
                "claim": "rd",
                "reason": "adapter row lacks domain-valid rate and quality evidence",
            }
        )
    if not runtime_ok:
        exclusions.append(
            {"claim": "runtime", "reason": "adapter timing is missing a named measured stratum"}
        )
    if not transport_ok:
        exclusions.append(
            {
                "claim": "standalone_transport",
                "reason": "standalone decode, positive bytes and reconciled ledger are required",
            }
        )
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

    record_class = "recoverable_evidence" if rd_measured else "historical_observation"
    artifact_path = str(raw.get("artifact_path") or "")
    run_id = str(raw.get("run_id") or raw.get("artifact_id") or "")
    return {
        "schema": SCHEMA_ID,
        "contract_revision": CONTRACT_REVISION,
        "record_class": record_class,
        "artifact_id": run_id or "missing_generation_run_id",
        "artifact_path": artifact_path or f"generation/{run_id or 'unspecified'}",
        "artifact_sha256": _artifact_sha256(raw),
        "code_revision": _code_revision_str(raw.get("code_revision"))
        or _code_revision_str(ckpt.get("config_identity"))
        or "unspecified",
        "source_ids": source_ids,
        "frame_ids": frame_ids,
        "operating_point_id": str(operating_point),
        "independent_match_ids": independent if isinstance(independent, list) else [],
        "frozen_procedure": bool(raw.get("frozen_procedure")),
        "trajectory_coverage": raw.get("trajectory_coverage") or "unspecified",
        "delivered_shape": shape,
        "claim_eligibility": {
            "rd": rd_ok,
            "runtime": runtime_ok,
            "standalone_transport": transport_ok,
            "trajectory": trajectory_ok,
            "generalization": generalization_ok,
            "rd_arms": {
                "psnr_y": _quality_ok("psnr_y", psnr),
                "ssim": _quality_ok("ssim", ssim),
                "vmaf": _quality_ok("vmaf", vmaf),
                "bytes": _positive_rate(bytes_total),
            },
            "exclusions": exclusions,
        },
        "timing_evidence": {
            "timing_evidence_id": timing.get("timing_evidence_id"),
            "host": host,
            "n_repeats": n_repeats if isinstance(n_repeats, int) else None,
            "measured_client_seconds": timing.get("measured_client_seconds"),
            "encoder_seconds": timing.get("encoder_seconds"),
        },
        "controls": {
            "standalone_decode": decode_status,
            "metric_calibration": calib_status,
            "wire_ledger": ledger_status,
            "conditioned_vs_shuffled": (
                "verified" if conditioning_ok else "unverified"
            ),
        },
        "evidence": {
            "metrics": {"psnr_y": psnr, "ssim": ssim, "vmaf": vmaf},
            "bytes": {"total": bytes_total, "residual": metrics.get("residual_bytes")},
            "parts": dict(raw["parts"]) if isinstance(raw.get("parts"), dict) else {},
        },
        "producer": "e02_generation_adapter",
        "backend_name": raw.get("backend_name"),
        "checkpoint_identity": ckpt,
        "note": "checkpoint digest is identity, not artifact_sha256",
    }


def campaign_record_from_diagnostic_matrix_file(
    path: Path,
    *,
    run_id: str,
    backend_name: str,
    arch: str,
) -> dict[str, Any]:
    """Build a current-revision record from an immutable diagnostic JSON file."""
    from src.runner.generation_adapter import adapt_diagnostic_matrix_file

    adapted = adapt_diagnostic_matrix_file(
        path,
        run_id=run_id,
        backend_name=backend_name,
        arch=arch,
    )
    return campaign_record_from_generation_adapter(adapted)


def write_campaign_record(record: Mapping[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(record), indent=2) + "\n", encoding="utf-8")
    return dest


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
