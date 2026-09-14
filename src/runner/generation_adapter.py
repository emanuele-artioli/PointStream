"""Adapter connecting generation run evaluations to the E01 frozen experiment schema.

Provides fail-closed translation, claim-eligibility tagging, timing-evidence
references, control verification (conditioning vs null/shuffled), and source/clip
uncertainty accounting. Conforms to pointstream.campaign_result.v1.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final

import numpy as np

from src.runner.generation_identity import IDENTITY_KEYS, config_identity_digest

SCHEMA_ID: Final[str] = "pointstream.campaign_result.v1"

CLAIM_SCOPES: Final[tuple[str, ...]] = (
    "rd",
    "runtime",
    "standalone_transport",
    "generalization",
)

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

SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{64}$")


def is_valid_sha256_digest(digest: Any) -> bool:
    """Check if digest is a genuine 64-character hexadecimal SHA-256 string."""
    if not isinstance(digest, str):
        return False
    return bool(SHA256_HEX_RE.fullmatch(digest.strip()))


@dataclass
class GenerationResultRecord:
    """Standardized generation evaluation result conforming to E01 protocol requirements."""

    schema: str = SCHEMA_ID
    artifact_id: str = ""
    artifact_path: str = ""
    artifact_sha256: str = ""
    code_revision: str = ""
    source_ids: list[str] = field(default_factory=list)
    frame_ids: dict[str, Any] = field(default_factory=dict)
    operating_point_id: str = "display_native"
    claim_eligibility: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)
    timing_evidence: dict[str, Any] | None = None

    # Extended metadata fields for evaluation and diagnostics
    doc_role: str = "generation_result"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    backend_name: str = ""
    arch: str = ""
    condition_type: str = "pose"
    checkpoint_identity: dict[str, Any] = field(default_factory=dict)
    operating_resolution: tuple[int, int] = (1080, 1920)
    crop_resolution: tuple[int, int] = (256, 256)
    frame_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    generator_controls: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metric_uncertainty(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute sample count, mean, sample std, standard error (SEM), and 95% CI.

    Explicitly marks single-sample (n=1) as uncertainty unavailable rather than
    fabricating zero-width certainty.
    """
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci_95": None,
            "uncertainty_status": "no_samples",
        }
    mean_val = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean_val, 4),
            "std": None,
            "sem": None,
            "ci_95": None,
            "uncertainty_status": "single_source_uncertainty_unavailable",
        }
    std_val = float(np.std(arr, ddof=1))
    sem_val = float(std_val / np.sqrt(n))
    return {
        "n": n,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "sem": round(sem_val, 4),
        "ci_95": [round(mean_val - 1.96 * sem_val, 4), round(mean_val + 1.96 * sem_val, 4)],
        "uncertainty_status": "grouped_uncertainty_estimated",
    }


def validate_generation_result(record: dict[str, Any] | GenerationResultRecord) -> tuple[bool, list[str]]:
    """Validate a generation result fail-closed against campaign protocol requirements."""
    data = record.to_dict() if isinstance(record, GenerationResultRecord) else record
    blockers: list[str] = []

    # 1. Top-level E01 schema validation
    if data.get("schema") != SCHEMA_ID:
        blockers.append(f"schema must be {SCHEMA_ID}")
    for field_name in REQUIRED_RECORD_FIELDS:
        if field_name not in data:
            blockers.append(f"missing required field: {field_name}")

    if not is_valid_sha256_digest(data.get("artifact_sha256")):
        blockers.append("artifact_sha256 must be a valid 64-character hexadecimal SHA-256 digest")

    if not data.get("code_revision"):
        blockers.append("code_revision must be non-empty")

    if not data.get("source_ids") or not isinstance(data.get("source_ids"), list):
        blockers.append("source_ids must be a non-empty list")

    # 2. Controls structure
    controls = data.get("controls")
    if not isinstance(controls, dict):
        blockers.append("controls must be a dictionary")
    else:
        for ctrl_field in REQUIRED_CONTROL_FIELDS:
            if ctrl_field not in controls:
                blockers.append(f"controls missing required field: {ctrl_field}")

    # 3. Checkpoint Identity validation
    ckpt_id = data.get("checkpoint_identity") or {}
    for key in IDENTITY_KEYS:
        if not ckpt_id.get(key):
            blockers.append(f"missing required checkpoint identity key: {key}")
    if not is_valid_sha256_digest(ckpt_id.get("checkpoint_sha256")):
        blockers.append("checkpoint_identity.checkpoint_sha256 must be a valid 64-character SHA-256 digest")

    # 4. Generator specific controls
    gen_controls = data.get("generator_controls") or {}
    if not gen_controls.get("conditioned_vs_shuffled_tested", False):
        blockers.append("missing conditioning sensitivity control (conditioned vs shuffled)")
    if not gen_controls.get("same_seed_determinism_tested", False):
        blockers.append("missing same-seed determinism control")

    # 5. Claims eligibility consistency
    elig = data.get("claim_eligibility")
    if not isinstance(elig, dict):
        blockers.append("claim_eligibility must be a dictionary")
        return len(blockers) == 0, blockers

    exclusions = elig.get("exclusions")
    if exclusions is None or not isinstance(exclusions, list):
        blockers.append("claim_eligibility.exclusions must be a list")
        exclusions = []

    metrics = data.get("metrics") or {}
    timing = data.get("timing_evidence") or {}

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
            if not isinstance(timing, dict) or not timing.get("timing_evidence_id"):
                blockers.append("runtime eligibility requires timing_evidence.timing_evidence_id")

    if elig.get("rd") is True:
        if metrics.get("residual_bytes") is None and metrics.get("total_bytes") is None:
            blockers.append("rd claim requires measured wire rate (residual_bytes or total_bytes)")
        if metrics.get("psnr_mean") is None and metrics.get("ssim_mean") is None:
            blockers.append("rd claim requires measured objective quality (psnr_mean or ssim_mean)")
        if not gen_controls.get("conditioning_sensitive", False):
            blockers.append("rd claim requires verified conditioning sensitivity")
        if not gen_controls.get("same_seed_deterministic", False):
            blockers.append("rd claim requires verified same-seed determinism")

    return len(blockers) == 0, blockers


def adapt_diagnostic_matrix_result(
    matrix_output: dict[str, Any],
    *,
    run_id: str,
    backend_name: str,
    arch: str,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    operating_resolution: tuple[int, int] = (1080, 1920),
    crop_resolution: tuple[int, int] = (256, 256),
    host_strata: str = "shared_gpu_server",
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
    code_revision: str | None = None,
    operating_point_id: str = "display_native",
    timing_evidence_id: str | None = None,
) -> dict[str, Any]:
    """Adapt diagnostic matrix run output into the standardized E01 result schema."""
    # Handle either producer matrix format ({'matrix': [...]}) or legacy format ({'normal': ..., 'shuffled': ...})
    matrix_rows = matrix_output.get("matrix")
    normal_run: dict[str, Any] = {}
    shuffled_run: dict[str, Any] = {}
    no_cond_run: dict[str, Any] = {}
    seed_repeat: dict[str, Any] = {}
    paste_run: dict[str, Any] = {}

    if isinstance(matrix_rows, list):
        for row in matrix_rows:
            corner = str(row.get("corner", ""))
            is_gen = bool(row.get("generation_on", False))
            is_shuffled = bool(row.get("shuffled_conditioning", False))
            is_repeat = bool(row.get("seed_repeat", False) or corner.endswith("_repeat"))
            is_no_cond = bool(row.get("no_conditioning", False) or corner.endswith("_no_cond"))
            is_paste = bool(row.get("control") == "pasted_reference" or not is_gen)

            if is_paste and not paste_run:
                paste_run = row
            elif is_repeat and not seed_repeat:
                seed_repeat = row
            elif is_no_cond and not no_cond_run:
                no_cond_run = row
            elif is_shuffled and not shuffled_run:
                shuffled_run = row
            elif is_gen and not is_shuffled and not is_repeat and not is_no_cond and not normal_run:
                normal_run = row

        # If normal_run wasn't matched by flags, pick first generation_on row
        if not normal_run:
            for row in matrix_rows:
                if row.get("generation_on") and not row.get("shuffled_conditioning"):
                    normal_run = row
                    break
    else:
        normal_run = matrix_output.get("normal") or {}
        shuffled_run = matrix_output.get("shuffled") or {}
        no_cond_run = matrix_output.get("no_conditioning") or {}
        seed_repeat = matrix_output.get("seed_repeat") or {}
        paste_run = matrix_output.get("paste") or {}

    # Extract checkpoint identity and digest fail-closed
    ckpt_file = Path(str(checkpoint_path)) if checkpoint_path else None
    sha = checkpoint_sha256 or matrix_output.get("checkpoint_sha256") or normal_run.get("checkpoint_sha256")
    if not sha and ckpt_file and ckpt_file.is_file():
        sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

    valid_sha = is_valid_sha256_digest(sha)
    ckpt_sha_clean = str(sha).strip().lower() if valid_sha else ""

    ckpt_id_str = f"{ckpt_file.name if ckpt_file else backend_name}:{ckpt_sha_clean or 'unresolved'}"
    ckpt_identity = {
        "checkpoint_id": ckpt_id_str,
        "checkpoint_sha256": ckpt_sha_clean,
        "name": backend_name,
        "arch": arch,
        "seed": normal_run.get("seed", matrix_output.get("seed", 0)),
    }
    ckpt_identity["config_identity"] = config_identity_digest(ckpt_identity)

    # Frame hashes & controls evaluation
    normal_hashes = normal_run.get("delivered_frame_hashes") or normal_run.get("frame_hashes") or []
    shuffled_hashes = shuffled_run.get("delivered_frame_hashes") or shuffled_run.get("frame_hashes") or []
    repeat_hashes = seed_repeat.get("delivered_frame_hashes") or seed_repeat.get("frame_hashes") or []
    paste_hashes = paste_run.get("delivered_frame_hashes") or paste_run.get("frame_hashes") or []

    shuffled_tested = bool(shuffled_hashes)
    shuffled_differs = bool(normal_hashes and shuffled_hashes and normal_hashes != shuffled_hashes)

    repeat_tested = bool(repeat_hashes)
    repeat_identical = bool(normal_hashes and repeat_hashes and normal_hashes == repeat_hashes)

    delivered_changed = bool(normal_hashes and paste_hashes and normal_hashes != paste_hashes)
    if not paste_hashes and normal_run.get("delivered_pixels_changed") is not None:
        delivered_changed = bool(normal_run["delivered_pixels_changed"])

    gen_controls = {
        "conditioned_vs_shuffled_tested": shuffled_tested,
        "conditioning_sensitive": shuffled_differs,
        "same_seed_determinism_tested": repeat_tested,
        "same_seed_deterministic": repeat_identical,
        "no_conditioning_tested": bool(no_cond_run),
        "delivered_pixels_changed": delivered_changed,
    }

    # E01 standardized minimal controls
    controls = {
        "standalone_decode": "verified_client_envelope" if normal_hashes else "unverified",
        "metric_calibration": "calibrated_anchors_present" if normal_run.get("psnr_y") is not None or normal_run.get("psnr_mean") is not None else "unverified",
        "wire_ledger": "verified_reconciliation" if normal_run.get("coded_bytes") is not None or normal_run.get("total_bytes") is not None else "unverified",
    }

    # Timing
    normal_seconds = normal_run.get("elapsed_seconds")
    timing_id = timing_evidence_id or (f"timing_{run_id}_{host_strata}" if normal_seconds and normal_seconds > 0 else None)
    timing_evidence = (
        {
            "timing_evidence_id": timing_id,
            "profiling_strata": host_strata,
            "measured_client_seconds": normal_seconds,
            "warmup_seconds": normal_run.get("warmup_seconds", 0.0),
            "p95_latency_ms": normal_run.get("p95_latency_ms"),
        }
        if timing_id
        else None
    )

    # Metrics
    psnr_val = normal_run.get("psnr_y", normal_run.get("psnr_mean"))
    ssim_val = normal_run.get("ssim", normal_run.get("ssim_mean"))
    vmaf_val = normal_run.get("vmaf", normal_run.get("vmaf_mean"))
    temp_val = normal_run.get("temporal_loss", normal_run.get("temporal_error_mean"))
    res_bytes = normal_run.get("residual_bytes")
    tot_bytes = normal_run.get("coded_bytes", normal_run.get("total_bytes"))

    metrics = {
        "psnr_mean": psnr_val,
        "ssim_mean": ssim_val,
        "vmaf_mean": vmaf_val,
        "temporal_error_mean": temp_val,
        "residual_bytes": res_bytes,
        "total_bytes": tot_bytes,
    }

    # Uncertainty: source-grouped
    per_frame_psnr = normal_run.get("per_frame_psnr") or []
    uncertainty = {
        "psnr": calculate_metric_uncertainty(per_frame_psnr),
    }

    # Claim eligibility fail-closed
    exclusions: list[dict[str, str]] = []
    rd_eligible = True

    if tot_bytes is None and res_bytes is None:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing measured wire byte payload"})

    if psnr_val is None and ssim_val is None:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing measured objective quality"})

    if not valid_sha:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing or invalid checkpoint SHA-256 digest"})

    if not gen_controls["conditioned_vs_shuffled_tested"]:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing conditioning sensitivity control"})
    elif not gen_controls["conditioning_sensitive"]:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "failed conditioning sensitivity control (shuffled match)"})

    if not gen_controls["same_seed_determinism_tested"]:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing same-seed determinism control"})
    elif not gen_controls["same_seed_deterministic"]:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "failed same-seed determinism control"})

    if not gen_controls["delivered_pixels_changed"] and paste_hashes:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "generator produced identical pixels to pasted reference"})

    speed_eligible = bool(normal_seconds is not None and normal_seconds > 0 and timing_evidence is not None)
    if not speed_eligible:
        exclusions.append({"claim": "runtime", "reason": "timing missing or not a transferable host stratum"})

    standalone_eligible = bool(normal_hashes and len(normal_hashes) > 0)
    if not standalone_eligible:
        exclusions.append({"claim": "standalone_transport", "reason": "scoring path not proven equal to serialized client output"})

    # Single-scene diagnostic runs are not independent-source generalization evidence
    generalization_eligible = False
    exclusions.append({"claim": "generalization", "reason": "single development scene / not held-out confirmation split"})

    claim_eligibility = {
        "rd": rd_eligible,
        "runtime": speed_eligible,
        "standalone_transport": standalone_eligible,
        "generalization": generalization_eligible,
        "exclusions": exclusions,
    }

    # Artifact metadata
    art_path_str = str(artifact_path or f"outputs/diagnostic/{run_id}.json")
    if artifact_sha256 and is_valid_sha256_digest(artifact_sha256):
        art_sha = artifact_sha256
    else:
        # Canonical hash of matrix output
        canonical_bytes = json.dumps(matrix_output, sort_keys=True).encode("utf-8")
        art_sha = hashlib.sha256(canonical_bytes).hexdigest()

    rev = code_revision or matrix_output.get("identity", {}).get("code_revision") or "unspecified_revision"

    video_id = matrix_output.get("video") or "unknown_video"
    scene_id = matrix_output.get("scene") or "unknown_scene"
    source_ids = [f"{video_id}_{scene_id}"] if scene_id != "unknown_scene" else [video_id]

    frame_count = len(normal_hashes) if normal_hashes else int(matrix_output.get("frames", 0))
    frame_ids = {
        "start": 0,
        "count": frame_count,
        "fps": float(matrix_output.get("fps", 24.0)),
    }

    record = GenerationResultRecord(
        schema=SCHEMA_ID,
        artifact_id=f"generation_{backend_name}_{run_id}",
        artifact_path=art_path_str,
        artifact_sha256=art_sha,
        code_revision=rev,
        source_ids=source_ids,
        frame_ids=frame_ids,
        operating_point_id=operating_point_id,
        claim_eligibility=claim_eligibility,
        controls=controls,
        timing_evidence=timing_evidence,
        doc_role="generation_result",
        backend_name=backend_name,
        arch=arch,
        checkpoint_identity=ckpt_identity,
        operating_resolution=operating_resolution,
        crop_resolution=crop_resolution,
        frame_count=frame_count,
        metrics=metrics,
        generator_controls=gen_controls,
        uncertainty=uncertainty,
    )
    return record.to_dict()


def adapt_campaign_eval_result(
    eval_result: dict[str, Any],
    *,
    run_id: str,
    backend_name: str,
    arch: str,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    operating_resolution: tuple[int, int] = (1080, 1920),
    crop_resolution: tuple[int, int] = (256, 256),
    host_strata: str = "shared_gpu_server",
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
    code_revision: str | None = None,
    operating_point_id: str = "display_native",
    timing_evidence_id: str | None = None,
) -> dict[str, Any]:
    """Adapt train_campaign aggregate evaluation output into the E01 result schema fail-closed."""
    agg = eval_result.get("aggregate", eval_result)
    per_clip = eval_result.get("per_clip", [])

    # Resolve checkpoint digest strictly
    sha = checkpoint_sha256 or agg.get("checkpoint_sha256")
    if not sha:
        ckpt_ident = str(agg.get("checkpoint_identity", ""))
        if ":" in ckpt_ident:
            possible_sha = ckpt_ident.split(":")[-1]
            if is_valid_sha256_digest(possible_sha):
                sha = possible_sha

    if not sha and checkpoint_path:
        p = Path(str(checkpoint_path))
        if p.is_file():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()

    valid_sha = is_valid_sha256_digest(sha)
    ckpt_sha_clean = str(sha).strip().lower() if valid_sha else ""

    ckpt_id_str = f"{backend_name}:{ckpt_sha_clean or 'unresolved'}"
    ckpt_identity = {
        "checkpoint_id": ckpt_id_str,
        "checkpoint_sha256": ckpt_sha_clean,
        "name": backend_name,
        "arch": arch,
        "seed": agg.get("seed", 0),
    }
    ckpt_identity["config_identity"] = config_identity_digest(ckpt_identity)

    # Group clips by video source to compute honest source-level uncertainty
    by_source: dict[str, list[dict[str, Any]]] = {}
    for c in per_clip:
        if isinstance(c, dict):
            src_key = c.get("video") or c.get("source_id") or "unknown_source"
            by_source.setdefault(src_key, []).append(c)

    source_psnrs: list[float] = []
    source_bytes: list[float] = []
    for _src, clips in by_source.items():
        src_psnr_vals = [float(c["psnr"]) for c in clips if c.get("psnr") is not None and np.isfinite(c["psnr"])]
        if src_psnr_vals:
            source_psnrs.append(float(np.mean(src_psnr_vals)))
        src_byte_vals = [float(c["residual_bytes"]) for c in clips if c.get("residual_bytes") is not None and np.isfinite(c["residual_bytes"])]
        if src_byte_vals:
            source_bytes.append(float(np.mean(src_byte_vals)))

    uncertainty = {
        "psnr": calculate_metric_uncertainty(source_psnrs),
        "residual_bytes": calculate_metric_uncertainty(source_bytes),
        "n_sources": len(by_source),
    }

    metrics = {
        "residual_bytes": agg.get("residual_bytes"),
        "total_bytes": agg.get("total_bytes"),
        "psnr_mean": agg.get("psnr_mean"),
        "ssim_mean": agg.get("ssim_mean"),
        "vmaf_mean": agg.get("vmaf_mean"),
        "temporal_error_mean": agg.get("temporal_error"),
    }

    client_sec = agg.get("client_seconds")
    timing_id = timing_evidence_id or (f"timing_{run_id}_{host_strata}" if client_sec and client_sec > 0 else None)
    timing_evidence = (
        {
            "timing_evidence_id": timing_id,
            "profiling_strata": host_strata,
            "measured_client_seconds": client_sec,
            "encoder_seconds": agg.get("encoder_seconds"),
        }
        if timing_id
        else None
    )

    # In campaign evaluations, controls must be explicitly reported - never defaulted to True!
    cond_tested = bool(agg.get("conditioning_tested", False))
    cond_sensitive = bool(agg.get("conditioning_sensitive", False))
    seed_tested = bool(agg.get("same_seed_tested", False))
    seed_deterministic = bool(agg.get("same_seed_deterministic", False))
    pix_changed = bool(agg.get("delivered_pixels_changed", False))

    gen_controls = {
        "conditioned_vs_shuffled_tested": cond_tested,
        "conditioning_sensitive": cond_sensitive,
        "same_seed_determinism_tested": seed_tested,
        "same_seed_deterministic": seed_deterministic,
        "delivered_pixels_changed": pix_changed,
    }

    controls = {
        "standalone_decode": "verified_client_envelope" if agg.get("success") else "unverified",
        "metric_calibration": "calibrated_anchors_present" if agg.get("psnr_mean") is not None else "unverified",
        "wire_ledger": "verified_reconciliation" if agg.get("total_bytes") is not None else "unverified",
    }

    exclusions: list[dict[str, str]] = []
    rd_eligible = True

    if not agg.get("success", False) or agg.get("eval_failed", False):
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "evaluation was marked unsuccessful or failed"})

    if not valid_sha:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing or invalid checkpoint SHA-256 digest"})

    if not cond_tested:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing conditioning sensitivity control"})
    elif not cond_sensitive:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "failed conditioning sensitivity control (shuffled match)"})

    if not seed_tested:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing same-seed determinism control"})
    elif not seed_deterministic:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "failed same-seed determinism control"})

    if metrics["residual_bytes"] is None and metrics["total_bytes"] is None:
        rd_eligible = False
        exclusions.append({"claim": "rd", "reason": "missing measured wire byte payload"})

    speed_eligible = bool(client_sec is not None and client_sec > 0 and timing_evidence is not None)
    if not speed_eligible:
        exclusions.append({"claim": "runtime", "reason": "timing missing or not a transferable host stratum"})

    standalone_eligible = bool(agg.get("success", False))
    if not standalone_eligible:
        exclusions.append({"claim": "standalone_transport", "reason": "scoring path not proven equal to serialized client output"})

    # Generalization requires independent confirmation sources, not development tracks
    gen_sources = len(by_source)
    generalization_eligible = bool(gen_sources >= 3 and agg.get("confirmation_split", False))
    if not generalization_eligible:
        exclusions.append({"claim": "generalization", "reason": "evaluation does not satisfy independent confirmation split requirement"})

    claim_eligibility = {
        "rd": rd_eligible,
        "runtime": speed_eligible,
        "standalone_transport": standalone_eligible,
        "generalization": generalization_eligible,
        "exclusions": exclusions,
    }

    art_path_str = str(artifact_path or f"outputs/campaign_eval/{run_id}.json")
    if artifact_sha256 and is_valid_sha256_digest(artifact_sha256):
        art_sha = artifact_sha256
    else:
        canonical_bytes = json.dumps(eval_result, sort_keys=True).encode("utf-8")
        art_sha = hashlib.sha256(canonical_bytes).hexdigest()

    rev = code_revision or agg.get("code_revision") or "unspecified_revision"
    source_ids = list(by_source.keys()) if by_source else ["unknown_source"]

    frame_ids = {
        "start": 0,
        "count": len(per_clip),
        "fps": float(agg.get("fps", 24.0)),
    }

    record = GenerationResultRecord(
        schema=SCHEMA_ID,
        artifact_id=f"campaign_eval_{backend_name}_{run_id}",
        artifact_path=art_path_str,
        artifact_sha256=art_sha,
        code_revision=rev,
        source_ids=source_ids,
        frame_ids=frame_ids,
        operating_point_id=operating_point_id,
        claim_eligibility=claim_eligibility,
        controls=controls,
        timing_evidence=timing_evidence,
        doc_role="generation_result",
        backend_name=backend_name,
        arch=arch,
        checkpoint_identity=ckpt_identity,
        operating_resolution=operating_resolution,
        crop_resolution=crop_resolution,
        frame_count=len(per_clip),
        metrics=metrics,
        generator_controls=gen_controls,
        uncertainty=uncertainty,
    )
    return record.to_dict()
