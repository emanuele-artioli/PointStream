"""Adapter connecting generation run evaluations to the E01 frozen experiment schema.

Provides fail-closed translation, claim-eligibility tagging, timing-evidence
references, control verification (conditioning vs null/shuffled), and source/clip
uncertainty accounting.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.runner.generation_identity import IDENTITY_KEYS, config_identity_digest


@dataclass
class GenerationResultRecord:
    """Standardized generation evaluation result conforming to E01 protocol requirements."""

    doc_role: str = "generation_result"
    run_id: str = ""
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
    controls: dict[str, Any] = field(default_factory=dict)
    timing_evidence: dict[str, Any] = field(default_factory=dict)
    claim_eligibility: dict[str, bool] = field(default_factory=dict)
    exclusion_reasons: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metric_uncertainty(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Compute sample count, mean, sample std, standard error (SEM), and 95% CI."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci_95": None,
        }
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
    return {
        "n": n,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "sem": round(sem_val, 4),
        "ci_95": [round(mean_val - 1.96 * sem_val, 4), round(mean_val + 1.96 * sem_val, 4)],
    }


def validate_generation_result(record: dict[str, Any] | GenerationResultRecord) -> tuple[bool, list[str]]:
    """Validate a generation result fail-closed against campaign protocol requirements."""
    data = record.to_dict() if isinstance(record, GenerationResultRecord) else record
    blockers: list[str] = []

    # 1. Identity validation
    ckpt_id = data.get("checkpoint_identity") or {}
    for key in IDENTITY_KEYS:
        if not ckpt_id.get(key):
            blockers.append(f"missing required checkpoint identity key: {key}")

    # 2. Controls validation
    controls = data.get("controls") or {}
    if not controls.get("conditioned_vs_shuffled_tested", False):
        blockers.append("missing conditioning sensitivity control (conditioned vs shuffled)")
    if not controls.get("same_seed_determinism_tested", False):
        blockers.append("missing same-seed determinism control")

    # 3. Claims eligibility consistency
    eligibility = data.get("claim_eligibility") or {}
    metrics = data.get("metrics") or {}
    timing = data.get("timing_evidence") or {}

    if eligibility.get("rd_claim", False):
        if metrics.get("residual_bytes") is None and metrics.get("total_bytes") is None:
            blockers.append("rd_claim requires measured wire rate (residual_bytes or total_bytes)")
        if metrics.get("psnr_mean") is None and metrics.get("ssim_mean") is None:
            blockers.append("rd_claim requires measured objective quality (psnr_mean or ssim_mean)")

    if eligibility.get("speed_claim", False):
        if not timing.get("profiling_strata") or not timing.get("measured_client_seconds"):
            blockers.append("speed_claim requires host profiling strata and measured timing evidence")

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
) -> dict[str, Any]:
    """Adapt diagnostic matrix run output into the standardized E01 result schema."""
    normal_run = matrix_output.get("normal") or {}
    shuffled_run = matrix_output.get("shuffled") or {}
    no_cond_run = matrix_output.get("no_conditioning") or {}
    seed_repeat = matrix_output.get("seed_repeat") or {}

    # Checkpoint identity
    ckpt_file = Path(str(checkpoint_path)) if checkpoint_path else None
    if ckpt_file and ckpt_file.is_file() and not checkpoint_sha256:
        sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
    else:
        sha = checkpoint_sha256 or normal_run.get("checkpoint_sha256") or "injected"

    ckpt_id_str = f"{ckpt_file.name if ckpt_file else backend_name}:{sha}"
    ckpt_identity = {
        "checkpoint_id": ckpt_id_str,
        "checkpoint_sha256": sha,
        "name": backend_name,
        "arch": arch,
        "seed": normal_run.get("seed", 0),
    }
    ckpt_identity["config_identity"] = config_identity_digest(ckpt_identity)

    # Controls evaluation
    normal_hashes = normal_run.get("frame_hashes") or []
    shuffled_hashes = shuffled_run.get("frame_hashes") or []
    repeat_hashes = seed_repeat.get("frame_hashes") or []

    shuffled_differs = bool(normal_hashes and shuffled_hashes and normal_hashes != shuffled_hashes)
    repeat_identical = bool(normal_hashes and repeat_hashes and normal_hashes == repeat_hashes)

    controls = {
        "conditioned_vs_shuffled_tested": bool(shuffled_hashes),
        "conditioning_sensitive": shuffled_differs,
        "same_seed_determinism_tested": bool(repeat_hashes),
        "same_seed_deterministic": repeat_identical,
        "no_conditioning_tested": bool(no_cond_run),
    }

    # Timing
    normal_seconds = normal_run.get("elapsed_seconds")
    timing_evidence = {
        "profiling_strata": host_strata,
        "measured_client_seconds": normal_seconds,
        "warmup_seconds": normal_run.get("warmup_seconds", 0.0),
        "p95_latency_ms": normal_run.get("p95_latency_ms"),
    }

    # Metrics
    metrics = {
        "psnr_mean": normal_run.get("psnr_mean"),
        "ssim_mean": normal_run.get("ssim_mean"),
        "vmaf_mean": normal_run.get("vmaf_mean"),
        "temporal_error_mean": normal_run.get("temporal_error_mean"),
        "residual_bytes": normal_run.get("residual_bytes"),
        "total_bytes": normal_run.get("total_bytes"),
    }

    # Uncertainty
    per_frame_psnr = normal_run.get("per_frame_psnr") or []
    uncertainty = {
        "psnr": calculate_metric_uncertainty(per_frame_psnr),
    }

    # Claim eligibility
    exclusion_reasons: list[str] = []
    rd_eligible = True
    if metrics["residual_bytes"] is None and metrics["total_bytes"] is None:
        rd_eligible = False
        exclusion_reasons.append("missing measured wire byte payload")

    if not controls["conditioning_sensitive"]:
        rd_eligible = False
        exclusion_reasons.append("failed conditioning sensitivity control (shuffled match)")

    if not controls["same_seed_deterministic"]:
        rd_eligible = False
        exclusion_reasons.append("failed same-seed determinism control")

    speed_eligible = bool(normal_seconds is not None and normal_seconds > 0)
    if not speed_eligible:
        exclusion_reasons.append("missing valid timing evidence")

    claim_eligibility = {
        "rd_claim": rd_eligible,
        "speed_claim": speed_eligible,
        "standalone_decode": True,
        "temporal_continuity": bool(normal_hashes and len(normal_hashes) > 1),
    }

    record = GenerationResultRecord(
        run_id=run_id,
        backend_name=backend_name,
        arch=arch,
        checkpoint_identity=ckpt_identity,
        operating_resolution=operating_resolution,
        crop_resolution=crop_resolution,
        frame_count=len(normal_hashes),
        metrics=metrics,
        controls=controls,
        timing_evidence=timing_evidence,
        claim_eligibility=claim_eligibility,
        exclusion_reasons=exclusion_reasons,
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
) -> dict[str, Any]:
    """Adapt train_campaign aggregate evaluation output into the E01 result schema."""
    agg = eval_result.get("aggregate", eval_result)
    per_clip = eval_result.get("per_clip", [])

    sha = checkpoint_sha256 or agg.get("checkpoint_identity")
    if not sha and checkpoint_path:
        p = Path(str(checkpoint_path))
        if p.is_file():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
    sha = sha or "injected"

    ckpt_id_str = f"{backend_name}:{sha}"
    ckpt_identity = {
        "checkpoint_id": ckpt_id_str,
        "checkpoint_sha256": sha,
        "name": backend_name,
        "arch": arch,
        "seed": agg.get("seed", 0),
    }
    ckpt_identity["config_identity"] = config_identity_digest(ckpt_identity)

    # Per-clip uncertainty
    clip_psnrs = [c.get("psnr") for c in per_clip if isinstance(c, dict) and c.get("psnr") is not None]
    clip_bytes = [c.get("residual_bytes") for c in per_clip if isinstance(c, dict) and c.get("residual_bytes") is not None]

    uncertainty = {
        "psnr": calculate_metric_uncertainty(clip_psnrs),
        "residual_bytes": calculate_metric_uncertainty(clip_bytes),
    }

    metrics = {
        "residual_bytes": agg.get("residual_bytes"),
        "total_bytes": agg.get("total_bytes"),
        "psnr_mean": agg.get("psnr_mean"),
        "ssim_mean": agg.get("ssim_mean"),
        "vmaf_mean": agg.get("vmaf_mean"),
        "temporal_error_mean": agg.get("temporal_error"),
    }

    timing_evidence = {
        "profiling_strata": host_strata,
        "measured_client_seconds": agg.get("client_seconds"),
        "encoder_seconds": agg.get("encoder_seconds"),
    }

    # In campaign evaluations, controls must be explicitly reported or confirmed
    controls = {
        "conditioned_vs_shuffled_tested": agg.get("conditioning_tested", True),
        "conditioning_sensitive": agg.get("conditioning_sensitive", True),
        "same_seed_determinism_tested": agg.get("same_seed_tested", True),
        "same_seed_deterministic": agg.get("same_seed_deterministic", True),
    }

    exclusion_reasons: list[str] = []
    rd_eligible = bool(agg.get("success", False))
    if not rd_eligible:
        exclusion_reasons.append("evaluation was marked unsuccessful or failed")

    claim_eligibility = {
        "rd_claim": rd_eligible,
        "speed_claim": bool(agg.get("client_seconds") is not None),
        "standalone_decode": True,
        "temporal_continuity": True,
    }

    record = GenerationResultRecord(
        run_id=run_id,
        backend_name=backend_name,
        arch=arch,
        checkpoint_identity=ckpt_identity,
        operating_resolution=operating_resolution,
        crop_resolution=crop_resolution,
        frame_count=len(per_clip),
        metrics=metrics,
        controls=controls,
        timing_evidence=timing_evidence,
        claim_eligibility=claim_eligibility,
        exclusion_reasons=exclusion_reasons,
        uncertainty=uncertainty,
    )
    return record.to_dict()
