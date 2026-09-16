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
from typing import Any, Mapping

import numpy as np

from src.runner.generation_identity import IDENTITY_KEYS, config_identity_digest, sha256_file


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
        if not (
            controls.get("metric_calibration_verified")
            or controls.get("metric_calibration") in {"verified", "inherited_named_artifact", "inherited"}
        ):
            blockers.append("rd_claim requires campaign metric calibration evidence")
        if not controls.get("no_conditioning_tested", False) and not data.get("model_free"):
            blockers.append("rd_claim requires a blank/no-conditioning control when generation is on")
        deployment = data.get("deployment") or {}
        if not data.get("model_free") and not deployment.get("mode"):
            blockers.append("rd_claim requires a declared model deployment cost policy")

    if eligibility.get("speed_claim", False):
        if not timing.get("profiling_strata") or not timing.get("measured_client_seconds"):
            blockers.append("speed_claim requires host profiling strata and measured timing evidence")
        if not timing.get("stages") or not timing.get("n_repeats"):
            blockers.append("speed_claim requires repeated client deserialize/decode/render stages")

    return len(blockers) == 0, blockers


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def derive_claim_eligibility(
    *,
    metrics: Mapping[str, Any],
    controls: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    deployment: Mapping[str, Any] | None = None,
    generation_on: bool = True,
    model_free: bool = False,
) -> tuple[dict[str, bool], list[str]]:
    """Fail closed: producer booleans cannot upgrade missing campaign evidence.

    RD stays eligible when speed is missing. Speed cannot be claimed from a
    single combined encoder+score second count without a named client stratum.
    Fitted-model rows need a declared deployment charge; a checkpoint digest is
    not delivered weights.
    """
    exclusions: list[str] = []
    has_rate = metrics.get("residual_bytes") is not None or metrics.get("total_bytes") is not None
    has_quality = any(
        metrics.get(key) is not None for key in ("psnr_mean", "ssim_mean", "vmaf_mean")
    )
    calib = bool(
        controls.get("metric_calibration_verified")
        or controls.get("metric_calibration")
        in {"verified", "inherited_named_artifact", "inherited"}
    )
    shuffled_ok = (not generation_on) or (
        bool(controls.get("conditioned_vs_shuffled_tested"))
        and bool(controls.get("conditioning_sensitive"))
    )
    blank_ok = (not generation_on) or bool(controls.get("no_conditioning_tested"))
    seed_ok = (not generation_on) or (
        bool(controls.get("same_seed_determinism_tested"))
        and bool(controls.get("same_seed_deterministic"))
    )
    if model_free:
        deployment_ok = True
    else:
        dep = dict(deployment or {})
        mode = str(dep.get("mode") or "")
        if mode == "shared":
            deployment_ok = bool(dep.get("receiver_availability")) and bool(
                dep.get("amortization_policy")
            ) and dep.get("storage_bytes") is not None
        elif mode == "per_video":
            charged = dep.get("charged_bytes", dep.get("storage_bytes"))
            deployment_ok = _positive_number(charged)
        else:
            deployment_ok = False
        if not deployment_ok:
            exclusions.append(
                "undeclared model deployment cost; checkpoint digest is not delivered weights"
            )

    if not has_rate:
        exclusions.append("missing measured wire byte payload")
    if not has_quality:
        exclusions.append("missing measured objective quality")
    if not calib:
        exclusions.append("missing campaign metric calibration evidence")
    if generation_on and not shuffled_ok:
        exclusions.append("failed or missing conditioning sensitivity control (shuffled)")
    if generation_on and not blank_ok:
        exclusions.append("missing blank/no-conditioning control")
    if generation_on and not seed_ok:
        exclusions.append("failed or missing same-seed determinism control")

    rd_claim = bool(
        has_rate
        and has_quality
        and calib
        and shuffled_ok
        and blank_ok
        and seed_ok
        and (model_free or deployment_ok)
    )

    n_repeats = timing_evidence.get("n_repeats", timing_evidence.get("sample_count"))
    strata = timing_evidence.get("profiling_strata") or timing_evidence.get("host")
    client_s = timing_evidence.get("measured_client_seconds")
    stages = timing_evidence.get("stages")
    speed_claim = bool(
        _positive_number(client_s)
        and isinstance(n_repeats, int)
        and not isinstance(n_repeats, bool)
        and n_repeats >= 1
        and isinstance(strata, str)
        and strata not in {"", "unassigned"}
        and isinstance(stages, dict)
        and stages
    )
    if not speed_claim:
        exclusions.append(
            "missing named client deserialize/decode/render stratum with repeats"
        )

    return (
        {
            "rd_claim": rd_claim,
            "speed_claim": speed_claim,
            "standalone_decode": bool(controls.get("standalone_decode_verified")),
            "temporal_continuity": bool(controls.get("temporal_continuity")),
        },
        exclusions,
    )


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pick_matrix_runs(matrix_output: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    matrix_rows = matrix_output.get("matrix")
    empty: dict[str, Any] = {}
    if isinstance(matrix_rows, list):
        normal_run: dict[str, Any] = {}
        shuffled_run: dict[str, Any] = {}
        no_cond_run: dict[str, Any] = {}
        seed_repeat: dict[str, Any] = {}
        paste_run: dict[str, Any] = {}
        for row in matrix_rows:
            if not isinstance(row, dict):
                continue
            corner = str(row.get("corner", ""))
            is_gen = bool(row.get("generation_on", False))
            is_shuffled = bool(row.get("shuffled_conditioning", False))
            is_repeat = bool(
                row.get("seed_repeat", False)
                or corner.endswith("_repeat")
                or corner.endswith("_same_seed")
            )
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
        if not normal_run:
            for row in matrix_rows:
                if isinstance(row, dict) and row.get("generation_on") and not row.get(
                    "shuffled_conditioning"
                ):
                    normal_run = row
                    break
        return {
            "normal": normal_run,
            "shuffled": shuffled_run,
            "no_conditioning": no_cond_run,
            "seed_repeat": seed_repeat,
            "paste": paste_run,
        }
    return {
        "normal": dict(matrix_output.get("normal") or empty),
        "shuffled": dict(matrix_output.get("shuffled") or empty),
        "no_conditioning": dict(matrix_output.get("no_conditioning") or empty),
        "seed_repeat": dict(matrix_output.get("seed_repeat") or empty),
        "paste": dict(matrix_output.get("paste") or empty),
    }


def _row_hashes(row: Mapping[str, Any]) -> list[Any]:
    hashes = row.get("delivered_frame_hashes") or row.get("frame_hashes") or []
    return list(hashes) if isinstance(hashes, list) else []


def _row_scores(row: Mapping[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return {
        "psnr_mean": _first_present(scores, "psnr_y", "psnr_mean")
        if scores
        else _first_present(row, "psnr_y", "psnr_mean"),
        "ssim_mean": _first_present(scores, "ssim", "ssim_mean")
        if scores
        else _first_present(row, "ssim", "ssim_mean"),
        "vmaf_mean": _first_present(scores, "vmaf", "vmaf_mean")
        if scores
        else _first_present(row, "vmaf", "vmaf_mean"),
        "temporal_error_mean": _first_present(scores, "temporal_loss", "temporal_error_mean")
        if scores
        else _first_present(row, "temporal_loss", "temporal_error_mean"),
    }


def _row_parts(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_parts = row.get("parts")
    parts: dict[str, Any] = dict(raw_parts) if isinstance(raw_parts, dict) else {}
    residual = _first_present(parts, "residual") if parts else row.get("residual_bytes")
    total = (
        _first_present(parts, "transport_total")
        if parts
        else _first_present(row, "coded_bytes", "total_bytes")
    )
    return {"residual_bytes": residual, "total_bytes": total, "parts": parts}


def _row_timing(row: Mapping[str, Any]) -> Any:
    nested = row.get("timing") if isinstance(row.get("timing"), dict) else {}
    if nested:
        return _first_present(nested, "client_seconds", "elapsed_seconds")
    return row.get("elapsed_seconds")


def _shape_hw(row: Mapping[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    shape = row.get("delivered_shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 3:
        return int(shape[-3]), int(shape[-2])
    operating = row.get("operating_resolution")
    if isinstance(operating, (list, tuple)) and len(operating) >= 2:
        return int(operating[0]), int(operating[1])
    return fallback


def _code_revision_from_identity(matrix_output: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    identity = matrix_output.get("identity") if isinstance(matrix_output.get("identity"), dict) else {}
    revision = identity.get("code_revision") if isinstance(identity, dict) else None
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    if isinstance(revision, dict) and revision.get("commit"):
        commit = str(revision["commit"])
        if revision.get("dirty"):
            digest = revision.get("diff_sha256")
            return f"{commit}+dirty:{digest}" if digest else f"{commit}+dirty"
        return commit
    return ""


def adapt_diagnostic_matrix_file(
    path: Path | str,
    *,
    run_id: str,
    backend_name: str,
    arch: str,
    checkpoint_path: Path | str | None = None,
    checkpoint_sha256: str | None = None,
    crop_resolution: tuple[int, int] = (256, 256),
    host_strata: str = "shared_gpu_server",
) -> dict[str, Any]:
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    ckpt = checkpoint_sha256 or identity.get("checkpoint_sha256") or payload.get("checkpoint_sha256")
    return adapt_diagnostic_matrix_result(
        payload,
        run_id=run_id,
        backend_name=backend_name,
        arch=arch,
        checkpoint_path=checkpoint_path or payload.get("checkpoint_path"),
        checkpoint_sha256=str(ckpt) if ckpt else None,
        crop_resolution=crop_resolution,
        host_strata=host_strata,
        artifact_path=artifact,
        artifact_sha256=sha256_file(artifact),
        code_revision=_code_revision_from_identity(payload, None) or None,
    )


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
) -> dict[str, Any]:
    """Adapt diagnostic matrix run output into the producer generation result."""
    picked = _pick_matrix_runs(matrix_output)
    normal_run = picked["normal"]
    shuffled_run = picked["shuffled"]
    no_cond_run = picked["no_conditioning"]
    seed_repeat = picked["seed_repeat"]
    paste_run = picked["paste"]

    ckpt_file = Path(str(checkpoint_path)) if checkpoint_path else None
    raw_identity = matrix_output.get("identity")
    identity: dict[str, Any] = dict(raw_identity) if isinstance(raw_identity, dict) else {}
    sha = (
        checkpoint_sha256
        or identity.get("checkpoint_sha256")
        or matrix_output.get("checkpoint_sha256")
        or normal_run.get("checkpoint_sha256")
    )
    if not sha and ckpt_file and ckpt_file.is_file():
        sha = sha256_file(ckpt_file)
    sha = sha or "injected"

    ckpt_id_str = f"{ckpt_file.name if ckpt_file else backend_name}:{sha}"
    ckpt_identity = {
        "checkpoint_id": ckpt_id_str,
        "checkpoint_sha256": sha,
        "name": backend_name,
        "arch": arch,
        "seed": normal_run.get("seed", matrix_output.get("seed", 0)),
    }
    ckpt_identity["config_identity"] = config_identity_digest(ckpt_identity)

    normal_hashes = _row_hashes(normal_run)
    shuffled_hashes = _row_hashes(shuffled_run)
    repeat_hashes = _row_hashes(seed_repeat)
    paste_hashes = _row_hashes(paste_run)

    shuffled_differs = bool(normal_hashes and shuffled_hashes and normal_hashes != shuffled_hashes)
    repeat_identical = bool(normal_hashes and repeat_hashes and normal_hashes == repeat_hashes)
    paste_differs = bool(normal_hashes and paste_hashes and normal_hashes != paste_hashes)

    controls = {
        "conditioned_vs_shuffled_tested": bool(shuffled_hashes),
        "conditioning_sensitive": shuffled_differs,
        "same_seed_determinism_tested": bool(repeat_hashes),
        "same_seed_deterministic": repeat_identical,
        "no_conditioning_tested": bool(no_cond_run),
        "delivered_pixels_changed": paste_differs,
    }

    scores = _row_scores(normal_run)
    parts = _row_parts(normal_run)
    normal_seconds = _row_timing(normal_run)
    resolved_hw = _shape_hw(normal_run, operating_resolution)

    timing_evidence = {
        "profiling_strata": host_strata,
        "measured_client_seconds": normal_seconds,
        "warmup_seconds": normal_run.get("warmup_seconds", 0.0),
        "p95_latency_ms": normal_run.get("p95_latency_ms"),
        "encoder_seconds": (normal_run.get("timing") or {}).get("encoder_seconds")
        if isinstance(normal_run.get("timing"), dict)
        else None,
    }

    metrics = {
        "psnr_mean": scores["psnr_mean"],
        "ssim_mean": scores["ssim_mean"],
        "vmaf_mean": scores["vmaf_mean"],
        "temporal_error_mean": scores["temporal_error_mean"],
        "residual_bytes": parts["residual_bytes"],
        "total_bytes": parts["total_bytes"],
    }

    per_frame_psnr = normal_run.get("per_frame_psnr") or []
    uncertainty = {
        "psnr": calculate_metric_uncertainty(per_frame_psnr),
    }

    exclusion_reasons: list[str] = []
    claim_eligibility, extra_exclusions = derive_claim_eligibility(
        metrics=metrics,
        controls=controls,
        timing_evidence=timing_evidence,
        deployment=matrix_output.get("deployment") if isinstance(matrix_output.get("deployment"), dict) else None,
        generation_on=True,
        model_free=False,
    )
    exclusion_reasons.extend(extra_exclusions)
    claim_eligibility["temporal_continuity"] = bool(normal_hashes and len(normal_hashes) > 1)

    video_id = matrix_output.get("video")
    scene_id = matrix_output.get("scene")
    source_ids: list[str] = []
    if isinstance(video_id, str) and video_id and isinstance(scene_id, str) and scene_id:
        source_ids = [f"{video_id}_{scene_id}"]

    frame_count = len(normal_hashes) if normal_hashes else int(matrix_output.get("frames") or 0)
    fps = matrix_output.get("fps")
    frame_ids: dict[str, Any] = {"start": 0, "count": frame_count} if frame_count else {}
    if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0:
        frame_ids["fps"] = float(fps)

    art_path = Path(artifact_path) if artifact_path is not None else None
    if artifact_sha256:
        art_sha = artifact_sha256
    elif art_path is not None and art_path.is_file():
        art_sha = sha256_file(art_path)
    else:
        art_sha = ""

    revision = _code_revision_from_identity(matrix_output, code_revision)

    record = GenerationResultRecord(
        run_id=run_id,
        backend_name=backend_name,
        arch=arch,
        checkpoint_identity=ckpt_identity,
        operating_resolution=resolved_hw,
        crop_resolution=crop_resolution,
        frame_count=frame_count,
        metrics=metrics,
        controls=controls,
        timing_evidence=timing_evidence,
        claim_eligibility=claim_eligibility,
        exclusion_reasons=exclusion_reasons,
        uncertainty=uncertainty,
    )
    payload = record.to_dict()
    payload["artifact_path"] = str(art_path) if art_path is not None else ""
    payload["artifact_sha256"] = art_sha
    payload["code_revision"] = revision
    payload["source_ids"] = source_ids
    payload["frame_ids"] = frame_ids
    payload["scores"] = {
        "psnr_y": metrics["psnr_mean"],
        "ssim": metrics["ssim_mean"],
        "vmaf": metrics["vmaf_mean"],
    }
    payload["parts"] = parts["parts"]
    payload["coded_bytes"] = parts["total_bytes"]
    payload["timing"] = {
        "client_seconds": normal_seconds,
        "encoder_seconds": timing_evidence.get("encoder_seconds"),
    }
    payload["delivered_shape"] = normal_run.get("delivered_shape")
    payload["wire_reconciliation"] = normal_run.get("wire_reconciliation")
    payload["video"] = video_id
    payload["scene"] = scene_id
    payload["frames"] = matrix_output.get("frames")
    payload["identity"] = identity
    return payload


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
    clip_psnrs: list[float] = [
        float(c["psnr"]) for c in per_clip if isinstance(c, dict) and c.get("psnr") is not None
    ]
    clip_bytes: list[float] = [
        float(c["residual_bytes"])
        for c in per_clip
        if isinstance(c, dict) and c.get("residual_bytes") is not None
    ]

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

    # Campaign evaluations must report controls explicitly; never default them on.
    controls = {
        "conditioned_vs_shuffled_tested": bool(agg.get("conditioning_tested", False)),
        "conditioning_sensitive": bool(agg.get("conditioning_sensitive", False)),
        "same_seed_determinism_tested": bool(agg.get("same_seed_tested", False)),
        "same_seed_deterministic": bool(agg.get("same_seed_deterministic", False)),
    }

    exclusion_reasons: list[str] = []
    claim_eligibility, extra_exclusions = derive_claim_eligibility(
        metrics=metrics,
        controls=controls,
        timing_evidence=timing_evidence,
        deployment=agg.get("deployment") if isinstance(agg.get("deployment"), dict) else None,
        generation_on=True,
        model_free=False,
    )
    exclusion_reasons.extend(extra_exclusions)
    if not bool(agg.get("success", False)):
        claim_eligibility["rd_claim"] = False
        exclusion_reasons.append("evaluation was marked unsuccessful or failed")

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
