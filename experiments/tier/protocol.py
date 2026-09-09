"""Experiment identity and fail-closed protocol enforcement for PointStream.

Implements EVAL-ACT-06 protocol validator:
- Complete experiment identity: configuration, exact codec builds/presets (SvtAv1EncApp v1.8.0,
  vvencapp 1.11.0), model hashes, manifest digest, metric versions, and runtime policy.
- Fail-closed protocol enforcement: requires evidence of actual client-output scoring,
  measured full wire cost, calibrated metrics/nulls, source eligibility, and uncertainty.
- Rejection of empty inputs, missing anchors, incomplete/duplicate sources (six independent
  matches required for Gate B, non-overlap curves unscorable).
- Explicit pilot vs confirmation labeling: pilot completion cannot imply confirmation.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.tier.gate_a_tools import resolve_tool_specs


@dataclass(frozen=True)
class ExperimentIdentity:
    """Cryptographic identity of an evaluation run.

    Includes full configuration, exact codec builds and presets, model weights hashes,
    manifest metadata and digest, metric versions, and runtime policy.
    """

    doc_role: str = "experiment_identity"
    config_fingerprint: str = ""
    manifest_path: str = ""
    manifest_sha256: str = ""
    manifest_schema: str = ""
    source_ids: tuple[str, ...] = ()
    codec_tools: dict[str, Any] = field(default_factory=dict)
    model_hashes: dict[str, str] = field(default_factory=dict)
    metric_versions: dict[str, str] = field(default_factory=dict)
    runtime_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_role": self.doc_role,
            "config_fingerprint": self.config_fingerprint,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_schema": self.manifest_schema,
            "source_ids": list(self.source_ids),
            "codec_tools": self.codec_tools,
            "model_hashes": self.model_hashes,
            "metric_versions": self.metric_versions,
            "runtime_policy": self.runtime_policy,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_match(self, expected: ExperimentIdentity | dict[str, Any]) -> tuple[bool, list[str]]:
        """Assert that this identity matches expected specification."""
        exp_dict = expected.to_dict() if isinstance(expected, ExperimentIdentity) else expected
        mismatches: list[str] = []

        if exp_dict.get("config_fingerprint") and self.config_fingerprint != exp_dict["config_fingerprint"]:
            mismatches.append(
                f"config fingerprint mismatch: got {self.config_fingerprint}, expected {exp_dict['config_fingerprint']}"
            )

        if exp_dict.get("manifest_sha256") and self.manifest_sha256 != exp_dict["manifest_sha256"]:
            mismatches.append(
                f"manifest SHA256 mismatch: got {self.manifest_sha256}, expected {exp_dict['manifest_sha256']}"
            )

        # Check codec tools (executables, versions, presets)
        for codec, exp_tool in (exp_dict.get("codec_tools") or {}).items():
            actual_tool = self.codec_tools.get(codec)
            if not actual_tool:
                mismatches.append(f"missing codec tool identity for {codec}")
                continue
            for key in ("binary", "version", "preset", "slowest_preset"):
                if key in exp_tool and actual_tool.get(key) != exp_tool[key]:
                    mismatches.append(
                        f"{codec} tool {key} mismatch: got {actual_tool.get(key)!r}, expected {exp_tool[key]!r}"
                    )

        # Check model hashes
        for model_name, exp_hash in (exp_dict.get("model_hashes") or {}).items():
            actual_hash = self.model_hashes.get(model_name)
            if actual_hash != exp_hash:
                mismatches.append(
                    f"model {model_name} hash mismatch: got {actual_hash!r}, expected {exp_hash!r}"
                )

        # Check metric versions
        for metric_name, exp_ver in (exp_dict.get("metric_versions") or {}).items():
            actual_ver = self.metric_versions.get(metric_name)
            if actual_ver != exp_ver:
                mismatches.append(
                    f"metric {metric_name} version mismatch: got {actual_ver!r}, expected {exp_ver!r}"
                )

        return len(mismatches) == 0, mismatches


def capture_metric_versions() -> dict[str, str]:
    """Capture versions of installed metric libraries."""
    versions: dict[str, str] = {
        "psnr_y": "numpy-luma-v1",
        "ssim": "pointstream.ssim.v1",
    }
    # Check libvmaf via ffmpeg
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
        )
        if "--enable-libvmaf" in out:
            versions["vmaf"] = "libvmaf (ffmpeg)"
        else:
            versions["vmaf"] = "ffmpeg-no-libvmaf"
    except Exception:
        versions["vmaf"] = "unavailable"
    return versions


def capture_current_identity(
    manifest_path: Path,
    *,
    config: Any | None = None,
    model_hashes: dict[str, str] | None = None,
    runtime_policy: dict[str, Any] | None = None,
) -> ExperimentIdentity:
    """Capture current environment identity for manifest, tools, and configs."""
    manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else b""
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes else ""
    manifest_data = json.loads(manifest_bytes.decode("utf-8")) if manifest_bytes else {}

    source_ids: list[str] = []
    for s in manifest_data.get("sources", []):
        if "source_id" in s:
            source_ids.append(s["source_id"])
    for s in manifest_data.get("scenes", []):
        if "source_id" in s:
            source_ids.append(s["source_id"])

    tools = resolve_tool_specs()
    # Augment tools with binary hashes
    tools_clean: dict[str, Any] = {}
    for codec, info in tools.items():
        bin_path = Path(str(info.get("binary", "")))
        bin_sha = hashlib.sha256(bin_path.read_bytes()).hexdigest() if bin_path.is_file() else ""
        tools_clean[codec] = {
            "codec": codec,
            "binary": str(bin_path),
            "binary_sha256": bin_sha,
            "version": info.get("version"),
            "slowest_preset": info.get("slowest_preset"),
            "available": info.get("available", False),
        }

    cfg_fingerprint = ""
    if config is not None:
        cfg_str = str(config)
        cfg_fingerprint = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()

    policy = runtime_policy or {
        "fail_closed": True,
        "hourly_checkpoint_seconds": 3600.0,
        "max_timeout_seconds": 28800.0,
    }

    return ExperimentIdentity(
        config_fingerprint=cfg_fingerprint,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        manifest_schema=manifest_data.get("schema", "unknown"),
        source_ids=tuple(source_ids),
        codec_tools=tools_clean,
        model_hashes=model_hashes or {},
        metric_versions=capture_metric_versions(),
        runtime_policy=policy,
    )


@dataclass
class ProtocolEvidence:
    """Verification evidence required before any confirmation or gate pass can be claimed.

    Enforces EVAL-ACT-06 requirements:
    - client_output_scored: proof that actual client decoded output was scored
    - measured_wire_cost: proof that all transmitted bytes (residual, masks, metadata) were counted
      and reconciled with the wire envelope
    - calibrated_metrics: proof that metric anchors (identical, mild, severe, unrelated) were evaluated
      and passed ordering and scale checks
    - calibrated_nulls: proof that temporal/spatial null controls were evaluated
    - source_eligibility: proof that sources are eligible (minimum 6 independent matches for Gate B)
    - source_uncertainty: proof that source-level uncertainty was evaluated across independent sources
    """

    client_output_scored: bool = False
    client_scoring_details: dict[str, Any] = field(default_factory=dict)
    measured_wire_cost: bool = False
    wire_cost_reconciled: bool = False
    wire_cost_details: dict[str, Any] = field(default_factory=dict)
    calibrated_metrics: bool = False
    metric_calibration_details: dict[str, Any] = field(default_factory=dict)
    calibrated_nulls: bool = False
    null_control_details: dict[str, Any] = field(default_factory=dict)
    source_eligibility: bool = False
    source_eligibility_details: dict[str, Any] = field(default_factory=dict)
    source_uncertainty: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate evidence fail-closed."""
        blockers: list[str] = []
        if not self.client_output_scored:
            blockers.append("missing evidence of actual client-output scoring (stand-alone client decode)")
        if not self.measured_wire_cost:
            blockers.append("missing evidence of measured full wire cost (wire envelope accounting)")
        if not self.wire_cost_reconciled:
            blockers.append("wire cost not reconciled with payload ledger")
        if not self.calibrated_metrics:
            blockers.append("missing or failed metric calibration with known anchors (identical/mild/severe/unrelated)")
        if not self.calibrated_nulls:
            blockers.append("missing or failed null controls (shuffled-frame temporal null)")
        if not self.source_eligibility:
            blockers.append("source eligibility unverified")
        if not self.source_uncertainty or not self.source_uncertainty.get("evaluated", False):
            blockers.append("missing source-level uncertainty quantification")

        return len(blockers) == 0, blockers


def calculate_source_uncertainty(
    sources: list[dict[str, Any]],
    *,
    codecs: tuple[str, ...] = ("av1", "vvc"),
) -> dict[str, Any]:
    """Compute source-level uncertainty across independent matches.

    Frames are not independent samples; statistics must be settled across sources.
    """
    results: dict[str, Any] = {"evaluated": False, "n_sources": len(sources), "codecs": {}}
    if len(sources) < 2:
        results["reason"] = "at least 2 independent sources required to compute source uncertainty"
        return results

    for codec in codecs:
        deltas: list[float] = []
        for s in sources:
            comp = (s.get("comparisons") or {}).get(codec, {}).get("continuous", {})
            val = comp.get("bd_rate_percent")
            if isinstance(val, (int, float)) and np.isfinite(val):
                deltas.append(float(val))

        if len(deltas) < len(sources):
            results["codecs"][codec] = {
                "n_finite": len(deltas),
                "error": "some sources lack finite BD-rate comparison",
            }
            continue

        arr = np.array(deltas, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem_val = float(std_val / np.sqrt(len(arr))) if len(arr) > 0 else 0.0

        results["codecs"][codec] = {
            "n_sources": len(arr),
            "bd_rate_mean_percent": round(mean_val, 3),
            "bd_rate_std_percent": round(std_val, 3),
            "bd_rate_sem_percent": round(sem_val, 3),
            "bd_rate_min_percent": round(float(np.min(arr)), 3),
            "bd_rate_max_percent": round(float(np.max(arr)), 3),
            "confidence_interval_95_percent": [
                round(mean_val - 1.96 * sem_val, 3),
                round(mean_val + 1.96 * sem_val, 3),
            ],
        }

    results["evaluated"] = all(
        results["codecs"].get(c, {}).get("n_sources") == len(sources) for c in codecs
    )
    return results


def evaluate_confirmation_protocol(
    sources: list[dict[str, Any]],
    alarms: list[str],
    *,
    identity: ExperimentIdentity | dict[str, Any] | None = None,
    expected_identity: ExperimentIdentity | dict[str, Any] | None = None,
    evidence: ProtocolEvidence | dict[str, Any] | None = None,
    is_pilot: bool = True,
    required_matches: int = 6,
) -> dict[str, Any]:
    """Comprehensive, fail-closed evaluation protocol for Gate B and development pilots.

    Enforces:
    1. Rejection of empty inputs.
    2. Verification of source eligibility and independent-source count:
       duplicate sources (same match_name or source_id) do not count as independent matches.
       Gate B confirmation requires >= 6 independent matches.
    3. Rejection of missing anchors (both AV1 and VVC required for every source).
    4. Rejection of unscorable / non-overlapping curves (no BD-rate extrapolation).
    5. Identity matching: config, tool builds/presets, manifest, model hashes.
    6. Protocol evidence validation: client scoring, wire cost, calibrated metrics/nulls, uncertainty.
    7. Clear separation between pilot runs and confirmation:
       pilot completion CANNOT imply confirmation (`is_pilot=True` always sets `gate_b_passed=False`).
    """
    blockers: list[str] = []

    # 1. Reject empty inputs
    if not sources:
        blockers.append(f"empty sources input: {required_matches} independent sources required, 0 provided")
        return {
            "execution_completed": False,
            "pilot_alarms_clear": False,
            "gate_b_passed": False,
            "confirmation_status": "empty_input",
            "confirmation_blockers": blockers,
            "n_sources": 0,
            "n_unique_matches": 0,
        }

    # 2. Check independent source count & duplicate rejection
    unique_matches: set[str] = set()
    for s in sources:
        # Match identifier: prefer match_name, fall back to source_id
        match_id = s.get("match_name") or s.get("source_id") or ""
        if match_id:
            unique_matches.add(str(match_id))

    if len(unique_matches) < required_matches:
        blockers.append(
            f"{required_matches} independent sources required; only {len(unique_matches)} "
            f"unique independent match(es) reported (total reported: {len(sources)})"
        )

    # 3. Check for alarms
    if alarms:
        blockers.append(f"execution or measurement alarms remain ({len(alarms)} alarms)")

    # 4. Check anchors and curve overlap
    for source in sources:
        sid = source.get("source_id", "unknown")
        comparisons = source.get("comparisons") or {}
        for codec in ("av1", "vvc"):
            if codec not in comparisons:
                blockers.append(f"{sid}/{codec}: missing anchor comparison")
                continue
            comparison = comparisons[codec].get("continuous", {})
            delta = comparison.get("bd_rate_percent")
            if not isinstance(delta, (int, float)) or not np.isfinite(delta):
                reason = comparison.get("reason", "no finite overlapping-curve comparison")
                blockers.append(f"{sid}/{codec}: non-overlapping or unscorable curves ({reason})")
            elif delta >= 0:
                blockers.append(f"{sid}/{codec}: recorded curve does not show a rate saving (delta={delta:+.2f}%)")

    # 5. Check experiment identity
    if expected_identity is not None and identity is not None:
        actual_id = (
            identity
            if isinstance(identity, ExperimentIdentity)
            else ExperimentIdentity(**identity)
        )
        matched, id_blockers = actual_id.verify_match(expected_identity)
        if not matched:
            blockers.extend([f"identity mismatch: {b}" for b in id_blockers])

    # 6. Check protocol evidence
    evidence_obj: ProtocolEvidence | None = None
    if evidence is not None:
        if isinstance(evidence, ProtocolEvidence):
            evidence_obj = evidence
        elif isinstance(evidence, dict):
            evidence_obj = ProtocolEvidence(**evidence)
        ev_ok, ev_blockers = evidence_obj.validate()
        if not ev_ok:
            blockers.extend(ev_blockers)
    else:
        blockers.append(
            "confirmation protocol validation is not implemented (EVAL-ACT-06): "
            "source eligibility, frozen identity, metric controls, object scores, "
            "source uncertainty, wire accounting and independent-output scoring"
        )

    # 7. Check pilot vs confirmation status
    if evidence is None:
        status = "incomplete_protocol"
    elif is_pilot:
        blockers.append("pilot completion cannot imply confirmation (development pilot run)")
        status = "development_pilot"
    elif blockers:
        status = "failed_protocol" if any("mismatch" in b or "alarms" in b for b in blockers) else "incomplete_protocol"
    else:
        status = "confirmed_gate_b"

    gate_b_passed = (not is_pilot) and (len(blockers) == 0)

    # Compute source uncertainty if available
    uncertainty = calculate_source_uncertainty(sources)

    return {
        "execution_completed": True,
        "pilot_alarms_clear": len(alarms) == 0,
        "gate_b_passed": gate_b_passed,
        "confirmation_status": status,
        "confirmation_blockers": blockers,
        "n_sources": len(sources),
        "n_unique_matches": len(unique_matches),
        "unique_matches": sorted(unique_matches),
        "source_uncertainty": uncertainty,
        "identity_verified": (expected_identity is not None and identity is not None and len([b for b in blockers if "identity" in b]) == 0),
        "evidence_verified": (evidence_obj is not None and len([b for b in blockers if "evidence" in b or "missing" in b]) == 0),
    }
