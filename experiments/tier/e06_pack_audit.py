"""Derived lossless-pack and client-timing audit of saved E06 transports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from experiments.tier.campaign_result import campaign_record_from_generation_adapter
from experiments.tier.e06_pack import pack_lossless_compact, zip_inventory
from experiments.tier.e06_probe import (
    E03B_RUN,
    E06_OUT,
    PREPARED_SHA256,
    SETTINGS,
    sha256_path,
    verify_immutable_reports,
)
from experiments.tier.e06_transport import (
    PREDICTOR_BBOX_RESIZE,
    PREDICTOR_PER_FRAME,
    require_exact_count,
    require_pixel_parity,
    reconstruct_standalone,
)
from src.runner.generation_adapter import derive_claim_eligibility

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/evaluation-20260914/e06/"
    "audit-20260916-lossless-pack"
)
PARENT_REPORT_SHA256 = "52c0179c724469e56a1dedeafd8695536b44623cd0a63c48bf8128a49f8b5e33"
PARENT_BOUNDS_SHA256 = "0c88b4a360220d452faa540f1ea499a403fdba86990c859466b2a685aceb2230"
PARENT_ROWS_SHA256 = "7bab384fee466c580256f8b9ce14a6409888fadbf97e1c397a9b83dda6ce3c68"
TRANSPORT_SHA256 = {
    "per_frame_crop_residual_off": "da3c66116c11136ea4f24afa17982c4335823d8930a18f54b7eb6a67335c946e",
    "per_frame_crop_residual_on": "e47e81e6e21ffb36952c66e6077f6ad8f8a6cac0b6f10d79247c406a5aa9fb62",
    "bbox_resized_first_reference_residual_off": "949fba3080f030ae24e0a20021b0bc774f99cd3487dad95c7b01a1ea7fa7d8bd",
    "bbox_resized_first_reference_residual_on": "91e0125106731b7dc54b24883d404cc7b23ffc0c8fc3fdc1e3d81cb941dc4a9b",
}

BOUNDS: dict[str, Any] = {
    "written_before_headline_scores": True,
    "written_before_client_timing": True,
    "compact_T_bytes": {
        "low": 1000,
        "high": 33177600,
        "basis": "Compact file cannot exceed uncompressed 360p RGB; tool floor >1 kB.",
    },
    "client_total_s": {
        "low": 0.05,
        "high": 60.0,
        "basis": "Saved 360p VVC plate plus WebP/AV1 client path; stop if one setting exceeds 60 s mean.",
    },
    "pixel_mismatch": {"low": 0, "high": 0},
}

NEIGHBOURS = {
    "av1_qp63": {"bytes": 19116, "psnr_y": 27.4},
    "vvc_qp47": {"bytes": 21288, "psnr_y": 24.6},
}


def code_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pixel_sha256(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames, dtype=np.uint8).tobytes()).hexdigest()


def load_calibration() -> dict[str, Any]:
    path = E03B_RUN / "metric-calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    vmaf = ((payload.get("metrics") or {}).get("vmaf") or {}).get("by_anchor") or {}
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "n_frames": payload.get("n_frames"),
        "resolution": payload.get("resolution"),
        "scope": "two-frame same-grid reused record; not a 48-frame rescore",
        "vmaf_identical": vmaf.get("identical"),
        "vmaf_unrelated": vmaf.get("unrelated-clip"),
        "vmaf_severe_blur": vmaf.get("severe-blur"),
        "source_valid": bool(payload.get("valid")),
    }


def refuse_overwrite(path: Path) -> None:
    if (path / "audit_report.json").is_file():
        raise FileExistsError(f"refusing to overwrite derived audit in {path}")


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    if len(values) == 1:
        return {"n": 1, "mean": round(values[0], 6), "std": None}
    return {
        "n": len(values),
        "mean": round(float(statistics.mean(values)), 6),
        "std": round(float(statistics.stdev(values)), 6),
    }


def profile_client(payload: bytes, *, warmup: int = 1, repeats: int = 3) -> dict[str, Any]:
    for _ in range(warmup):
        reconstruct_standalone(payload)
    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    last = None
    for _ in range(repeats):
        timings: dict[str, float] = {}
        started = time.perf_counter()
        frames = reconstruct_standalone(payload, timings=timings)
        elapsed = time.perf_counter() - started
        totals.append(elapsed)
        last = frames
        for key, value in timings.items():
            stages.setdefault(key, []).append(float(value))
    assert last is not None
    return {
        "warmup": warmup,
        "repeats": repeats,
        "repeat_seconds": [round(item, 6) for item in totals],
        "total_s": _mean_std(totals),
        "stages": {name: _mean_std(vals) for name, vals in stages.items()},
        "frames": last,
    }


def client_timing_to_evidence(
    profile: Mapping[str, Any],
    *,
    setting_id: str,
    host: str,
) -> dict[str, Any]:
    """Named client stratum with actual repeat seconds, not a copied mean."""
    stages_in = profile.get("stages") or {}
    stages: dict[str, Any] = {}
    if isinstance(stages_in, dict):
        for name, vals in stages_in.items():
            if isinstance(vals, dict):
                stages[str(name)] = {
                    "mean": vals.get("mean"),
                    "std": vals.get("std"),
                    "n": vals.get("n"),
                }
    repeats = profile.get("repeat_seconds") or []
    total = profile.get("total_s") or {}
    return {
        "timing_evidence_id": f"e06-client-{setting_id}",
        "host": host,
        "profiling_strata": host,
        "n_repeats": int(profile.get("repeats") or 0),
        "repeat_seconds": [float(item) for item in repeats],
        "measured_client_seconds": total.get("mean"),
        "stages": stages,
    }


def _in_band(value: float, band: dict[str, float]) -> bool:
    return float(band["low"]) <= float(value) <= float(band["high"])


def run_audit(out_dir: Path) -> dict[str, Any]:
    refuse_overwrite(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "bounds.json", BOUNDS)
    identities = verify_immutable_reports()
    parent_report = sha256_path(E06_OUT / "probe_report.json")
    parent_bounds = sha256_path(E06_OUT / "bounds.json")
    parent_rows = sha256_path(E06_OUT / "campaign_rows.json")
    if parent_report != PARENT_REPORT_SHA256:
        raise ValueError(f"parent probe_report mutated: {parent_report}")
    if parent_bounds != PARENT_BOUNDS_SHA256:
        raise ValueError(f"parent bounds mutated: {parent_bounds}")
    if parent_rows != PARENT_ROWS_SHA256:
        raise ValueError(f"parent rows mutated: {parent_rows}")
    original_rows = json.loads((E06_OUT / "campaign_rows.json").read_text(encoding="utf-8"))
    original_by_id = {row["id"]: row for row in original_rows}
    calibration = load_calibration()
    derived_rows = []
    alarms: list[str] = []
    wall0 = time.perf_counter()
    for setting in SETTINGS:
        setting_id = str(setting["id"])
        original = original_by_id[setting_id]
        source = E06_OUT / setting_id / "transport.npz"
        payload = source.read_bytes()
        digest = sha256_bytes(payload)
        if digest != TRANSPORT_SHA256[setting_id]:
            raise ValueError(f"{setting_id} transport SHA-256 {digest}")
        baseline_frames = reconstruct_standalone(payload)
        require_exact_count(
            baseline_frames,
            expected=48,
            height=360,
            width=640,
            source=f"{setting_id}-baseline",
        )
        compact = pack_lossless_compact(payload)
        compact_frames = reconstruct_standalone(compact)
        require_pixel_parity(baseline_frames, compact_frames, source=setting_id)
        inventory = zip_inventory(compact)
        profile = profile_client(compact)
        require_pixel_parity(baseline_frames, profile["frames"], source=f"{setting_id}-timed")
        mean_t = float(profile["total_s"]["mean"] or 0)
        if not _in_band(float(inventory["transport_total"]), BOUNDS["compact_T_bytes"]):
            alarms.append(f"{setting_id}: compact T outside band")
        if not _in_band(mean_t, BOUNDS["client_total_s"]):
            alarms.append(f"{setting_id}: client seconds outside band")
        setting_dir = out_dir / setting_id
        setting_dir.mkdir(parents=True, exist_ok=True)
        compact_path = setting_dir / "transport_compact.npz"
        compact_path.write_bytes(compact)
        pixel_digest = pixel_sha256(baseline_frames)
        parent_parts = original["parts"]
        timing_evidence = client_timing_to_evidence(
            profile,
            setting_id=setting_id,
            host=socket.gethostname(),
        )
        controls = {
            "metric_calibration": "inherited_named_artifact",
            "metric_calibration_verified": True,
            "standalone_decode_verified": True,
            "no_conditioning_tested": False,
            "conditioned_vs_shuffled_tested": False,
            "pose_present": False,
        }
        metrics = {
            "total_bytes": inventory["transport_total"],
            "residual_bytes": parent_parts["residual"],
            "psnr_mean": original["scores"]["psnr_y"],
            "ssim_mean": original["scores"]["ssim"],
            "vmaf_mean": original["scores"]["vmaf"],
        }
        eligibility, exclusions = derive_claim_eligibility(
            metrics=metrics,
            controls=controls,
            timing_evidence=timing_evidence,
            generation_on=False,
            model_free=True,
        )
        adapter_row = {
            "run_id": f"e06-pack-{setting_id}",
            "backend_name": "model_free_e06",
            "model_free": True,
            "generation_on": False,
            "metrics": metrics,
            "scores": original["scores"],
            "parts": {
                "panorama": parent_parts["panorama"],
                "actor_reference": parent_parts["actor_reference"],
                "residual": parent_parts["residual"],
                "transport_total": inventory["transport_total"],
            },
            "controls": {
                "standalone_decode": "verified",
                "metric_calibration": "inherited_named_artifact",
                "wire_ledger": "reconciled",
                "no_conditioning_tested": False,
            },
            "timing_evidence": timing_evidence,
            "claim_eligibility": eligibility,
            "exclusion_reasons": exclusions,
            "artifact_path": str(compact_path),
            "code_revision": code_head(),
            "source_ids": ["federer_djokovic_scene_007"],
            "frame_ids": {"start": 0, "count": 48, "fps": 12.0},
            "operating_point_id": "display_low",
            "delivered_shape": [48, 360, 640, 3],
            "wire_reconciliation": {
                "matched": True,
                "verdict": "matched",
                "wire_bytes": inventory["transport_total"],
                "transport_total": inventory["transport_total"],
            },
        }
        campaign = campaign_record_from_generation_adapter(adapter_row)
        row = {
            "id": setting_id,
            "predictor": original["predictor"],
            "residual_on": original["residual_on"],
            "pose_present": False,
            "limitations": "union-mask / multiple-player; bbox resize is not a geometry warp",
            "parent_transport_sha256": digest,
            "compact_sha256": sha256_path(compact_path),
            "pixel_sha256": pixel_digest,
            "original_T": original["parts"]["transport_total"],
            "compact_inventory": inventory,
            "original_parts": parent_parts,
            "parent_scores": original["scores"],
            "client_timing": {k: v for k, v in profile.items() if k != "frames"},
            "claim_eligibility": eligibility,
            "campaign_claim_eligibility": campaign["claim_eligibility"],
            "exclusions": exclusions,
        }
        _write(setting_dir / "row.json", row)
        derived_rows.append(row)
        del profile["frames"]
    wall = time.perf_counter() - wall0
    if wall > 1800:
        raise TimeoutError(f"audit wall {wall:.1f}s exceeded 30 minutes")
    compact_ts = [int(row["compact_inventory"]["transport_total"]) for row in derived_rows]
    optimistic = min(compact_ts)
    charged_frontier = {
        "lowest_compact_T": optimistic,
        "neighbours": NEIGHBOURS,
        "optimistic_T_still_above_av1_qp63": optimistic > NEIGHBOURS["av1_qp63"]["bytes"],
        "optimistic_T_still_above_vvc_qp47": optimistic > NEIGHBOURS["vvc_qp47"]["bytes"],
        "stop_codec_search": optimistic > NEIGHBOURS["vvc_qp47"]["bytes"],
        "note": (
            "Whole-envelope deflate cannot remove residual payload. "
            "If the cheapest complete compact T still exceeds A(q), stop this configuration search."
        ),
    }
    report = {
        "schema": "pointstream.e06_pack_audit.v1",
        "code_head": code_head(),
        "host": socket.gethostname(),
        "run_dir": str(out_dir),
        "parent_run_dir": str(E06_OUT),
        "parent_hashes": {
            "probe_report": parent_report,
            "bounds": parent_bounds,
            "campaign_rows": parent_rows,
            "prepared": PREPARED_SHA256,
            "e03b_report": identities["e03b_report"],
            "e03b_bounds": identities["e03b_bounds"],
            "e04a_report": identities["e04a_report"],
        },
        "calibration": calibration,
        "confirmation_scoring_authorized": False,
        "native_reencodes": 0,
        "n": 1,
        "settings": derived_rows,
        "wall_seconds": round(wall, 3),
        "alarms": alarms,
        "charged_frontier": charged_frontier,
        "predictors": [PREDICTOR_PER_FRAME, PREDICTOR_BBOX_RESIZE],
    }
    _write(out_dir / "audit_report.json", report)
    _write(out_dir / "campaign_rows.json", derived_rows)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=AUDIT_OUT)
    args = parser.parse_args(argv)
    if args.launch:
        print(json.dumps(run_audit(args.out_dir), indent=2, default=str))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
