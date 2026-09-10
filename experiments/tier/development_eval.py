"""Development evaluation driver for overnight recovery (residual fidelity & credible anchors).

Uses manifests/development_recovery.json on already exposed BP46 sequences.
Preserves held-out confirmation sources in manifests/gate_b_confirmation.json untouched.

Provides:
- Full experiment identity (exact builds: SvtAv1EncApp v1.8.0, vvencapp 1.11.0, metric versions).
- AV1 and VVC native-resolution curves and resolution-adaptive curves (1.0, 0.5, 0.25 scales).
- Display grid restoration and rescaling time accounting.
- Metric calibration with identical, mild, severe, unrelated anchors, and temporal nulls.
- Residual-on high-fidelity ladder support (coordinating with Worker A).
- Fail-closed protocol enforcement: labeled strictly as a development pilot (gate_b_passed=False).
"""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.tier.calibrate import run_full_metric_calibration
from experiments.tier.gate_a_tools import write_tool_identity
from experiments.tier.low_rate_clips import load_e1_sequence
from experiments.tier.protocol import (
    capture_current_identity,
    evaluate_confirmation_protocol,
)
from experiments.tier.resolution_adaptive import (
    HIGH_FIDELITY_RESIDUAL_RUNGS,
    build_nondominated_envelope,
    compare_curves_no_extrapolation,
    configure_high_fidelity_residual_rung,
    encode_resolution_arm,
)
from experiments.tier.gate_b_confirmation import _validate_point
from src.contracts import paths as ps_paths
from src.contracts.frozen_procedure import (
    FROZEN_RUNGS,
    check_adjacent_rungs,
    configure_frozen_rung,
    get_frozen_bounds,
)
from src.runner.config_io import load_tier


DEFAULT_DEV_MANIFEST = ps_paths.repo_root() / "manifests" / "development_recovery.json"


def load_development_manifest(path: Path = DEFAULT_DEV_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Development recovery manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Verify holdout protection
    forbidden = set(data.get("holdout_protection", {}).get("confirmation_sources_forbidden", []))
    for scene in data.get("scenes", []):
        sid = scene.get("source_id", "")
        if sid in forbidden or any(f in sid for f in forbidden):
            raise RuntimeError(
                f"FATAL: confirmation source {sid} found in development manifest! "
                "Held-out confirmation sources must remain untouched."
            )
    return data


def run_development_pilot(
    destination: Path,
    *,
    manifest_path: Path = DEFAULT_DEV_MANIFEST,
    n_frames: int = 48,
    scales: tuple[float, ...] = (1.0, 0.5, 0.25),
    ladder_type: str = "frozen",  # "frozen" or "residual_high_fidelity"
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute bounded development pilot with full identity and anchor coverage."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_development_manifest(manifest_path)
    write_tool_identity(destination)

    base = load_tier("balanced")
    identity = capture_current_identity(manifest_path, config=base)
    (destination / "experiment-identity.json").write_text(
        json.dumps(identity.to_dict(), indent=2) + "\n"
    )

    if dry_run:
        summary = {
            "status": "development_pilot_dry_run_complete",
            "manifest": str(manifest_path),
            "identity": identity.to_dict(),
            "scenes": [s["source_id"] for s in manifest.get("scenes", [])],
            "scales": list(scales),
            "ladder_type": ladder_type,
        }
        (destination / "dry-run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    per_source_reports: list[dict[str, Any]] = []
    all_alarms: list[str] = []

    # 1. Metric calibration
    print("\n--- Running Metric Calibration and Null Controls ---")
    first_scene = manifest["scenes"][0]
    clips = load_e1_sequence(first_scene["video"], [first_scene["scene"]], n_frames=n_frames)
    first_frames = np.asarray(clips[0].frames)[:n_frames]

    calib_result = run_full_metric_calibration(["psnr", "ssim", "vmaf"], first_frames[:4])
    (destination / "metric-calibration.json").write_text(
        json.dumps(calib_result, indent=2) + "\n"
    )
    if not calib_result.get("valid", False):
        all_alarms.extend(calib_result.get("alarms", []))

    # 2. Iterate scenes
    for sc_meta in manifest.get("scenes", []):
        sid = sc_meta["source_id"]
        vname = sc_meta["video"]
        sname = sc_meta["scene"]
        print(f"\n--- Loading development scene: {sid} ({vname}/{sname}) ---")
        seq_clips = load_e1_sequence(vname, [sname], n_frames=n_frames)
        frames = np.asarray(seq_clips[0].frames)[:n_frames]
        T, H, W, _ = frames.shape

        bounds = get_frozen_bounds(n_frames=n_frames, height=H, width=W, n_scenes=1)

        # 2a. Anchors: AV1 and VVC across native and resolution-adaptive scales
        anchors_report: dict[str, Any] = {}
        for codec in ("av1", "vvc"):
            preset = manifest.get("anchors", {}).get(codec, {}).get("preset", "8" if codec == "av1" else "medium")
            qps = manifest.get("anchors", {}).get(codec, {}).get("qps", [63, 55, 47, 39])

            codec_points: list[dict[str, Any]] = []
            curves_by_scale: dict[str, list[dict[str, Any]]] = {}

            for scale in scales:
                scale_key = "native" if scale == 1.0 else f"res_{int(round(scale * 100))}"
                scale_list: list[dict[str, Any]] = []
                for qp in qps:
                    print(f"  Anchor {codec} scale={scale} qp={qp} preset={preset}")
                    pt = encode_resolution_arm(
                        frames,
                        codec=codec,
                        qp=qp,
                        preset=preset,
                        scale=scale,
                        fps=float(sc_meta.get("fps", 24.0)),
                    )
                    scale_list.append(pt)
                    codec_points.append(pt)
                curves_by_scale[scale_key] = scale_list

            envelope = build_nondominated_envelope(codec_points)
            anchors_report[codec] = {
                "curves_by_scale": curves_by_scale,
                "nondominated_envelope": envelope,
            }

        # 2b. PointStream Candidate ladder
        ps_rows: list[dict[str, Any]] = []
        if ladder_type == "frozen":
            from experiments.tier.low_rate_sweep import pointstream_e1
            for rung in FROZEN_RUNGS:
                cfg = configure_frozen_rung(base, rung, context_id=f"dev_{sid}")
                payload = pointstream_e1(seq_clips, cfg)
                row = {
                    "name": rung.name,
                    "bytes": payload["coded_bytes"],
                    "scores": payload["scores"],
                    "parts": payload["parts"],
                    "usable": payload["usable"],
                    "is_rate": payload["is_rate"],
                    "timing": {
                        "encoder_seconds": payload.get("encoder_seconds"),
                        "client_seconds": payload.get("client_seconds"),
                        "evaluation_seconds": payload.get("evaluation_seconds"),
                        "attempt_wall": payload.get("run_seconds"),
                    },
                    "late_frame": payload.get("late_frame"),
                }
                alarms = _validate_point(row, bounds)
                if ps_rows:
                    alarms.extend(check_adjacent_rungs(ps_rows[-1], row))
                if alarms:
                    all_alarms.extend(alarms)
                ps_rows.append(row)

        elif ladder_type == "residual_high_fidelity":
            from experiments.tier.low_rate_sweep import pointstream_e1
            for res_rung in HIGH_FIDELITY_RESIDUAL_RUNGS:
                cfg = configure_high_fidelity_residual_rung(base, res_rung)
                payload = pointstream_e1(seq_clips, cfg)
                row = {
                    "name": res_rung.rung_id,
                    "bytes": payload["coded_bytes"],
                    "scores": payload["scores"],
                    "parts": payload["parts"],
                    "usable": payload["usable"],
                    "is_rate": payload["is_rate"],
                    "timing": {
                        "encoder_seconds": payload.get("encoder_seconds"),
                        "client_seconds": payload.get("client_seconds"),
                        "evaluation_seconds": payload.get("evaluation_seconds"),
                        "attempt_wall": payload.get("run_seconds"),
                    },
                    "late_frame": payload.get("late_frame"),
                }
                ps_rows.append(row)

        # 2c. Comparisons against native and resolution-adaptive envelope
        comparisons: dict[str, Any] = {}
        for codec in ("av1", "vvc"):
            native_curve = anchors_report[codec]["curves_by_scale"].get("native", [])
            envelope_curve = anchors_report[codec]["nondominated_envelope"]

            comparisons[codec] = {
                "continuous": compare_curves_no_extrapolation(ps_rows, native_curve, metric_name="vmaf"),
                "native": compare_curves_no_extrapolation(ps_rows, native_curve, metric_name="vmaf"),
                "resolution_adaptive_envelope": compare_curves_no_extrapolation(
                    ps_rows, envelope_curve, metric_name="vmaf"
                ),
            }

        per_source_reports.append(
            {
                "source_id": sid,
                "video": vname,
                "scene": sname,
                "resolution": f"{W}x{H}",
                "n_frames": n_frames,
                "pointstream_rungs": ps_rows,
                "anchors": anchors_report,
                "comparisons": comparisons,
            }
        )

    # 3. Protocol evaluation (explicitly a development pilot)
    protocol_verdict = evaluate_confirmation_protocol(
        per_source_reports,
        all_alarms,
        identity=identity,
        is_pilot=True,
        required_matches=6,
    )

    final_report = {
        "doc_role": "development_pilot_report",
        "status": "development_pilot_complete",
        "ladder_type": ladder_type,
        "n_sources": len(per_source_reports),
        "sources": per_source_reports,
        "alarms": all_alarms,
        "metric_calibration": calib_result,
        "identity": identity.to_dict(),
        **protocol_verdict,
        "timestamp_unix": time.time(),
    }

    (destination / "report.json").write_text(json.dumps(final_report, indent=2) + "\n")
    print("\n=== Development Pilot Complete ===")
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="PointStream development evaluation pilot")
    parser.add_argument("--dry-run", action="store_true", help="Preflight check only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ps_paths.outputs() / "development-pilot",
        help="Destination directory",
    )
    parser.add_argument("--frames", type=int, default=48, help="Frame count (default: 48)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DEV_MANIFEST,
        help="Path to development recovery manifest",
    )
    parser.add_argument(
        "--ladder",
        choices=["frozen", "residual_high_fidelity"],
        default="frozen",
        help="Rate ladder type to evaluate",
    )
    args = parser.parse_args()

    run_development_pilot(
        destination=args.output_dir,
        manifest_path=args.manifest,
        n_frames=args.frames,
        ladder_type=args.ladder,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
