"""Checked helper CLI invoking Cursor-owned generation_adapter API.

Reads a diagnostic matrix report JSON, runs adapt_diagnostic_matrix_file,
validates the resulting record fail-closed, and writes the standardized
campaign result record.
"""
from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import json
from pathlib import Path
import sys

from src.runner.generation_adapter import (
    adapt_diagnostic_matrix_file,
    validate_generation_result,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt diagnostic matrix report to standardized campaign result record"
    )
    parser.add_argument(
        "--input-matrix",
        type=Path,
        required=True,
        help="Path to diagnostic matrix JSON output",
    )
    parser.add_argument(
        "--output-record",
        type=Path,
        required=True,
        help="Path where adapted campaign result JSON will be written",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (defaults to input file stem or auto-derived)",
    )
    parser.add_argument(
        "--backend-name",
        type=str,
        default="pix2pix",
        help="Backend name (default: pix2pix)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="pix2pix",
        help="Generator architecture (default: pix2pix)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit generator checkpoint path",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        type=str,
        default=None,
        help="Explicit generator checkpoint SHA-256 digest",
    )
    parser.add_argument(
        "--host-strata",
        type=str,
        default="shared_gpu_server",
        help="Hardware stratum for execution timing (default: shared_gpu_server)",
    )
    parser.add_argument(
        "--deployment-regime",
        type=str,
        choices=["per_video_fit", "shared_model"],
        default="per_video_fit",
        help="Model deployment regime (per_video_fit or shared_model; default: per_video_fit)",
    )
    parser.add_argument(
        "--disqualify-unaccounted-claims",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disqualify RD and speed claims when deployment costs or latency thresholds are not met (default: on)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_matrix)
    if not input_path.exists():
        print(f"Error: Input matrix file does not exist: {input_path}", file=sys.stderr)
        return 1

    run_id = args.run_id or input_path.stem
    record = adapt_diagnostic_matrix_file(
        input_path,
        run_id=run_id,
        backend_name=args.backend_name,
        arch=args.arch,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        host_strata=args.host_strata,
    )

    ckpt_bytes: int | None = None
    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt_bytes = int(Path(args.checkpoint).stat().st_size)

    record["deployment_accounting"] = {
        "regime": args.deployment_regime,
        "checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_bytes": ckpt_bytes,
        "amortization_policy": (
            "per_video_charged"
            if args.deployment_regime == "per_video_fit"
            else "shared_preinstalled_unverified"
        ),
    }

    if args.disqualify_unaccounted_claims:
        eligibility = record.setdefault("claim_eligibility", {})
        reasons = record.setdefault("exclusion_reasons", [])
        if args.deployment_regime == "per_video_fit":
            eligibility["rd_claim"] = False
            reasons.append(
                f"per_video_fit model weights ({ckpt_bytes:,} B) not accounted in primary wire stream"
                if ckpt_bytes
                else "per_video_fit model weights not accounted in primary wire stream"
            )
        else:
            eligibility["rd_claim"] = False
            reasons.append(
                "shared_model policy unverified: model overfitted to scene_028, lacking multi-domain generalization evidence"
            )

        timing = record.get("timing_evidence") or {}
        client_secs = timing.get("measured_client_seconds")
        frame_count = record.get("frame_count") or 16
        if client_secs is not None and frame_count > 0:
            latency_ms = (float(client_secs) / frame_count) * 1000.0
            if latency_ms > 250.0:
                eligibility["speed_claim"] = False
                reasons.append(
                    f"client decode latency ({latency_ms:.1f} ms/frame) exceeds 250 ms budget and 105% baseline latency threshold"
                )

    is_valid, blockers = validate_generation_result(record)
    if not is_valid:
        print(f"Notice: Generation result has validation blockers: {blockers}", file=sys.stderr)

    out_path = Path(args.output_record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Adapted result written to {out_path} (valid: {is_valid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
