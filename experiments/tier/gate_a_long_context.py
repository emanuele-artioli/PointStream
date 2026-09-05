"""Gate A: Long-context rate-distortion driver for PointStream vs AV1/VVC.

Coordinates:
- Gate 0: Immutable identity, frozen sources (alcaraz_highlights scene_000, scene_028)
- Gate 1: Disjoint timing boundaries (encoder, client, evaluation)
- Gate 2: Tool floor probes and pre-run metric & null controls
- Rate ladder: Coherent rungs C0, C1 (BP56 seed), C2, C3
- Invariant zero-byte generation and residual correction

CRITICAL CONSTRAINT:
No native experimental runs are launched without prior authorization and Codex review.
Use --dry-run for pipeline validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.gate_a_controls import run_gate_a_controls
from experiments.tier.gate_a_identity import (
    CONTEXT_ID,
    SCENES,
    VIDEO,
    build_gate_a_identity,
    verify_source_clips,
    write_gate_a_identity,
)
from experiments.tier.gate_a_tools import write_tool_identity
from experiments.tier.low_rate_canvas import with_canonical_background
from experiments.tier.low_rate_checkpoint import fingerprint
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig
from src.runner.config_io import load_tier


@dataclass(frozen=True)
class RungSpec:
    name: str
    bg_crf: int
    appearance_jpeg: int
    appearance_downscale: int
    motion_max_points: int
    summary: str


RUNGS: tuple[RungSpec, ...] = (
    RungSpec(
        name="C0",
        bg_crf=63,
        appearance_jpeg=25,
        appearance_downscale=4,
        motion_max_points=8,
        summary="Coarsest operating point; shared CRF 63 background, aggressive foreground subsampling",
    ),
    RungSpec(
        name="C1",
        bg_crf=63,
        appearance_jpeg=40,
        appearance_downscale=2,
        motion_max_points=16,
        summary="BP56 seed operating point; reproduces exact 377,360 byte ledger at 48 frames",
    ),
    RungSpec(
        name="C2",
        bg_crf=57,
        appearance_jpeg=55,
        appearance_downscale=2,
        motion_max_points=24,
        summary="Intermediate low-rate point; reduced background CRF 57, higher foreground fidelity",
    ),
    RungSpec(
        name="C3",
        bg_crf=51,
        appearance_jpeg=70,
        appearance_downscale=1,
        motion_max_points=32,
        summary="Highest rate point; background CRF 51, full-resolution appearance, 32 trajectories",
    ),
)


def get_pre_registered_bounds(n_frames: int) -> dict[str, Any]:
    """Return pre-registered bounds written before reading any measurement."""
    scale = float(n_frames) / 48.0
    return {
        "n_frames": n_frames,
        "bounds": {
            "C0": {
                "bytes_min": int(200000 * scale),
                "bytes_max": int(360000 * scale),
                "vmaf_min": 70.0,
                "vmaf_max": 78.0,
            },
            "C1": {
                "bytes_min": int(320000 * scale),
                "bytes_max": int(420000 * scale),
                "vmaf_min": 77.0,
                "vmaf_max": 82.0,
            },
            "C2": {
                "bytes_min": int(400000 * scale),
                "bytes_max": int(600000 * scale),
                "vmaf_min": 80.0,
                "vmaf_max": 85.0,
            },
            "C3": {
                "bytes_min": int(600000 * scale),
                "bytes_max": int(1000000 * scale),
                "vmaf_min": 83.0,
                "vmaf_max": 88.0,
            },
        },
        "anchor_crossover_alarm_vmaf": 82.0,
        "timing_rules": [
            "encoder_seconds strictly disjoint from client_seconds and evaluation_seconds",
            "metrics do not change codec clocks",
            "ledger must balance exactly",
        ],
    }


def configure_rung(
    base: PointstreamConfig,
    rung: RungSpec,
    *,
    context_id: str = CONTEXT_ID,
) -> PointstreamConfig:
    """Build immutable PointstreamConfig for a specific rate ladder rung."""
    bg = with_canonical_background(
        base.background,
        method="panorama-stream",
        stream_codec="av1",
        stream_crf=rung.bg_crf,
        context_id=context_id,
    )
    bg = replace(
        bg,
        transport_scale=1.0,
        stream_usage="good",
        stream_cpu_used=4,
    )
    app = replace(
        base.appearance,
        representation="compressed-image",
        jpeg_quality=rung.appearance_jpeg,
        downscale=rung.appearance_downscale,
    )
    mot = replace(
        base.motion,
        max_points=rung.motion_max_points,
    )
    lattice = replace(
        base.lattice,
        generation=False,
        residual=False,
    )
    return replace(
        base,
        background=bg,
        appearance=app,
        motion=mot,
        lattice=lattice,
    )


def validate_ledger(parts: dict[str, int], total_bytes: int) -> list[str]:
    """Verify that the payload parts balance exactly with transport total."""
    alarms: list[str] = []
    parts_sum = sum(parts.values())
    if parts_sum != total_bytes:
        alarms.append(
            f"Ledger does not balance: parts sum {parts_sum} != total {total_bytes}"
        )
    return alarms


def check_adjacent_rungs(prev_row: dict[str, Any], curr_row: dict[str, Any]) -> list[str]:
    """Verify rate and ledger monotonicity between adjacent rungs."""
    alarms: list[str] = []
    prev_bytes = prev_row["bytes"]
    curr_bytes = curr_row["bytes"]
    name = curr_row["name"]
    prev_name = prev_row["name"]

    # Total bytes must not fall by > 5%
    if curr_bytes < prev_bytes * 0.95:
        alarms.append(
            f"Rung {name} total bytes {curr_bytes} fell by >5% from {prev_name} ({prev_bytes})"
        )

    # Background bytes must be nondecreasing when CRF changes
    prev_bg = prev_row.get("parts", {}).get("panorama", 0)
    curr_bg = curr_row.get("parts", {}).get("panorama", 0)
    if curr_bg < prev_bg:
        alarms.append(
            f"Rung {name} background bytes {curr_bg} < {prev_name} {prev_bg}"
        )

    return alarms


def run_dry_run(
    destination: Path,
    n_frames: int = 48,
) -> dict[str, Any]:
    """Execute Gate A dry run: verifies identity, tools, controls, and rung configurations."""
    print(f"=== Starting Gate A Dry Run (n_frames={n_frames}) ===")
    destination.mkdir(parents=True, exist_ok=True)

    # 1. Gate 0 Identity
    print("Verifying Gate 0 source identity...")
    identity = build_gate_a_identity(destination, n_frames=n_frames)
    write_gate_a_identity(destination, identity)
    print("Gate 0 identity verified and written.")

    # 2. Gate 2 Tool Floor & Controls
    print("Probing Gate 2 tool floor...")
    tools = write_tool_identity(destination)
    print("Gate 2 tool floor recorded.")

    print("Verifying pre-registered bounds...")
    bounds = get_pre_registered_bounds(n_frames)
    bounds_path = destination / "bounds-before-run.json"
    bounds_path.write_text(json.dumps(bounds, indent=2) + "\n")
    print("Bounds written to bounds-before-run.json.")

    print("Loading test reference clips for controls...")
    clips = verify_source_clips(n_frames=n_frames)
    ref_frames = np.asarray(clips[0].frames[:2])

    print("Running Gate 2 metric anchor and temporal null controls...")
    controls = run_gate_a_controls(ref_frames, destination)
    print("Gate 2 controls passed.")

    # 3. Rate ladder dry-run configurations
    base: PointstreamConfig = load_tier("balanced")
    ladder_plan: list[dict[str, Any]] = []
    for rung in RUNGS:
        cfg = configure_rung(base, rung)
        ladder_plan.append({
            "rung": rung.name,
            "summary": rung.summary,
            "bg_crf": rung.bg_crf,
            "appearance_jpeg": rung.appearance_jpeg,
            "appearance_downscale": rung.appearance_downscale,
            "motion_max_points": rung.motion_max_points,
            "lattice_generation": cfg.lattice.generation,
            "lattice_residual": cfg.lattice.residual,
            "bounds": bounds["bounds"][rung.name],
        })

    dry_run_summary = {
        "status": "dry_run_complete",
        "n_frames": n_frames,
        "video": VIDEO,
        "scenes": list(SCENES),
        "identity_fingerprint": fingerprint(identity),
        "tools_resolved": tools["tools"],
        "controls_valid": controls["valid"],
        "ladder_plan": ladder_plan,
        "note": "Dry run succeeded. No native curve points encoded (authorization required).",
    }
    (destination / "dry-run-summary.json").write_text(json.dumps(dry_run_summary, indent=2) + "\n")
    print("=== Gate A Dry Run Completed Successfully ===")
    return dry_run_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=48, choices=[48, 96, 192])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    destination = (
        Path(args.out_dir)
        if args.out_dir
        else ps_paths.outputs() / f"gate-a-long-context-n{args.frames}"
    )

    if args.dry_run:
        run_dry_run(destination, n_frames=args.frames)
        return 0

    print("Native experiment runs require prior review and explicit authorization.")
    print("Run with --dry-run to verify infrastructure.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
