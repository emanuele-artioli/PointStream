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
import atexit
import os
import time
import threading
from dataclasses import asdict, dataclass, replace
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
from experiments.tier.low_rate_fallback import run_fallback_control
from experiments.tier.low_rate_references import compare_candidate_to_anchor, encode_reference_curve
from experiments.tier.low_rate_sweep import pointstream_e1
from experiments.tier.low_rate_checkpoint import (
    fingerprint,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
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
    stream_codec: str = "vvc"
    appearance_format: str = "webp"


RUNGS: tuple[RungSpec, ...] = (
    RungSpec(
        name="C0",
        bg_crf=63,
        appearance_jpeg=30,
        appearance_downscale=2,
        motion_max_points=8,
        summary="Coarsest operating point; shared VVC QP 63 background, WebP Q30 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    RungSpec(
        name="C1",
        bg_crf=55,
        appearance_jpeg=45,
        appearance_downscale=1,
        motion_max_points=16,
        summary="Intermediate low-rate point; VVC QP 55 background, WebP Q45 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    RungSpec(
        name="C2",
        bg_crf=48,
        appearance_jpeg=60,
        appearance_downscale=1,
        motion_max_points=24,
        summary="Target competitive point; VVC QP 48 background, WebP Q60 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
    RungSpec(
        name="C3",
        bg_crf=42,
        appearance_jpeg=75,
        appearance_downscale=1,
        motion_max_points=32,
        summary="Highest rate point; VVC QP 42 background, WebP Q75 foreground",
        stream_codec="vvc",
        appearance_format="webp",
    ),
)


def get_pre_registered_bounds(n_frames: int) -> dict[str, Any]:
    """Return the frozen instrument-alarm bounds from the Gate-A brief."""
    raw_bytes = 2 * n_frames * 2160 * 3840 * 3
    return {
        "n_frames_per_scene": n_frames,
        "decoded_shape_required": [2 * n_frames, 2160, 3840, 3],
        "coded_bytes": {"low_exclusive": 0, "high_inclusive": raw_bytes + 1048576},
        "quality": {
            "vmaf": [0.0, 98.0],
            "psnr_y_db": [8.0, 55.0],
            "ssim": [0.0, 1.0],
        },
        "late_frame_last_minus_first": {
            "vmaf": [-25.0, 8.0],
            "psnr_y_db": [-8.0, 3.0],
        },
        "bd_rate_vmaf_percent": [-90.0, 300.0],
        "timing": {
            "required": ["encoder_seconds", "client_seconds", "evaluation_seconds", "attempt_wall"],
            "finite_nonnegative": True,
            "component_le_attempt_wall_tolerance_seconds": 1.0,
            "ranked_encode_decode_must_be_non_null": True,
        },
        "curve": {
            "max_adjacent_inversion_fraction_of_span": 0.05,
            "endpoint_rate_and_quality_inversion_is_alarm": True,
            "continuous_to_segmented_anchor_bytes_max_ratio": 1.05,
            "minimum_usable_points": 4,
            "minimum_vmaf_overlap": 5.0,
        },
        "checkpoint_gap_seconds_max": 3599.0,
        "nonresumable_operation_timeout_seconds": 3300.0,
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
        stream_codec=rung.stream_codec,
        stream_crf=rung.bg_crf,
        context_id=context_id,
    )
    if rung.stream_codec == "av1":
        bg = replace(
            bg,
            transport_scale=1.0,
            stream_usage="good",
            stream_cpu_used=4,
        )
    else:
        bg = replace(
            bg,
            transport_scale=1.0,
        )
    app = replace(
        base.appearance,
        representation="compressed-image",
        jpeg_quality=rung.appearance_jpeg,
        downscale=rung.appearance_downscale,
        format=rung.appearance_format,
    )
    mot = replace(
        base.motion,
        max_points=rung.motion_max_points,
    )
    lattice = replace(
        base.lattice,
        generation=False,
        residual=False,
        pose=False,
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
        alarms.append(f"Ledger does not balance: parts sum {parts_sum} != total {total_bytes}")
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
        alarms.append(f"Rung {name} background bytes {curr_bg} < {prev_name} {prev_bg}")

    return alarms


def run_dry_run(
    destination: Path,
    n_frames: int = 48,
) -> dict[str, Any]:
    """Execute Gate A dry run: verifies identity, tools, controls, and rung configurations."""
    print(f"=== Starting Gate A Dry Run (n_frames={n_frames}) ===")
    destination.mkdir(parents=True, exist_ok=True)

    # 1. Gate 0 identity includes the freshly driven tool floor.
    print("Probing Gate 2 tool floor...")
    tools = write_tool_identity(destination)
    print("Verifying Gate 0 source identity...")
    identity = build_gate_a_identity(
        destination,
        n_frames=n_frames,
        extra={"tools": tools["tools"], "rate_ladder": [asdict(rung) for rung in RUNGS]},
    )
    write_gate_a_identity(destination, identity)
    print("Gate 0 identity verified and written.")
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
    print("Gate 2 metric and temporal controls passed; native controls remain pending.")

    # 3. Rate ladder dry-run configurations
    base: PointstreamConfig = load_tier("balanced")
    ladder_plan: list[dict[str, Any]] = []
    for rung in RUNGS:
        cfg = configure_rung(base, rung)
        ladder_plan.append(
            {
                "rung": rung.name,
                "summary": rung.summary,
                "bg_crf": rung.bg_crf,
                "appearance_jpeg": rung.appearance_jpeg,
                "appearance_downscale": rung.appearance_downscale,
                "motion_max_points": rung.motion_max_points,
                "lattice_generation": cfg.lattice.generation,
                "lattice_residual": cfg.lattice.residual,
                "bounds": bounds,
            }
        )

    dry_run_summary = {
        "status": "dry_run_preflight_complete",
        "n_frames": n_frames,
        "video": VIDEO,
        "scenes": list(SCENES),
        "identity_fingerprint": fingerprint(identity),
        "tools_resolved": tools["tools"],
        "controls_valid": controls["valid"],
        "controls_pending": controls["pending"],
        "ladder_plan": ladder_plan,
        "note": "Dry run succeeded. No native curve points encoded (authorization required).",
    }
    (destination / "dry-run-summary.json").write_text(json.dumps(dry_run_summary, indent=2) + "\n")
    print("=== Gate A Dry Run Completed Successfully ===")
    return dry_run_summary


POOL_LIMITS = {
    "pointstream": 48.0 * 3600.0,
    "anchors": 56.0 * 3600.0,
    "controls": 16.0 * 3600.0,
}
TOTAL_LIMIT_SECONDS = 120.0 * 3600.0
ELAPSED_LIMIT_SECONDS = 96.0 * 3600.0
HEARTBEAT_INTERVAL_SECONDS = 600.0


def _budget_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "total_limit_seconds": TOTAL_LIMIT_SECONDS,
        "elapsed_limit_seconds": ELAPSED_LIMIT_SECONDS,
        "native_started_unix": time.time(),
        "pool_limits_seconds": POOL_LIMITS,
        "pool_spent_seconds": {name: 0.0 for name in POOL_LIMITS},
        "events": [],
    }


def _initialize_budget(path: Path) -> None:
    if not path.is_file():
        write_json(path, _budget_state(path))


def _require_elapsed_budget(path: Path) -> None:
    state = _budget_state(path)
    elapsed = time.time() - float(state["native_started_unix"])
    if elapsed >= float(state["elapsed_limit_seconds"]):
        raise SystemExit("Gate A 96-hour elapsed wall budget exhausted")


def _progress(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"unix": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _start_heartbeat(destination: Path) -> None:
    log_path = destination / "heartbeat.jsonl"
    stopped = threading.Event()

    def beat() -> None:
        while not stopped.wait(HEARTBEAT_INTERVAL_SECONDS):
            _progress(log_path, "heartbeat")

    _progress(log_path, "native-driver-start")
    thread = threading.Thread(target=beat, name="gate-a-heartbeat", daemon=True)
    thread.start()

    def finish() -> None:
        stopped.set()
        _progress(log_path, "native-driver-stop")

    atexit.register(finish)


def _charge(path: Path, pool: str, name: str, seconds: float) -> None:
    state = _budget_state(path)
    spent = float(state["pool_spent_seconds"][pool])
    limit = float(state["pool_limits_seconds"][pool])
    if (
        spent + seconds > limit
        or sum(state["pool_spent_seconds"].values()) + seconds > TOTAL_LIMIT_SECONDS
    ):
        raise SystemExit(f"Gate A budget exhausted before charging {name}")
    state["pool_spent_seconds"][pool] = spent + seconds
    state["events"].append({"pool": pool, "name": name, "seconds": seconds})
    write_json(path, state)


def _require_budget_reserve(path: Path, pool: str) -> None:
    state = _budget_state(path)
    spent = float(state["pool_spent_seconds"][pool])
    limit = float(state["pool_limits_seconds"][pool])
    if limit - spent < 0.15 * limit:
        raise SystemExit(f"Gate A {pool} pool has less than its 15% reserve")


def _checkpointed(
    points: Path,
    budget_path: Path,
    name: str,
    pool: str,
    operation: Any,
) -> dict[str, Any]:
    previous = load_checkpoint(points, name)
    if previous is not None:
        return previous
    attempt_path = points / f"{name}.attempt.json"
    attempt_state = (
        json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt_path.is_file()
        else {"attempts": 0}
    )
    attempt_number = int(attempt_state.get("attempts", 0)) + 1
    if attempt_number > 2:
        raise SystemExit(f"Gate A {name} already used its one permitted retry")
    write_json(
        attempt_path,
        {
            "name": name,
            "pool": pool,
            "attempts": attempt_number,
            "status": "running",
            "started_unix": time.time(),
        },
    )
    _require_elapsed_budget(budget_path)
    _progress(points.parent / "heartbeat.jsonl", "operation-start", name=name, pool=pool)
    _require_budget_reserve(budget_path, pool)
    started = time.perf_counter()
    try:
        row = operation()
    except BaseException:
        write_json(
            attempt_path,
            {
                "name": name,
                "pool": pool,
                "attempts": attempt_number,
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "charged": True,
            },
        )
        _progress(points.parent / "heartbeat.jsonl", "operation-failed", name=name, pool=pool)
        _charge(budget_path, pool, f"{name}:failed", time.perf_counter() - started)
        raise
    elapsed = time.perf_counter() - started
    row["attempt_wall_seconds"] = elapsed
    save_checkpoint(points, name, row)
    write_json(
        attempt_path,
        {
            "name": name,
            "pool": pool,
            "attempts": attempt_number,
            "status": "complete",
            "elapsed_seconds": elapsed,
            "charged": True,
        },
    )
    _progress(points.parent / "heartbeat.jsonl", "operation-complete", name=name, pool=pool)
    _charge(budget_path, pool, name, elapsed)
    return row


def _pointstream_row(
    clips: list[Any], base: PointstreamConfig, rung: RungSpec, points: Path
) -> dict[str, Any]:
    config = configure_rung(base, rung)
    payload = pointstream_e1(clips, config, checkpoint_dir=points / f"{rung.name}.run")
    return {
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


def _object_stream_off(clips: list[Any], base: PointstreamConfig, points: Path) -> dict[str, Any]:
    config = configure_rung(base, RUNGS[1])
    lattice = replace(
        config.lattice,
        appearance=False,
        motion=False,
        temporal_policy=False,
        detection=False,
        selection=False,
        tracking=False,
        pose=False,
        segmentation=False,
        rigid_objects=False,
    )
    payload = pointstream_e1(
        clips,
        replace(config, lattice=lattice),
        checkpoint_dir=points / "object-stream-off.run",
    )
    return {
        "usable": payload["usable"],
        "bytes": payload["coded_bytes"],
        "scores": payload["scores"],
    }


def _validate_point(row: dict[str, Any], bounds: dict[str, Any]) -> list[str]:
    alarms = validate_ledger(row["parts"], int(row["bytes"]))
    if row.get("usable") is not True or row.get("is_rate") is not True:
        alarms.append(f"{row['name']}: unusable or not a coded rate")
    high = int(bounds["coded_bytes"]["high_inclusive"])
    if not 0 < int(row["bytes"]) <= high:
        alarms.append(f"{row['name']}: bytes outside (0,{high}]")
    for key in ("encoder_seconds", "client_seconds", "evaluation_seconds"):
        value = (row.get("timing") or {}).get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            alarms.append(f"{row['name']}: invalid {key}={value}")
    alarms.extend((row.get("late_frame") or {}).get("alarms") or [])
    return alarms


def run_native_duration(destination: Path, n_frames: int) -> dict[str, Any]:
    """Run one authorized duration. It never advances to another duration."""
    os.environ["PS_CODEC_TIMEOUT_SECONDS"] = "3300"
    os.environ["PS_CODEC_MAX_ATTEMPTS"] = "2"
    destination.mkdir(parents=True, exist_ok=True)
    points = destination / "points"
    budget_path = destination / "budget.json"
    _initialize_budget(budget_path)
    _start_heartbeat(destination)
    tools = write_tool_identity(destination)
    identity = build_gate_a_identity(
        destination,
        n_frames=n_frames,
        extra={"tools": tools["tools"], "rate_ladder": [asdict(rung) for rung in RUNGS]},
    )
    write_gate_a_identity(destination, identity)
    bounds = get_pre_registered_bounds(n_frames)
    write_json(destination / "bounds-before-run.json", bounds)
    clips = verify_source_clips(n_frames=n_frames)
    source = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
    base: PointstreamConfig = load_tier("balanced")

    anchors: dict[str, Any] = {}
    for codec in ("av1", "vvc"):
        preset = str(tools["tools"][codec]["slowest_preset"])
        qps = (63, 55, 47, 39)
        anchors[codec] = _checkpointed(
            points,
            budget_path,
            f"{codec}-curves",
            "anchors",
            lambda codec=codec, preset=preset, qps=qps: encode_reference_curve(
                clips,
                codec=codec,
                preset=preset,
                qps=qps,
                fps=24.0,
                checkpoint_root=points / f"{codec}.points",
            ),
        )

    fallback_rows: dict[str, Any] = {}
    for codec in ("av1", "vvc"):
        preset = str(tools["tools"][codec]["slowest_preset"])
        fallback_rows[codec] = _checkpointed(
            points,
            budget_path,
            f"{codec}-fallback",
            "controls",
            lambda codec=codec, preset=preset: run_fallback_control(
                source,
                base.fallback,
                codec=codec,
                qp=63,
                preset=preset,
                fps=24.0,
                rate_rel=(0.95, 1.05),
                vmaf_abs=(-1.0, 1.0),
            ),
        )
    fallback = {
        "passed": all(
            (row.get("comparison") or {}).get("held") is True for row in fallback_rows.values()
        ),
        "anchors": fallback_rows,
    }
    object_off = _checkpointed(
        points,
        budget_path,
        "object-stream-off",
        "controls",
        lambda: _object_stream_off(clips, base, points),
    )
    controls = run_gate_a_controls(
        source[:2], destination, object_stream_off=object_off, fallback=fallback
    )
    if not controls["valid"]:
        raise SystemExit("Gate A native controls failed")

    rows: list[dict[str, Any]] = []
    alarms: list[str] = []
    for rung in RUNGS:
        row = _checkpointed(
            points,
            budget_path,
            rung.name,
            "pointstream",
            lambda rung=rung: _pointstream_row(clips, base, rung, points),
        )
        alarms.extend(_validate_point(row, bounds))
        if rows:
            alarms.extend(check_adjacent_rungs(rows[-1], row))
        rows.append(row)
        write_json(
            destination / "partial-report.json",
            {"identity": identity, "rows": rows, "alarms": alarms},
        )

    comparisons: dict[str, Any] = {}
    for codec, curve in anchors.items():
        comparisons[codec] = {}
        for pattern in ("continuous", "segmented"):
            comparisons[codec][pattern] = compare_candidate_to_anchor(
                rows,
                curve["access_patterns"][pattern],
            )
    completion = {
        "submitted": 4,
        "succeeded": sum(row.get("usable") is True for row in rows),
        "failed": sum(row.get("usable") is not True for row in rows),
    }
    report = {
        "identity": identity,
        "bounds": bounds,
        "controls": controls,
        "anchors": anchors,
        "pointstream": rows,
        "comparisons": comparisons,
        "alarms": alarms,
        "completion": completion,
        "next_duration_authorized": False,
        "note": "Codex must adjudicate this duration before a longer duration starts.",
    }
    write_json(destination / "report.json", report)
    if alarms or completion["failed"]:
        raise SystemExit("Gate A duration completed with alarms; ranking/expansion refused")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=48, choices=[48, 96, 192, 384])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--native", action="store_true", default=False)
    parser.add_argument("--authorize-native", action="store_true", default=False)
    args = parser.parse_args(argv)

    destination = (
        Path(args.out_dir)
        if args.out_dir
        else ps_paths.outputs() / f"gate-a-long-context-n{args.frames}"
    )

    if args.dry_run:
        run_dry_run(destination, n_frames=args.frames)
        return 0

    if args.native:
        if not args.authorize_native:
            raise SystemExit("--native requires --authorize-native after Codex review")
        run_native_duration(destination, n_frames=args.frames)
        return 0

    print("Choose --dry-run or reviewed --native --authorize-native.")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
