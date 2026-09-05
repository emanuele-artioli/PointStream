"""Bounded BP56 background encoder-effort diagnostic.

At most three scale-1 PointStream points on the BP52 pair:
realtime cpu-used 8 CRF51, good cpu-used 4 CRF51, then good cpu-used 4 CRF63
only if time remains. Never writes BP49–BP53 output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.bp52_background_search import (
    EXPECTED_CONTEXT,
    EXPECTED_SHAPE,
    EXPECTED_SOURCE,
    _manifest_snapshot,
    _verify_input,
    run_metric_controls,
)
from experiments.tier.bp53_background_scale import (
    BP52_CRF51,
    BP52_FFMPEG,
    _as_int,
)
from experiments.tier.bp53_budget import (
    HEARTBEAT_INTERVAL_S,
    POINT_RESERVE_S,
    AttemptSession,
    budget_alarms,
    finish_attempt,
    load_budget,
    longer_runs_cleared,
    over_budget,
    reconcile_checkpoints,
    recover_interrupted,
    remaining_seconds,
)
from experiments.tier.low_rate_checkpoint import (
    completion_counts,
    fingerprint,
    guard_checkpoints,
    implementation_digest,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from experiments.tier.low_rate_clips import (
    DEFAULT_SPAN_FRAMES,
    DEFAULT_VIDEO,
    load_e1_sequence,
)
from experiments.tier.low_rate_measure import primary_preset, stream_codec_provenance
from experiments.tier.low_rate_plan import named_point
from experiments.tier.low_rate_sweep import apply_point, pointstream_e1
from experiments.tier.low_rate_validate import DECLARED_FPS
from src.components.background.scale import HEADER_BYTES
from src.components.background.stream import (
    CANDIDATE_STREAM_CPU_USED,
    CANDIDATE_STREAM_USAGE,
    DEFAULT_STREAM_CPU_USED,
    DEFAULT_STREAM_USAGE,
    assert_independent_prefixes_stable,
    ffmpeg_provenance,
    ffmpeg_timeout,
    independent_prefix_payloads,
    last_encode_record,
    probe_stream_effort,
)
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig


POINT_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("bg-realtime8-crf51", DEFAULT_STREAM_USAGE, DEFAULT_STREAM_CPU_USED, "bg-crf51"),
    ("bg-good4-crf51", CANDIDATE_STREAM_USAGE, CANDIDATE_STREAM_CPU_USED, "bg-crf51"),
    ("bg-good4-crf63", CANDIDATE_STREAM_USAGE, CANDIDATE_STREAM_CPU_USED, "bg-crf63"),
)
POINT_ENCODE_TIMEOUT_S = 3600.0
SCORING_RESERVE_S = 900.0
CONTROL_NAME = "bg-realtime8-crf51"
METRIC_PATHS: tuple[str, ...] = (
    "src/components/metrics/__init__.py",
    "src/components/metrics/evaluator.py",
    "src/components/metrics/psnr.py",
    "src/components/metrics/ssim.py",
    "src/components/metrics/vmaf.py",
    "experiments/tier/low_rate_measure.py",
    "experiments/tier/calibrate.py",
)


def bounds_document() -> dict[str, Any]:
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "bp56-background-encoder-effort",
        "result_read_after": "bounds and controls",
        "basis": {
            "control": (
                "Reproduce BP53 scale-1 CRF51 / BP52 CRF51 quality and "
                "panorama/actor bytes. Metadata may differ only by documented "
                "geometry-header accounting (2*HEADER_BYTES vs BP52). "
                "Run 30--10,800 s."
            ),
            "candidate": (
                "Broad diagnostic bands, not hoped-for gains: VMAF 0--98, "
                "Y-PSNR 8--45 dB, SSIM 0--1, positive coded bytes below 50 MB."
            ),
            "late_frame": (
                "Scene-local last-minus-first from BP49/BP52: VMAF [-25,+8] and "
                "Y-PSNR [-8,+3] dB."
            ),
        },
        "invariants": {
            "frames": 96,
            "resolution": "3840x2160",
            "fps": 24.0,
            "generation": False,
            "residual": False,
            "transport_scale": 1.0,
            "point_count_max": 3,
            "checkpoint_gap_seconds_max": 3600.0,
            "hourly_gap_clearance_seconds": 1.0,
            "encode_timeout_seconds": POINT_ENCODE_TIMEOUT_S,
            "geometry_header_bytes_per_scene": HEADER_BYTES,
        },
        "points": {
            CONTROL_NAME: {
                "coded_bytes": [80000, 50000000],
                "vmaf": [15.0, 97.0],
                "psnr_y": [16.0, 45.0],
                "ssim": [0.72, 0.995],
                "run_seconds": [30.0, 10800.0],
            },
            "bg-good4-crf51": {
                "coded_bytes": [1, 50000000],
                "vmaf": [0.0, 98.0],
                "psnr_y": [8.0, 45.0],
                "ssim": [0.0, 1.0],
            },
            "bg-good4-crf63": {
                "coded_bytes": [1, 50000000],
                "vmaf": [0.0, 98.0],
                "psnr_y": [8.0, 45.0],
                "ssim": [0.0, 1.0],
            },
        },
        "late_frame": {"vmaf": [-25.0, 8.0], "psnr_y": [-8.0, 3.0]},
    }


def _point_bounds(name: str, bounds: dict[str, Any]) -> dict[str, list[float]]:
    return bounds["points"][name]


def _point_alarms(name: str, row: dict[str, Any], bounds: dict[str, Any]) -> list[str]:
    payload = row.get("pointstream")
    if not payload:
        return [f"{name}: point failed without a PointStream result"]
    alarms: list[str] = []
    if payload.get("usable") is not True or payload.get("is_rate") is not True:
        alarms.append(f"{name}: unusable or non-coded result")
    parts = payload.get("parts") or {}
    if not parts or sum(parts.values()) != payload.get("coded_bytes"):
        alarms.append(f"{name}: byte ledger does not balance")
    if payload.get("n_frames") != 96:
        alarms.append(f"{name}: n_frames={payload.get('n_frames')} != 96")
    for key in ("vmaf", "psnr_y", "ssim"):
        value = (payload.get("scores") or {}).get(key)
        if not isinstance(value, (int, float)):
            alarms.append(f"{name}: missing numeric score {key}")
            continue
        low, high = _point_bounds(name, bounds)[key]
        if not low <= float(value) <= high:
            alarms.append(f"{name}: {key}={value} outside [{low},{high}]")
    low, high = _point_bounds(name, bounds)["coded_bytes"]
    coded = payload.get("coded_bytes")
    if not isinstance(coded, int) or not low <= coded <= high:
        alarms.append(f"{name}: coded_bytes={coded} outside [{low},{high}]")
    if payload.get("recovery_alarm") is not None:
        alarms.append(f"{name}: {payload['recovery_alarm']}")
    alarms.extend(
        f"{name}: {item}"
        for item in ((payload.get("late_frame") or {}).get("alarms") or [])
    )
    headers = [
        item.get("geometry_header_bytes")
        for item in (payload.get("background_payloads") or [])
    ]
    if headers and any(item != HEADER_BYTES for item in headers):
        alarms.append(f"{name}: geometry header bytes {headers} != {HEADER_BYTES}")
    return alarms


def _bp52_crf51_payload() -> dict[str, Any] | None:
    path = ps_paths.outputs() / "bp52-background-crf" / "background-search.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    for row in document.get("points") or []:
        if row.get("name") == "bg-crf51":
            payload = row.get("pointstream")
            return payload if isinstance(payload, dict) else None
    return None


def _control_alarms(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return [f"{CONTROL_NAME}: missing PointStream result for the realtime control"]
    historical = _bp52_crf51_payload()
    if historical is None:
        historical = {
            "coded_bytes": BP52_CRF51["coded_bytes"],
            "scores": {
                "vmaf": BP52_CRF51["vmaf"],
                "psnr_y": BP52_CRF51["psnr_y"],
                "ssim": BP52_CRF51["ssim"],
            },
            "parts": {
                "panorama": BP52_CRF51["panorama"],
                "actor_reference": BP52_CRF51["actor_reference"],
                "metadata": BP52_CRF51["metadata"],
                "residual": BP52_CRF51["residual"],
            },
        }
    alarms: list[str] = []
    scores = payload.get("scores") or {}
    old_scores = historical.get("scores") or {}
    parts = payload.get("parts") or {}
    old_parts = historical.get("parts") or {}
    for key in ("vmaf", "psnr_y", "ssim"):
        got = scores.get(key)
        expected = old_scores.get(key)
        if (
            not isinstance(got, (int, float))
            or not isinstance(expected, (int, float))
            or float(got) != float(expected)
        ):
            alarms.append(f"{CONTROL_NAME}: {key}={got} does not reproduce BP52 {expected}")
    if _as_int(parts, "panorama") != _as_int(old_parts, "panorama"):
        alarms.append(
            f"{CONTROL_NAME}: panorama={parts.get('panorama')} "
            f"does not reproduce BP52 {old_parts.get('panorama')}"
        )
    if _as_int(parts, "actor_reference") != _as_int(old_parts, "actor_reference"):
        alarms.append(f"{CONTROL_NAME}: actor_reference does not reproduce BP52")
    if _as_int(parts, "residual") != _as_int(old_parts, "residual"):
        alarms.append(
            f"{CONTROL_NAME}: residual={parts.get('residual')} "
            f"does not reproduce BP52 {old_parts.get('residual')}"
        )
    old_meta = _as_int(old_parts, "metadata")
    expected_meta = None if old_meta is None else old_meta + 2 * HEADER_BYTES
    got_meta = _as_int(parts, "metadata")
    if got_meta != expected_meta:
        alarms.append(
            f"{CONTROL_NAME}: metadata={parts.get('metadata')} != "
            f"BP52 metadata + 2*{HEADER_BYTES}={expected_meta}"
        )
    old_total = _as_int(historical, "coded_bytes")
    expected_total = None if old_total is None else old_total + 2 * HEADER_BYTES
    if _as_int(payload, "coded_bytes") != expected_total:
        alarms.append(
            f"{CONTROL_NAME}: coded_bytes={payload.get('coded_bytes')} != "
            f"BP52 coded_bytes + 2*{HEADER_BYTES}={expected_total}"
        )
    return alarms


def _file_digest(root: Path, relative: str) -> str:
    path = root / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_code_identity(root: Path) -> dict[str, str]:
    return {name: _file_digest(root, name) for name in METRIC_PATHS}


def _bp52_search() -> dict[str, Any] | None:
    path = ps_paths.outputs() / "bp52-background-crf" / "background-search.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _reference_citation(tools: dict[str, Any], source: list[dict[str, Any]]) -> dict[str, Any]:
    """Cite BP52 AV1 QP63 / VVC QP51/QP39 only when identities match."""
    document = _bp52_search()
    reasons: list[str] = []
    source_ok = [item.get("sha256") for item in source] == list(EXPECTED_SOURCE)
    if not source_ok:
        reasons.append("source RGB hashes do not match BP49/BP52")
    ffmpeg_ok = bool(tools.get("matches_bp52_ffmpeg"))
    if not ffmpeg_ok:
        reasons.append("background ffmpeg path/version does not match BP52")
    preset = tools.get("reference_preset")
    bp52_preset = None if document is None else (document.get("input") or {}).get("preset")
    if document is None:
        reasons.append("BP52 background-search.json is missing")
    elif str(preset) != str(bp52_preset):
        reasons.append(
            f"reference preset {preset!r} != BP52 {bp52_preset!r}"
        )
    color_ok = tools.get("reference_pix_fmt") == "yuv420p"
    if not color_ok:
        reasons.append("reference pix_fmt is not yuv420p")
    metric_ok = bool(tools.get("metric_code_unchanged_from_origin_main"))
    if not metric_ok:
        reasons.append("metric/scoring files differ from origin/main")
    comparable = not reasons
    cited: dict[str, Any] = {}
    if document is not None:
        for row in document.get("points") or []:
            payload = row.get("pointstream") or {}
            for item in payload.get("references") or []:
                name = str(item.get("name") or "")
                if name in {"av1-qp63", "vvc-qp51", "vvc-qp39"}:
                    cited[name] = {
                        "coded_bytes": item.get("coded_bytes") or item.get("bytes"),
                        "scores": item.get("scores"),
                    }
        if not cited:
            outcome = document.get("outcome") or {}
            for key in ("av1_qp63", "vvc_qp51", "vvc_qp39"):
                if key in outcome:
                    cited[key] = outcome[key]
    return {
        "status": "cited-immutable" if comparable else "unranked-diagnostics",
        "comparable": comparable,
        "reasons": reasons,
        "path": str(ps_paths.outputs() / "bp52-background-crf" / "background-search.json"),
        "cited": cited if comparable else {},
        "note": (
            "BP52 continuous AV1 QP63 and VVC QP51/QP39 are cited only after "
            "source, ffmpeg, preset, color and metric-code checks. "
            "No extra reference encodes were run."
            if comparable
            else "Verification failed; references are not ranked against these points."
        ),
    }


def _tool_identity(root: Path, preset: str) -> dict[str, Any]:
    ffmpeg = ffmpeg_provenance()
    comparable = (
        ffmpeg.get("path") == BP52_FFMPEG["path"]
        and str(ffmpeg.get("version") or "").startswith(BP52_FFMPEG["version_prefix"])
    )
    metric_files = _metric_code_identity(root)
    origin_ok = True
    origin_errors: list[str] = []
    try:
        import subprocess

        for name in METRIC_PATHS:
            current = (root / name).read_bytes()
            historic = subprocess.check_output(
                ["git", "show", f"origin/main:{name}"], cwd=root
            )
            if current != historic:
                origin_ok = False
                origin_errors.append(name)
    except Exception as exc:  # noqa: BLE001
        origin_ok = False
        origin_errors.append(repr(exc))
    return {
        "background_ffmpeg": ffmpeg,
        "background_stream_codec_default": stream_codec_provenance("av1"),
        "background_stream_codec_candidate": stream_codec_provenance(
            "av1",
            usage=CANDIDATE_STREAM_USAGE,
            cpu_used=CANDIDATE_STREAM_CPU_USED,
        ),
        "matches_bp52_ffmpeg": comparable,
        "reference_preset": preset,
        "reference_pix_fmt": "yuv420p",
        "metric_code": metric_files,
        "metric_code_unchanged_from_origin_main": origin_ok,
        "metric_code_differs": origin_errors,
        "background_command_template_default": (
            "ffmpeg ... -c:v libaom-av1 -crf {stream_crf} "
            "-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0 ..."
        ),
        "background_command_template_candidate": (
            "ffmpeg ... -c:v libaom-av1 -crf {stream_crf} "
            "-cpu-used 4 -usage good -lag-in-frames 0 -bf 0 ..."
        ),
        "encode_seconds": None,
        "decode_seconds": None,
        "timing_note": (
            "Semantic encoder/client clocks remain null. Background "
            "ffmpeg encode/decode seconds are recorded per chain when available. "
            "Do not infer them by subtracting metrics."
        ),
    }


def _synthetic_frames() -> dict[str, list[np.ndarray]]:
    rng = np.random.default_rng(56)
    height, width = 48, 64
    canvas = np.full((height, width + 16, 3), 36, dtype=np.uint8)
    for _ in range(14):
        top = int(rng.integers(0, height - 10))
        left = int(rng.integers(0, canvas.shape[1] - 14))
        canvas[top : top + 10, left : left + 14] = rng.integers(50, 240, 3, dtype=np.uint8)
    textured = [np.ascontiguousarray(canvas[:, k * 3 : k * 3 + width]) for k in range(4)]
    static = [np.ascontiguousarray(textured[0].copy()) for _ in range(4)]
    translated = [
        np.ascontiguousarray(np.roll(textured[0], shift * 2, axis=1)) for shift in range(4)
    ]
    return {"textured": textured, "static": static, "translated": translated}


def run_prefix_proof(destination: Path) -> dict[str, Any]:
    probe = probe_stream_effort(
        usage=CANDIDATE_STREAM_USAGE, cpu_used=CANDIDATE_STREAM_CPU_USED
    )
    if probe.get("supported") is not True:
        raise SystemExit(
            f"candidate libaom effort is unsupported: {probe.get('error')}"
        )
    families = _synthetic_frames()
    proofs: dict[str, Any] = {}
    for name, frames in families.items():
        payloads = independent_prefix_payloads(
            frames,
            usage=CANDIDATE_STREAM_USAGE,
            cpu_used=CANDIDATE_STREAM_CPU_USED,
            crf=51,
        )
        assert_independent_prefixes_stable(payloads)
        proofs[name] = {
            "payload_sizes": {
                str(count): [len(item) for item in blobs]
                for count, blobs in payloads.items()
            },
            "last_command": last_encode_record().get("argv"),
        }
    document = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probe": probe,
        "families": proofs,
        "candidate": {
            "usage": CANDIDATE_STREAM_USAGE,
            "cpu_used": CANDIDATE_STREAM_CPU_USED,
        },
    }
    write_json(destination / "prefix-proof.json", document)
    return document


def _report(
    *,
    identity: dict[str, Any],
    bounds: dict[str, Any],
    controls: dict[str, Any],
    rows: list[dict[str, Any]],
    alarms: list[str],
    destination: Path,
    points_dir: Path,
    tools: dict[str, Any],
    prefix_proof: dict[str, Any] | None,
    budget: dict[str, Any] | None = None,
    capability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    citation = _reference_citation(tools, identity.get("source") or [])
    planned = len(POINT_SPECS)
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": "complete" if len(rows) == planned and not alarms else "partial",
        "experiment": "bp56-background-encoder-effort",
        "input": identity,
        "implementation_frozen_before_measurement": True,
        "configuration": {
            "background_method": "panorama-stream",
            "transport_scale": 1.0,
            "points": [
                {
                    "name": name,
                    "stream_usage": usage,
                    "stream_cpu_used": cpu,
                    "stream_crf": named_point(crf).stream_crf,
                }
                for name, usage, cpu, crf in POINT_SPECS
            ],
            "appearance_jpeg_quality": 40,
            "appearance_downscale": 2,
            "motion_max_points": 16,
            "canonical_canvas": True,
            "generation": False,
            "residual": False,
            "geometry_header_bytes": HEADER_BYTES,
        },
        "capability": capability,
        "prefix_proof": prefix_proof,
        "tool_identity": tools,
        "bp52_references": citation,
        "bounds": bounds,
        "metric_controls": controls,
        "checkpoint_dir": str(points_dir),
        "output_dir": str(destination),
        "points": rows,
        "completion": completion_counts(rows),
        "alarms": alarms,
        "budget": budget,
        "reproduction": (
            "PYTHONPATH=/home/itec/emanuele/pointstream-bp56 "
            "PS_DATA_ROOT=/home/itec/emanuele/pointstream-data "
            "python -m experiments.tier.bp56_background_effort"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_028"])
    parser.add_argument("--frames", type=int, default=DEFAULT_SPAN_FRAMES)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args(argv)
    if args.video != DEFAULT_VIDEO or list(args.scenes) != ["scene_000", "scene_028"]:
        raise SystemExit("BP56 is fixed to alcaraz_highlights scene_000 scene_028")
    if args.frames != DEFAULT_SPAN_FRAMES:
        raise SystemExit(f"BP56 is fixed to {DEFAULT_SPAN_FRAMES} frames per scene")

    destination = (
        Path(args.out_dir) if args.out_dir else ps_paths.outputs() / "bp56-background-effort"
    )
    if any(token in str(destination) for token in ("bp49", "bp52", "bp53")):
        raise SystemExit("refusing to write into a BP49/BP52/BP53 output path")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    points_dir = destination / "points"
    leftover = [
        path
        for path in destination.iterdir()
        if path.name not in {
            "logs",
            "prefix-proof.json",
            "bounds-before-run.json",
            "metric-controls.json",
            "background-effort.json",
            "budget.json",
        }
    ]
    if leftover and not (points_dir / "identity.json").is_file() and not args.prefix_only:
        raise SystemExit(
            f"{destination} already contains an unverified output identity; "
            "choose a new documented suffix"
        )
    root = Path(__file__).resolve().parents[2]
    prefix_proof = run_prefix_proof(destination)
    capability = prefix_proof.get("probe")
    if args.prefix_only:
        return 0

    bounds_path = destination / "bounds-before-run.json"
    if bounds_path.exists():
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    else:
        bounds = bounds_document()
        write_json(bounds_path, bounds)

    clips = load_e1_sequence(args.video, list(args.scenes), n_frames=args.frames)
    source = _verify_input(clips)
    if source[0]["context_id"] != EXPECTED_CONTEXT or source[0]["shape"] != EXPECTED_SHAPE:
        raise SystemExit("BP56 input identity drifted from BP52")
    manifest = _manifest_snapshot(args.video, list(args.scenes))
    from src.runner.config_io import load_tier

    base: PointstreamConfig = load_tier("balanced")
    preset = primary_preset("av1")
    tools = _tool_identity(root, preset)
    identity: dict[str, Any] = {
        "video": args.video,
        "scenes": list(args.scenes),
        "frames_per_scene": args.frames,
        "fps": DECLARED_FPS,
        "codec": "av1",
        "source": source,
        "manifest": manifest,
        "preset": preset,
        "implementation": implementation_digest(root),
        "points": [
            {
                "name": name,
                "stream_usage": usage,
                "stream_cpu_used": cpu,
                "stream_crf": named_point(crf).stream_crf,
                "transport_scale": 1.0,
                "lag_in_frames": 0,
                "bf": 0,
            }
            for name, usage, cpu, crf in POINT_SPECS
        ],
        "bounds": fingerprint(bounds),
        "base_config": fingerprint(base),
        "header_bytes": HEADER_BYTES,
        "expected_sources": list(EXPECTED_SOURCE),
        "metric_code": tools["metric_code"],
        "background_effort_default": {
            "usage": DEFAULT_STREAM_USAGE,
            "cpu_used": DEFAULT_STREAM_CPU_USED,
        },
        "background_effort_candidate": {
            "usage": CANDIDATE_STREAM_USAGE,
            "cpu_used": CANDIDATE_STREAM_CPU_USED,
        },
    }
    guard_checkpoints(points_dir, identity)
    budget_path = destination / "budget.json"
    recover_interrupted(budget_path, points_dir)
    reconcile_checkpoints(budget_path, points_dir, [name for name, _, _, _ in POINT_SPECS])
    controls_path = destination / "metric-controls.json"
    report_path = destination / "background-effort.json"
    rows: list[dict[str, Any]] = []
    alarms: list[str] = []

    def emit() -> dict[str, Any]:
        snapshot = load_budget(budget_path)
        alarms.extend(item for item in budget_alarms(snapshot) if item not in alarms)
        controls_payload: dict[str, Any]
        if controls_path.is_file():
            controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
        else:
            controls_payload = {"valid": False, "alarms": ["controls not written"]}
        document = _report(
            identity=identity,
            bounds=bounds,
            controls=controls_payload,
            rows=rows,
            alarms=alarms,
            destination=destination,
            points_dir=points_dir,
            tools=tools,
            prefix_proof=prefix_proof,
            budget=snapshot,
            capability=capability,
        )
        write_json(report_path, document)
        return document

    recovered = load_budget(budget_path)
    if recovered.get("unknown_crash_interval") is True:
        alarms.append(
            "unknown crash interval after last heartbeat; budget compliance "
            "is unresolved; stopping for review"
        )
        emit()
        return 1

    if controls_path.is_file():
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
    else:
        with AttemptSession(
            budget_path,
            "metric-controls",
            kind="controls",
            interval_s=HEARTBEAT_INTERVAL_S,
        ) as session:
            controls = run_metric_controls(
                np.asarray(clips[0].frames[:2]), controls_path
            )
            control_wall = session.elapsed()
        finish_attempt(
            budget_path, "metric-controls", control_wall, kind="controls"
        )
    alarms.extend(list(controls.get("alarms") or []))
    if controls.get("valid") is not True:
        emit()
        return 1

    for name, usage, cpu_used, plan_name in POINT_SPECS:
        budget = load_budget(budget_path)
        remaining = remaining_seconds(budget)
        existing = load_checkpoint(points_dir, name)
        if existing is None and (remaining < POINT_RESERVE_S or over_budget(budget)):
            alarms.append(
                f"{name}: not started; remaining {remaining:.0f}s "
                f"(reserve {POINT_RESERVE_S}s, over_budget={over_budget(budget)})"
            )
            emit()
            return 1
        if existing is not None:
            row = existing
            print(f"resume {name}", flush=True)
        else:
            point = named_point(plan_name)
            tuned = apply_point(
                base,
                point,
                codec="av1",
                preset=preset,
                context_id=source[0]["context_id"],
            )
            tuned = replace(
                tuned,
                background=replace(
                    tuned.background,
                    transport_scale=1.0,
                    stream_usage=usage,
                    stream_cpu_used=cpu_used,
                ),
            )
            row = {
                "name": name,
                "config": {
                    "transport_scale": 1.0,
                    "stream_crf": point.stream_crf,
                    "stream_usage": usage,
                    "stream_cpu_used": cpu_used,
                    "background_method": point.background_method,
                    "lag_in_frames": 0,
                    "bf": 0,
                },
            }
            encode_timeout = min(
                POINT_ENCODE_TIMEOUT_S,
                max(1.0, remaining - SCORING_RESERVE_S),
            )
            with AttemptSession(
                budget_path,
                name,
                kind="attempt",
                interval_s=HEARTBEAT_INTERVAL_S,
            ) as session:
                try:
                    with ffmpeg_timeout(encode_timeout):
                        row["pointstream"] = pointstream_e1(
                            clips, tuned, checkpoint_dir=points_dir / f"{name}.run"
                        )
                except Exception as exc:  # noqa: BLE001
                    row["pointstream_error"] = repr(exc)
                row["attempt_wall_seconds"] = round(session.elapsed(), 3)
                row["ffmpeg_timeout_seconds"] = encode_timeout
            save_checkpoint(points_dir, name, row)
            finish_attempt(budget_path, name, float(row["attempt_wall_seconds"]))
        rows.append(row)
        alarms.extend(_point_alarms(name, row, bounds))
        payload = row.get("pointstream") or {}
        gap = payload.get("max_checkpoint_gap_seconds")
        if isinstance(gap, (int, float)) and not longer_runs_cleared(float(gap)):
            budget = load_budget(budget_path)
            budget["longer_runs_operationally_cleared"] = False
            budget["max_checkpoint_gap_seconds"] = float(gap)
            budget["hourly_note"] = (
                f"{name} max checkpoint gap {float(gap):.3f}s is within 1s "
                "of the hourly limit; longer runs are not cleared"
            )
            write_json(budget_path, budget)
        if name == CONTROL_NAME:
            alarms.extend(_control_alarms(row.get("pointstream")))
        emit()
        if alarms:
            print(f"stopping batch after {name}: {alarms}", flush=True)
            return 1
        print(
            f"{name}: {row.get('pointstream', {}).get('coded_bytes', 'FAILED')} B",
            flush=True,
        )

    final = emit()
    print(f"wrote {report_path}", flush=True)
    counts = completion_counts(rows)
    return 1 if alarms or counts["failed"] or final["outcome"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
