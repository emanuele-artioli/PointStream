"""Bounded BP53 background-transport scaling diagnostic.

Exactly three PointStream points on the BP49/BP52 pair:
scale 1.0 CRF51, scale 0.5 CRF51, scale 0.5 CRF63.
Never writes BP49 or BP52 output directories.
"""

from __future__ import annotations

import argparse
import json
import time
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
from src.components.background.stream import ffmpeg_provenance
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig


POINT_SPECS: tuple[tuple[str, float, str], ...] = (
    ("bg-scale1-crf51", 1.0, "bg-crf51"),
    ("bg-scale05-crf51", 0.5, "bg-crf51"),
    ("bg-scale05-crf63", 0.5, "bg-crf63"),
)
BP52_CRF51 = {
    "coded_bytes": 474313,
    "vmaf": 77.417052,
    "psnr_y": 33.003064,
    "ssim": 0.96694254,
    "panorama": 445513,
    "actor_reference": 8599,
    "metadata": 20201,
    "residual": 0,
}
BP52_FFMPEG = {
    "path": "/opt/local/bin/ffmpeg",
    "version_prefix": "ffmpeg version n7.1.1-56-gc2184b65d2",
}
WALL_BUDGET_S = 8 * 3600
POINT_RESERVE_S = 3500


def bounds_document() -> dict[str, Any]:
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "bp53-background-transport-scale",
        "result_read_after": "bounds and controls",
        "basis": {
            "scale1_crf51": (
                "Reproduce BP52 CRF51 quality exactly. Panorama and actor bytes "
                "must match. Metadata may grow by exactly 2*HEADER_BYTES of "
                "charged geometry headers. Run 30--10,800 s."
            ),
            "half_scale": (
                "Broad diagnostic bands, not a pixel-ratio prediction: VMAF 0--98, "
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
            "point_count": 3,
            "checkpoint_gap_seconds_max": 3600.0,
            "geometry_header_bytes_per_scene": HEADER_BYTES,
        },
        "points": {
            "bg-scale1-crf51": {
                "coded_bytes": [80000, 50000000],
                "vmaf": [15.0, 97.0],
                "psnr_y": [16.0, 45.0],
                "ssim": [0.72, 0.995],
                "run_seconds": [30.0, 10800.0],
            },
            "bg-scale05-crf51": {
                "coded_bytes": [1, 50000000],
                "vmaf": [0.0, 98.0],
                "psnr_y": [8.0, 45.0],
                "ssim": [0.0, 1.0],
            },
            "bg-scale05-crf63": {
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
        return ["bg-scale1-crf51: missing PointStream result for the scale-1 control"]
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
        if not isinstance(got, (int, float)) or not isinstance(expected, (int, float)) or float(got) != float(expected):
            alarms.append(f"bg-scale1-crf51: {key}={got} does not reproduce BP52 {expected}")
    if int(parts.get("panorama") or -1) != int(old_parts.get("panorama") or -2):
        alarms.append(
            f"bg-scale1-crf51: panorama={parts.get('panorama')} "
            f"does not reproduce BP52 {old_parts.get('panorama')}"
        )
    if int(parts.get("actor_reference") or -1) != int(old_parts.get("actor_reference") or -2):
        alarms.append("bg-scale1-crf51: actor_reference does not reproduce BP52")
    if int(parts.get("residual") or -1) != int(old_parts.get("residual") or 0):
        alarms.append("bg-scale1-crf51: residual is not zero")
    expected_meta = int(old_parts.get("metadata") or BP52_CRF51["metadata"]) + 2 * HEADER_BYTES
    if int(parts.get("metadata") or -1) != expected_meta:
        alarms.append(
            f"bg-scale1-crf51: metadata={parts.get('metadata')} != "
            f"BP52 metadata + 2*{HEADER_BYTES}={expected_meta}"
        )
    expected_total = int(historical.get("coded_bytes") or BP52_CRF51["coded_bytes"]) + 2 * HEADER_BYTES
    if int(payload.get("coded_bytes") or -1) != expected_total:
        alarms.append(
            f"bg-scale1-crf51: coded_bytes={payload.get('coded_bytes')} != "
            f"BP52 coded_bytes + 2*{HEADER_BYTES}={expected_total}"
        )
    return alarms


def _tool_identity() -> dict[str, Any]:
    ffmpeg = ffmpeg_provenance()
    comparable = (
        ffmpeg.get("path") == BP52_FFMPEG["path"]
        and str(ffmpeg.get("version") or "").startswith(BP52_FFMPEG["version_prefix"])
    )
    return {
        "background_ffmpeg": ffmpeg,
        "background_stream_codec": stream_codec_provenance("av1"),
        "matches_bp52_ffmpeg": comparable,
        "background_command_template": (
            "ffmpeg ... -c:v libaom-av1 -crf {stream_crf} "
            "-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0 ..."
        ),
        "downsample_upsample_seconds": None,
        "downsample_upsample_reason": (
            "resample time is recorded on the live model during transmit/decode "
            "but is not a runner stage clock; left null rather than relabelled"
        ),
        "encode_seconds": None,
        "decode_seconds": None,
        "timing_note": (
            "Separate semantic encoder/client clocks remain Codex's "
            "instrumentation task. Runner wall includes reconstruction and scoring."
        ),
    }


def _bp52_references(comparable: bool) -> dict[str, Any]:
    root = ps_paths.outputs() / "bp52-background-crf"
    search = root / "background-search.json"
    if not search.is_file():
        return {"status": "missing", "path": str(search), "comparable": False}
    document = json.loads(search.read_text(encoding="utf-8"))
    return {
        "status": "cited-immutable" if comparable else "not-comparable",
        "path": str(search),
        "comparable": comparable,
        "note": (
            "BP52 continuous AV1 QP63 and VVC QP51/QP39 are cited as prior "
            "diagnostics only. They are not copied into this checkpoint identity."
            if comparable
            else "ffmpeg path/version does not match BP52; do not compare those JSONs."
        ),
        "bp52_outcome": document.get("outcome"),
    }


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
) -> dict[str, Any]:
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": "complete" if len(rows) == len(POINT_SPECS) and not alarms else "partial",
        "experiment": "bp53-background-transport-scale",
        "input": identity,
        "implementation_frozen_before_measurement": True,
        "configuration": {
            "background_method": "panorama-stream",
            "points": [
                {"name": name, "transport_scale": scale, "stream_crf": named_point(crf).stream_crf}
                for name, scale, crf in POINT_SPECS
            ],
            "appearance_jpeg_quality": 40,
            "appearance_downscale": 2,
            "motion_max_points": 16,
            "canonical_canvas": True,
            "generation": False,
            "residual": False,
            "geometry_header_bytes": HEADER_BYTES,
        },
        "tool_identity": tools,
        "bp52_references": _bp52_references(bool(tools.get("matches_bp52_ffmpeg"))),
        "bounds": bounds,
        "metric_controls": controls,
        "checkpoint_dir": str(points_dir),
        "output_dir": str(destination),
        "points": rows,
        "completion": completion_counts(rows),
        "alarms": alarms,
        "reproduction": (
            "PYTHONPATH=/home/itec/emanuele/pointstream-bp53 "
            "PS_DATA_ROOT=/home/itec/emanuele/pointstream-data "
            "python -m experiments.tier.bp53_background_scale"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_028"])
    parser.add_argument("--frames", type=int, default=DEFAULT_SPAN_FRAMES)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.video != DEFAULT_VIDEO or list(args.scenes) != ["scene_000", "scene_028"]:
        raise SystemExit("BP53 is fixed to alcaraz_highlights scene_000 scene_028")
    if args.frames != DEFAULT_SPAN_FRAMES:
        raise SystemExit(f"BP53 is fixed to {DEFAULT_SPAN_FRAMES} frames per scene")

    destination = Path(args.out_dir) if args.out_dir else ps_paths.outputs() / "bp53-background-scale"
    if "bp49" in str(destination) or "bp52" in str(destination):
        raise SystemExit("refusing to write into a BP49/BP52 output path")
    points_dir = destination / "points"
    if (
        destination.exists()
        and any(destination.iterdir())
        and not (points_dir / "identity.json").is_file()
    ):
        raise SystemExit(
            f"{destination} already contains an unverified output identity; "
            "choose a new documented suffix"
        )
    destination.mkdir(parents=True, exist_ok=True)
    bounds_path = destination / "bounds-before-run.json"
    if bounds_path.exists():
        bounds = json.loads(bounds_path.read_text(encoding="utf-8"))
    else:
        bounds = bounds_document()
        write_json(bounds_path, bounds)

    clips = load_e1_sequence(args.video, list(args.scenes), n_frames=args.frames)
    source = _verify_input(clips)
    if source[0]["context_id"] != EXPECTED_CONTEXT or source[0]["shape"] != EXPECTED_SHAPE:
        raise SystemExit("BP53 input identity drifted from BP52")
    manifest = _manifest_snapshot(args.video, list(args.scenes))
    from src.runner.config_io import load_tier

    base: PointstreamConfig = load_tier("balanced")
    preset = primary_preset("av1")
    tools = _tool_identity()
    identity: dict[str, Any] = {
        "video": args.video,
        "scenes": list(args.scenes),
        "frames_per_scene": args.frames,
        "fps": DECLARED_FPS,
        "codec": "av1",
        "source": source,
        "manifest": manifest,
        "preset": preset,
        "implementation": implementation_digest(),
        "points": [name for name, _, _ in POINT_SPECS],
        "bounds": fingerprint(bounds),
        "base_config": fingerprint(base),
        "header_bytes": HEADER_BYTES,
        "expected_sources": list(EXPECTED_SOURCE),
    }
    guard_checkpoints(points_dir, identity)
    controls_path = destination / "metric-controls.json"
    controls = (
        json.loads(controls_path.read_text(encoding="utf-8"))
        if controls_path.is_file()
        else run_metric_controls(np.asarray(clips[0].frames[:2]), controls_path)
    )
    report_path = destination / "background-scale.json"
    rows: list[dict[str, Any]] = []
    alarms = list(controls.get("alarms") or [])
    started = time.monotonic()
    if controls.get("valid") is not True:
        report = _report(
            identity=identity,
            bounds=bounds,
            controls=controls,
            rows=rows,
            alarms=alarms,
            destination=destination,
            points_dir=points_dir,
            tools=tools,
        )
        write_json(report_path, report)
        return 1

    for name, scale, plan_name in POINT_SPECS:
        remaining = WALL_BUDGET_S - (time.monotonic() - started)
        existing = load_checkpoint(points_dir, name)
        if existing is None and remaining < POINT_RESERVE_S:
            alarms.append(f"{name}: not started; remaining {remaining:.0f}s < {POINT_RESERVE_S}s")
            report = _report(
                identity=identity,
                bounds=bounds,
                controls=controls,
                rows=rows,
                alarms=alarms,
                destination=destination,
                points_dir=points_dir,
                tools=tools,
            )
            write_json(report_path, report)
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
                background=replace(tuned.background, transport_scale=float(scale)),
            )
            wall_started = time.perf_counter()
            row = {
                "name": name,
                "config": {
                    "transport_scale": scale,
                    "stream_crf": point.stream_crf,
                    "background_method": point.background_method,
                },
            }
            try:
                row["pointstream"] = pointstream_e1(
                    clips, tuned, checkpoint_dir=points_dir / f"{name}.run"
                )
            except Exception as exc:  # noqa: BLE001
                row["pointstream_error"] = repr(exc)
            row["attempt_wall_seconds"] = round(time.perf_counter() - wall_started, 3)
            save_checkpoint(points_dir, name, row)
        rows.append(row)
        alarms.extend(_point_alarms(name, row, bounds))
        if name == "bg-scale1-crf51":
            alarms.extend(_control_alarms(row.get("pointstream")))
        report = _report(
            identity=identity,
            bounds=bounds,
            controls=controls,
            rows=rows,
            alarms=alarms,
            destination=destination,
            points_dir=points_dir,
            tools=tools,
        )
        write_json(report_path, report)
        if alarms:
            print(f"stopping batch after {name}: {alarms}", flush=True)
            return 1
        print(
            f"{name}: {row.get('pointstream', {}).get('coded_bytes', 'FAILED')} B",
            flush=True,
        )

    final = _report(
        identity=identity,
        bounds=bounds,
        controls=controls,
        rows=rows,
        alarms=alarms,
        destination=destination,
        points_dir=points_dir,
        tools=tools,
    )
    write_json(report_path, final)
    print(f"wrote {report_path}", flush=True)
    counts = completion_counts(rows)
    return 1 if alarms or counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
