"""Bounded BP52 background-CRF search on the exact BP49 diagnostic pair.

This is deliberately a small batch, not the general low-rate ladder:
``bg-crf51`` is a fresh control followed by ``bg-crf63`` and ``bg-crf57``.
All three points keep the BP49 object stream and canonical background settings.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.calibrate import calibrate
from experiments.long_scenes.loader import MANIFEST_PATH, get_long_scene_manifest
from experiments.tier.low_rate_checkpoint import (
    completion_counts,
    fingerprint,
    guard_checkpoints,
    implementation_digest,
    load_checkpoint,
    save_checkpoint,
    source_identity,
    write_json,
)
from experiments.tier.low_rate_clips import (
    DEFAULT_SPAN_FRAMES,
    DEFAULT_VIDEO,
    load_e1_sequence,
)
from experiments.tier.low_rate_measure import (
    primary_preset,
    stream_codec_provenance,
)
from experiments.tier.low_rate_plan import SweepPoint, named_point
from experiments.tier.low_rate_sweep import run_point
from experiments.tier.low_rate_validate import DECLARED_FPS
from src.components.background.stream import ffmpeg_provenance
from src.contracts import paths as ps_paths
from src.contracts.config import PointstreamConfig


POINT_NAMES: tuple[str, ...] = ("bg-crf51", "bg-crf63", "bg-crf57")
EXPECTED_SOURCE: tuple[str, ...] = (
    "388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6",
    "e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb",
)
EXPECTED_CONTEXT = "alcaraz_highlights_main_court"
EXPECTED_SHAPE = [48, 2160, 3840, 3]


def bounds_document() -> dict[str, Any]:
    """Return the pre-result alarms for this bounded search."""
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": "bp52-background-crf-search",
        "result_read_after": "bounds and controls",
        "basis": {
            "crf51": (
                "Carry BP49's native-resolution bands: 96 frames, 3840x2160, "
                "80,000--50,000,000 coded bytes, VMAF 15--97, Y-PSNR 16--45 dB, "
                "SSIM 0.72--0.995, run 30--10,800 s."
            ),
            "crf63_crf57": (
                "The stronger-degradation diagnostic bands are deliberately "
                "broad and independent of BP49: VMAF 0--98, Y-PSNR 8--45 dB, "
                "SSIM 0--1, and positive coded bytes below 50 MB."
            ),
            "late_frame": (
                "Carry the scene-local last-minus-first bands from BP49: VMAF "
                "[-25,+8] and Y-PSNR [-8,+3] dB. A joined-scene delta is not a "
                "scene drift measure."
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
        },
        "points": {
            "bg-crf51": {
                "coded_bytes": [80000, 50000000],
                "vmaf": [15.0, 97.0],
                "psnr_y": [16.0, 45.0],
                "ssim": [0.72, 0.995],
                "run_seconds": [30.0, 10800.0],
            },
            "bg-crf63": {
                "coded_bytes": [1, 50000000],
                "vmaf": [0.0, 98.0],
                "psnr_y": [8.0, 45.0],
                "ssim": [0.0, 1.0],
            },
            "bg-crf57": {
                "coded_bytes": [1, 50000000],
                "vmaf": [0.0, 98.0],
                "psnr_y": [8.0, 45.0],
                "ssim": [0.0, 1.0],
            },
        },
        "late_frame": {
            "vmaf": [-25.0, 8.0],
            "psnr_y": [-8.0, 3.0],
        },
    }


def _as_float(value: object) -> float:
    if value == "inf":
        return float("inf")
    return float(str(value))


def run_metric_controls(reference: np.ndarray, destination: Path) -> dict[str, Any]:
    """Run the existing native-resolution calibration fixtures before points."""
    outcome = calibrate(["psnr", "ssim", "vmaf"], reference)
    table = outcome.get("metrics", {})
    required = ("identical", "mild-blur", "severe-blur", "unrelated-clip")
    alarms: list[str] = []
    for name in ("psnr", "ssim", "vmaf"):
        values = (table.get(name) or {}).get("by_anchor") or {}
        missing = [anchor_name for anchor_name in required if anchor_name not in values]
        if missing:
            alarms.append(f"{name}: missing anchors {missing}")
            continue
        identical = _as_float(values["identical"])
        mild = _as_float(values["mild-blur"])
        severe = _as_float(values["severe-blur"])
        unrelated = _as_float(values["unrelated-clip"])
        if not identical > mild > severe:
            alarms.append(
                f"{name}: required identical > mild > severe failed "
                f"({identical}, {mild}, {severe})"
            )
        if not mild > unrelated:
            alarms.append(
                f"{name}: required mild > unrelated failed ({mild}, {unrelated})"
            )
    vmaf = (table.get("vmaf") or {}).get("by_anchor") or {}
    if vmaf:
        if not 95.0 <= _as_float(vmaf["identical"]) <= 99.0:
            alarms.append(f"vmaf identical absolute scale outside [95,99]: {vmaf['identical']}")
        if not 0.0 <= _as_float(vmaf["unrelated-clip"]) <= 40.0:
            alarms.append(
                f"vmaf unrelated absolute scale outside [0,40]: {vmaf['unrelated-clip']}"
            )
    result = {
        **outcome,
        "fixture": "experiments.tier.calibrate.anchors",
        "reference_shape": list(reference.shape),
        "required_checks": [
            "identical > mild-blur > severe-blur for higher-is-better metrics",
            "mild-blur > unrelated-clip for higher-is-better metrics",
            "VMAF identical absolute scale [95,99]",
            "VMAF unrelated absolute scale [0,40]",
        ],
        "severe_vs_unrelated_order_required": False,
        "alarms": alarms,
        "valid": not alarms,
    }
    write_json(destination, result)
    return result


def _verify_input(clips: list[Any]) -> list[dict[str, Any]]:
    found = source_identity(clips)
    if len(found) != len(EXPECTED_SOURCE):
        raise SystemExit(f"expected two BP49 scenes, loaded {len(found)}")
    for index, (record, expected_hash) in enumerate(zip(found, EXPECTED_SOURCE, strict=True)):
        if record["context_id"] != EXPECTED_CONTEXT:
            raise SystemExit(
                f"scene index {index} context {record['context_id']!r} "
                f"does not match {EXPECTED_CONTEXT!r}"
            )
        if record["shape"] != EXPECTED_SHAPE:
            raise SystemExit(
                f"scene index {index} shape {record['shape']} does not match "
                f"{EXPECTED_SHAPE}"
            )
        if record["sha256"] != expected_hash:
            raise SystemExit(
                f"scene index {index} RGB hash {record['sha256']} does not match "
                f"the BP49 frame identity {expected_hash}"
            )
    return found


def _manifest_snapshot(video: str, scenes: list[str]) -> dict[str, Any]:
    manifest = get_long_scene_manifest()
    selected = [
        record
        for record in manifest.get("scenes", [])
        if record.get("video") == video and record.get("scene") in scenes
    ]
    if len(selected) != len(scenes):
        raise SystemExit(
            f"BP46 manifest at {MANIFEST_PATH} has {len(selected)} requested "
            f"scene records, expected {len(scenes)}"
        )
    return {
        "path": str(MANIFEST_PATH),
        "selected_scene_records": selected,
        "selected_scene_records_sha256": fingerprint(selected),
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
    for key, expected in (("vmaf", "vmaf"), ("psnr_y", "psnr_y"), ("ssim", "ssim")):
        value = (payload.get("scores") or {}).get(key)
        if not isinstance(value, (int, float)):
            alarms.append(f"{name}: missing numeric score {key}")
            continue
        low, high = _point_bounds(name, bounds)[expected]
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
    return alarms


def _bp49_comparison(row: dict[str, Any]) -> dict[str, Any]:
    path = ps_paths.outputs() / "bp49-native-recovery" / (
        "sweep-alcaraz_highlights-scene_000+scene_028-n48-av1.json"
    )
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    historical = json.loads(path.read_text(encoding="utf-8"))
    old = next(
        (item.get("pointstream") for item in historical.get("rows", []) if item.get("name") == "bg-crf51"),
        None,
    )
    new = row.get("pointstream")
    if not old or not new:
        return {"status": "unavailable", "path": str(path)}
    fields = ("coded_bytes", "vmaf", "psnr_y", "ssim", "run_seconds")
    deltas: dict[str, float] = {}
    for field in fields:
        old_value = old.get(field) if field in {"coded_bytes", "run_seconds"} else (old.get("scores") or {}).get(field)
        new_value = new.get(field) if field in {"coded_bytes", "run_seconds"} else (new.get("scores") or {}).get(field)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            deltas[field] = float(new_value) - float(old_value)
    return {
        "status": "available",
        "path": str(path),
        "historical": {field: old.get(field) if field in {"coded_bytes", "run_seconds"} else (old.get("scores") or {}).get(field)
                       for field in fields},
        "fresh": {field: new.get(field) if field in {"coded_bytes", "run_seconds"} else (new.get("scores") or {}).get(field)
                  for field in fields},
        "delta_fresh_minus_historical": deltas,
        "note": "Any unexplained difference must be investigated before using CRF51 as a regression control.",
    }


def _control_alarms(comparison: dict[str, Any]) -> list[str]:
    """The fixed CRF51 control must reproduce size and quality, not host timing."""
    if comparison.get("status") != "available":
        return ["bg-crf51: historical regression control unavailable"]
    deltas = comparison.get("delta_fresh_minus_historical") or {}
    return [
        f"bg-crf51: regression control mismatch or missing {field}"
        for field in ("coded_bytes", "vmaf", "psnr_y", "ssim")
        if deltas.get(field) != 0.0
    ]


def _background_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {
        row["name"]: row.get("pointstream", {}).get("background_payloads", [])
        for row in rows
        if row.get("pointstream")
    }
    bytes_by_point = {
        name: [item.get("payload_bytes") for item in payloads]
        for name, payloads in by_name.items()
    }
    hashes_by_point = {
        name: [item.get("decoded_plate_sha256") for item in payloads]
        for name, payloads in by_name.items()
    }
    distinct_bytes = len(
        {tuple(values) for values in bytes_by_point.values() if values and None not in values}
    )
    distinct_hashes = len(
        {tuple(values) for values in hashes_by_point.values() if values and None not in values}
    )
    alarms: list[str] = []
    if len(by_name) == len(POINT_NAMES) and distinct_bytes < 2:
        alarms.append("background payload bytes did not change across CRF points")
    if len(by_name) == len(POINT_NAMES) and distinct_hashes < 2:
        alarms.append("decoded background plates did not change across CRF points")
    return {
        "payload_bytes_by_point": bytes_by_point,
        "decoded_plate_sha256_by_point": hashes_by_point,
        "distinct_payload_vectors": distinct_bytes,
        "distinct_decoded_plate_vectors": distinct_hashes,
        "requested_crf_order": [51, 63, 57],
        "alarms": alarms,
        "valid": not alarms,
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
) -> dict[str, Any]:
    return {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": "complete" if len(rows) == len(POINT_NAMES) and not alarms else "partial",
        "experiment": "bp52-background-crf-search",
        "input": identity,
        "implementation_frozen_before_measurement": True,
        "configuration": {
            "background_method": "panorama-stream",
            "background_encoder": stream_codec_provenance("av1"),
            "effective_crfs": [51, 63, 57],
            "appearance_jpeg_quality": 40,
            "appearance_downscale": 2,
            "motion_max_points": 16,
            "canonical_canvas": True,
            "generation": False,
            "residual": False,
            "full_resolution_delivery": True,
        },
        "tool_identity": {
            "background_ffmpeg": ffmpeg_provenance(),
            "background_command_template": (
                "ffmpeg ... -c:v libaom-av1 -crf {stream_crf} "
                "-cpu-used 8 -usage realtime -lag-in-frames 0 -bf 0 ..."
            ),
            "python": "recorded by the invoking conda pointstream environment",
        },
        "bounds": bounds,
        "metric_controls": controls,
        "checkpoint_dir": str(points_dir),
        "output_dir": str(destination),
        "points": rows,
        "completion": completion_counts(rows),
        "background_effect": _background_effect(rows),
        "alarms": alarms,
        "reproduction": (
            "PYTHONPATH=/home/itec/emanuele/pointstream-bp52 "
            "PS_DATA_ROOT=/home/itec/emanuele/pointstream-data "
            "python -m experiments.tier.bp52_background_search"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_028"])
    parser.add_argument("--frames", type=int, default=DEFAULT_SPAN_FRAMES)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.video != DEFAULT_VIDEO or args.scenes != ["scene_000", "scene_028"]:
        raise SystemExit("BP52 is fixed to alcaraz_highlights scene_000 scene_028")
    if args.frames != DEFAULT_SPAN_FRAMES:
        raise SystemExit(f"BP52 is fixed to {DEFAULT_SPAN_FRAMES} frames per scene")

    destination = Path(args.out_dir) if args.out_dir else ps_paths.outputs() / "bp52-background-crf"
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
    manifest = _manifest_snapshot(args.video, list(args.scenes))
    from src.runner.config_io import load_tier

    base: PointstreamConfig = load_tier("balanced")
    preset = primary_preset("av1")
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
        "points": list(POINT_NAMES),
        "bounds": fingerprint(bounds),
        "base_config": fingerprint(base),
    }
    guard_checkpoints(points_dir, identity)
    controls_path = destination / "metric-controls.json"
    controls = (
        json.loads(controls_path.read_text(encoding="utf-8"))
        if controls_path.is_file()
        else run_metric_controls(np.asarray(clips[0].frames[:2]), controls_path)
    )
    report_path = destination / "background-search.json"
    rows: list[dict[str, Any]] = []
    alarms = list(controls.get("alarms") or [])
    if controls.get("valid") is not True:
        report = _report(
            identity=identity,
            bounds=bounds,
            controls=controls,
            rows=rows,
            alarms=alarms,
            destination=destination,
            points_dir=points_dir,
        )
        write_json(report_path, report)
        return 1

    for point_name in POINT_NAMES:
        existing = load_checkpoint(points_dir, point_name)
        if existing is not None:
            row = existing
            print(f"resume {point_name}", flush=True)
        else:
            point: SweepPoint = named_point(point_name)
            started = time.perf_counter()
            row = run_point(
                clips,
                base,
                point,
                codec="av1",
                preset=preset,
                checkpoint_dir=points_dir / f"{point_name}.run",
            )
            row["attempt_wall_seconds"] = round(time.perf_counter() - started, 3)
            row["effective_background_stream_crf"] = point.stream_crf
            save_checkpoint(points_dir, point_name, row)
        rows.append(row)
        alarms.extend(_point_alarms(point_name, row, bounds))
        if point_name == "bg-crf51":
            row["bp49_comparison"] = _bp49_comparison(row)
            alarms.extend(_control_alarms(row["bp49_comparison"]))
        background_effect = _background_effect(rows)
        point_alarms = alarms + list(background_effect.get("alarms") or [])
        report = _report(
            identity=identity,
            bounds=bounds,
            controls=controls,
            rows=rows,
            alarms=point_alarms,
            destination=destination,
            points_dir=points_dir,
        )
        write_json(report_path, report)
        if point_alarms:
            print(f"stopping batch after {point_name}: {point_alarms}", flush=True)
            return 1
        print(
            f"{point_name}: {row.get('pointstream', {}).get('coded_bytes', 'FAILED')} B",
            flush=True,
        )

    final_effect = _background_effect(rows)
    alarms.extend(final_effect.get("alarms") or [])
    final = _report(
        identity=identity,
        bounds=bounds,
        controls=controls,
        rows=rows,
        alarms=alarms,
        destination=destination,
        points_dir=points_dir,
    )
    write_json(report_path, final)
    print(f"wrote {report_path}", flush=True)
    return 1 if alarms or completion_counts(rows)["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
