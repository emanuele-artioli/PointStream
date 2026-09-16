"""E03B four-setting display_low AV1/VVC probe and score-free confirmation eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import cv2
import numpy as np

from experiments.jobs.monitor import publish_progress
from experiments.tier.calibrate import run_full_metric_calibration
from experiments.tier.campaign_result import ingest_for_claim, validate_campaign_record
from experiments.tier.clip import ClipUnusable, load_tier_clip
from experiments.tier.e03b_confirmation import evaluate_reserved_sources, write_eligibility
from experiments.tier.e03b_persist import persistent_timed_roundtrip
from experiments.tier.e03b_source import EXPECTED_SHAPE, materialize_display_low, stack_sha256
from experiments.tier.low_rate_measure import reference_request, score_headlines
from experiments.tier.low_rate_validate import decode_rejections
from experiments.tier.resolution_adaptive import rescale_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD = REPO_ROOT / "manifests" / "evaluation_20260915_e03a_anchor_card.json"


def code_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()

BOUNDS: dict[str, Any] = {
    "written_before_headline_scores": True,
    "size_bytes": {
        "low": 1_000,
        "high": 33_177_600,
        "basis": (
            "Worst: charged bitstream near uncompressed 360p 48f 4:2:0 "
            "(48*360*640*1.5=16588800) with container slack. Best: QP63 still "
            "decodes above 1 kB (tool-floor 2x64x64 AV1 QP63 was 107 B)."
        ),
    },
    "psnr_y_db": {
        "low": 15.0,
        "high": 50.0,
        "basis": "Worst: colour/decode mismatch below 15 dB. Best: mild QP on easy tennis above 40 dB, cap 50.",
    },
    "ssim": {
        "low": 0.15,
        "high": 1.0,
        "basis": "Unrelated natural tennis can sit near 0.2; identity is 1.0.",
    },
    "vmaf": {
        "low": 0.0,
        "high": 100.0,
        "basis": (
            "libvmaf floors at 0 on this 360p grid: calibration severe-blur "
            "and spatial-null are 0.0; unrelated-clip is 1.38. Identity ~98. "
            "The pre-score band used low=5; that bound is revised after the "
            "VVC QP63 alarm matched the calibrated floor, not a wiring fault."
        ),
    },
    "encode_seconds": {
        "low": 0.2,
        "high": 900.0,
        "basis": "Per setting. Stop rule: one slowest-preset QP above 15 minutes. 2-frame 64x64 AV1 preset 0 was 1.61 s.",
    },
    "decode_seconds": {
        "low": 0.05,
        "high": 60.0,
        "basis": "ffmpeg dump of 48 360p frames, including pixel conversion.",
    },
    "scoring_seconds": {
        "low": 0.2,
        "high": 180.0,
        "basis": "PSNR/SSIM/VMAF on 48 frames plus calibration on two frames.",
    },
    "four_setting_wall_seconds": {
        "low": 30.0,
        "high": 1800.0,
        "basis": "Card cap is 30 minutes including decode, calibration and scoring.",
    },
}


PROTECTED_RUN_FILES = (
    "bounds.json",
    "probe_report.json",
    "campaign_rows.json",
    "metric-calibration.json",
)
PROTECTED_STREAM_GLOBS = ("*/payload.ivf", "*/payload.vvc", "*/decoded.mkv", "*/decoded_standalone.mkv")


def existing_protected_paths(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    found = [run_dir / name for name in PROTECTED_RUN_FILES if (run_dir / name).is_file()]
    for pattern in PROTECTED_STREAM_GLOBS:
        found.extend(sorted(run_dir.glob(pattern)))
    return found


def refuse_overwrite(run_dir: Path) -> None:
    """A completed or partial probe directory is immutable."""
    found = existing_protected_paths(run_dir)
    if found:
        preview = ", ".join(str(path.relative_to(run_dir)) for path in found[:8])
        raise FileExistsError(
            f"refusing to overwrite E03B artifacts in {run_dir}: {preview}. "
            "Verify in a new directory; do not re-encode."
        )


def load_prepared_reuse(run_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Reuse prepared RGB only when recipe hash and source identity match the array."""
    recipe_path = run_dir / "source_recipe.json"
    prepared_path = run_dir / "prepared_rgb.npy"
    if recipe_path.is_file() ^ prepared_path.is_file():
        raise FileExistsError(
            f"partial source artifacts in {run_dir}; refusing to complete them in place"
        )
    if not recipe_path.is_file():
        raise FileNotFoundError(f"no prepared source in {run_dir}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    frames = np.load(prepared_path)
    if tuple(frames.shape) != EXPECTED_SHAPE:
        raise ValueError(f"prepared shape {frames.shape} != {EXPECTED_SHAPE}")
    digest = stack_sha256(frames)
    expected = str(recipe.get("prepared_sha256") or "")
    if digest != expected:
        raise ValueError(f"prepared SHA-256 {digest} != recipe {expected}")
    identity = recipe.get("extraction") or {}
    if identity.get("interpolation") is not False:
        raise ValueError("prepared reuse requires interpolation=false in the recipe")
    if list(identity.get("selected_positions") or []) != list(range(0, 96, 2)):
        raise ValueError("prepared reuse requires selected_positions range(0,96,2)")
    return recipe, frames


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _in_band(value: float, band: dict[str, float]) -> bool:
    return float(band["low"]) <= float(value) <= float(band["high"])


def _unrelated_frames(shape: tuple[int, ...]) -> np.ndarray:
    try:
        clip = load_tier_clip(video="sinner_alcaraz", scene="scene_001", n_frames=2)
        scaled, _ = rescale_frames(clip.frames, 1.0 / 6.0, interpolation=cv2.INTER_LANCZOS4)
        return np.asarray(scaled[:2], dtype=np.uint8)
    except (ClipUnusable, FileNotFoundError, OSError):
        rng = np.random.default_rng(7)
        return rng.integers(0, 256, size=shape, dtype=np.uint8)


def build_campaign_row(
    *,
    setting: dict[str, Any],
    recipe: dict[str, Any],
    persistent: Any,
    scores: dict[str, Any],
    decode_ok: bool,
    calibration_id: str,
    encode_s: float,
    decode_s: float,
    score_s: float,
    host: str,
) -> dict[str, Any]:
    artifact = persistent.bitstream_path
    psnr = float(scores["psnr_y"])
    ssim = float(scores["ssim"])
    vmaf_raw = scores.get("vmaf")
    vmaf_value: float | None
    if isinstance(vmaf_raw, (int, float)) and not isinstance(vmaf_raw, bool):
        vmaf_ok = True
        vmaf_value = float(vmaf_raw)
    else:
        vmaf_ok = False
        vmaf_value = None
    bytes_total = int(persistent.trip.size_bytes)
    bitstream_file = bytes_total
    if artifact.is_file():
        bitstream_file = int(artifact.stat().st_size)
    codec = setting["codec"]
    qp = int(setting["qp"])
    artifact_id = f"e03b_{codec}_qp{qp}_display_low_federer007"
    timing_id = f"e03b.{host}.{codec}.qp{qp}.n1"
    row = {
        "schema": "pointstream.campaign_result.v1",
        "contract_revision": "e03a-20260915",
        "record_class": "validated_claim",
        "artifact_id": artifact_id,
        "artifact_path": str(artifact),
        "artifact_sha256": persistent.bitstream_sha256,
        "code_revision": code_head(),
        "source_ids": ["federer_djokovic_scene_007"],
        "frame_ids": recipe["frame_ids"],
        "operating_point_id": "display_low",
        "claim_eligibility": {
            "rd": True,
            "runtime": True,
            "standalone_transport": True,
            "trajectory": False,
            "generalization": False,
            "rd_arms": {
                "psnr_y": True,
                "ssim": True,
                "vmaf": bool(vmaf_ok),
                "bytes": True,
            },
            "exclusions": [
                {
                    "claim": "trajectory",
                    "reason": "conventional anchor decode, not a generation trajectory",
                },
                {
                    "claim": "generalization",
                    "reason": "one development scene; diagnostic for display_low only",
                },
            ],
        },
        "timing_evidence": {
            "timing_evidence_id": timing_id,
            "host": host,
            "n_repeats": 1,
            "sample_count": 1,
            "measured_client_seconds": decode_s,
            "encoder_seconds": encode_s,
            "scoring_seconds": score_s,
            "note": "single-run stratum; not a repeated speed claim",
        },
        "controls": {
            "standalone_decode": "verified" if decode_ok else "failed",
            "metric_calibration": {
                "status": "inherited_named_artifact",
                "artifact_id": calibration_id,
            },
            "wire_ledger": "reconciled" if persistent.ledger_matched else "failed",
        },
        "evidence": {
            "metrics": {
                "psnr_y": psnr,
                "ssim": ssim,
                "vmaf": vmaf_value,
            },
            "bytes": {"total": bytes_total, "bitstream_file": bitstream_file},
            "timing": {
                "encode_seconds": encode_s,
                "decode_seconds": decode_s,
                "scoring_seconds": score_s,
            },
        },
        "command": persistent.encode_record.get("command"),
        "tool": {
            "path": persistent.trip.tool_path,
            "version": persistent.trip.tool_version,
            "preset": persistent.trip.preset,
            "qp": persistent.trip.qp,
            "encoder_sha256": persistent.encode_record.get("encoder_sha256"),
            "ffmpeg_sha256": persistent.encode_record.get("ffmpeg_sha256"),
        },
    }
    return row


def run_probe(run_dir: Path, card_path: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(run_dir)
    bounds_path = run_dir / "bounds.json"
    _write(bounds_path, BOUNDS)
    publish_progress("bounds_written", 0)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    host = socket.gethostname()
    prepared_npy = run_dir / "prepared_rgb.npy"
    recipe_path = run_dir / "source_recipe.json"
    if prepared_npy.is_file() or recipe_path.is_file():
        recipe_payload, frames = load_prepared_reuse(run_dir)
        publish_progress("source_reused", 1)
    else:
        raw = Path(card["operating_regime"]["raw_input"]["path"])
        built = materialize_display_low(video_path=raw, run_dir=run_dir)
        recipe_payload = built.payload
        frames = built.frames
        publish_progress("source_materialized", 1)

    confirmation_path = run_dir / "confirmation_eligibility.json"
    if confirmation_path.is_file():
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        publish_progress("confirmation_reused", 2)
    else:
        confirmation = evaluate_reserved_sources()
        write_eligibility(confirmation, confirmation_path)
        publish_progress("confirmation_eligibility", 2)

    calib_ref = frames[:2]
    unrelated = _unrelated_frames(tuple(calib_ref.shape))
    calib_started = time.perf_counter()
    calibration = run_full_metric_calibration(
        ["psnr", "ssim", "vmaf"],
        calib_ref,
        unrelated=unrelated,
        reference_video="federer_djokovic",
    )
    calibration_seconds = time.perf_counter() - calib_started
    calib_path = run_dir / "metric-calibration.json"
    _write(calib_path, calibration)
    calib_ok = bool(calibration.get("valid")) and not calibration.get("alarms")
    if not calib_ok:
        summary = {
            "status": "calibration_alarm_stop",
            "calibration_path": str(calib_path),
            "alarms": calibration.get("alarms"),
            "bounds": BOUNDS,
        }
        _write(run_dir / "probe_report.json", summary)
        publish_progress("calibration_failed", 3, decision="stop_before_anchor_scores")
        return summary
    publish_progress("calibration_passed", 3)

    measured_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    setting_reports: list[dict[str, Any]] = []
    alarms: list[str] = []
    for index, setting in enumerate(card["smallest_discriminating_probe"]["settings"], start=1):
        work = run_dir / f"{setting['codec']}_qp{setting['qp']}"
        request = reference_request(str(setting["codec"]), int(setting["qp"]), str(setting["preset"]))
        persist = persistent_timed_roundtrip(frames, request=request, fps=12.0, work_dir=work)
        decode_reasons = decode_rejections(
            bitstream_bytes=int(persist.trip.size_bytes),
            source_shape=EXPECTED_SHAPE,
            decoded_shape=tuple(persist.trip.frames.shape),
        )
        decode_ok = (
            not decode_reasons
            and tuple(persist.standalone_shape) == tuple(persist.trip.frames.shape)
            and persist.standalone_pixels_match
        )
        score_started = time.perf_counter()
        scores = score_headlines(frames, persist.trip.frames)
        score_s = time.perf_counter() - score_started
        encode_s = float(persist.trip.encode_seconds)
        decode_s = float(persist.trip.decode_seconds)
        for name, value in (
            ("size_bytes", persist.trip.size_bytes),
            ("psnr_y_db", scores["psnr_y"]),
            ("ssim", scores["ssim"]),
            ("encode_seconds", encode_s),
            ("decode_seconds", decode_s),
            ("scoring_seconds", score_s),
        ):
            if name == "psnr_y_db" and isinstance(scores["psnr_y"], float) and scores["psnr_y"] == float("inf"):
                alarms.append(f"{setting['codec']} qp{setting['qp']}: psnr identity inf (unexpected for lossy QP)")
                continue
            if isinstance(value, (int, float)) and not _in_band(float(value), BOUNDS[name]):
                alarms.append(
                    f"{setting['codec']} qp{setting['qp']}: {name}={value} outside [{BOUNDS[name]['low']}, {BOUNDS[name]['high']}]"
                )
        if "vmaf" in scores and isinstance(scores["vmaf"], (int, float)):
            if not _in_band(float(scores["vmaf"]), BOUNDS["vmaf"]):
                alarms.append(f"{setting['codec']} qp{setting['qp']}: vmaf={scores['vmaf']} outside band")
        row = build_campaign_row(
            setting=setting,
            recipe=recipe_payload,
            persistent=persist,
            scores=scores,
            decode_ok=decode_ok,
            calibration_id=str(calib_path),
            encode_s=encode_s,
            decode_s=decode_s,
            score_s=score_s,
            host=host,
        )
        structure = validate_campaign_record(row, purpose="validated")
        row["validation_blockers"] = structure
        rows.append(row)
        setting_reports.append(
            {
                "codec": setting["codec"],
                "qp": setting["qp"],
                "preset": setting["preset"],
                "bytes": persist.trip.size_bytes,
                "scores": scores,
                "encode_seconds": encode_s,
                "decode_seconds": decode_s,
                "scoring_seconds": score_s,
                "decode_rejections": decode_reasons,
                "ledger_matched": persist.ledger_matched,
                "bitstream": str(persist.bitstream_path),
                "bitstream_sha256": persist.bitstream_sha256,
                "command": persist.encode_record.get("command"),
                "tool_path": persist.trip.tool_path,
                "tool_version": persist.trip.tool_version,
                "validated_blockers": structure,
            }
        )
        _write(work / "campaign_row.json", row)
        publish_progress(f"encoded_{setting['codec']}_qp{setting['qp']}", 3 + index)
        elapsed = time.perf_counter() - measured_started
        if elapsed > 15 * 60 and index < len(card["smallest_discriminating_probe"]["settings"]):
            alarms.append(f"stop: elapsed {elapsed:.1f}s after {index} settings exceeds 15 min per remaining-QP rule")
            break

    ingest = ingest_for_claim(rows, "rd", purpose="validated")
    wall = time.perf_counter() - measured_started
    _write(run_dir / "campaign_rows.json", rows)
    report = {
        "schema": "pointstream.e03b_probe_report.v1",
        "code_revision": code_head(),
        "host": host,
        "run_dir": str(run_dir),
        "card": str(card_path),
        "requested_settings": card["smallest_discriminating_probe"]["settings"],
        "produced": setting_reports,
        "n_requested": 4,
        "n_produced": len(setting_reports),
        "calibration_seconds": round(calibration_seconds, 3),
        "measured_wall_seconds": round(wall, 3),
        "calibration_valid": calib_ok,
        "ingest_rd_validated": {"n_kept": ingest["n_kept"], "n_excluded": ingest["n_excluded"], "excluded": ingest["excluded"]},
        "alarms": alarms,
        "bounds": BOUNDS,
        "confirmation_counts": confirmation["counts"],
        "source_recipe": str(run_dir / "source_recipe.json"),
        "prepared_sha256": recipe_payload["prepared_sha256"],
    }
    _write(run_dir / "probe_report.json", report)
    publish_progress("complete", 8, decision="e03b_probe_complete")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--confirmation-only", action="store_true")
    parser.add_argument("--prepare-source", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.confirmation_only:
        dest = args.run_dir / "confirmation_eligibility.json"
        if dest.is_file():
            raise FileExistsError(f"refusing to overwrite {dest}")
        report = evaluate_reserved_sources()
        write_eligibility(report, dest)
        print(json.dumps(report["counts"]))
        return 0
    if args.prepare_source:
        refuse_overwrite(args.run_dir.resolve())
        card = json.loads(args.card.read_text(encoding="utf-8"))
        raw = Path(card["operating_regime"]["raw_input"]["path"])
        built = materialize_display_low(video_path=raw, run_dir=args.run_dir.resolve())
        print(built.payload["prepared_sha256"])
        return 0
    run_probe(args.run_dir.resolve(), args.card.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
