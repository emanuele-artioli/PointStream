"""Gate B: Confirmation on held-out content using frozen C0–C3 procedure.

Coordinates:
- Verification of held-out sequences in manifests/gate_b_confirmation.json
- Automated player detection and tracking (YoloDetector + HeuristicSelector)
- Frozen rate ladder: Rungs C0, C1, C2, C3 (VVC stream background + WebP crops)
- Disjoint timing boundaries (encoder, client, evaluation)
- Pre-registered metric thresholds, rot bounds, and monotonicity checks
- Three-axis reporting (size, quality, speed) with source-level uncertainty
"""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from experiments.headroom.real import extract_24fps_pngs
from experiments.tier.gate_a_tools import write_tool_identity
from experiments.tier.low_rate_references import compare_candidate_to_anchor, encode_reference_curve
from experiments.tier.low_rate_sweep import pointstream_e1
from src.components.codec.tools import resolve_ffmpeg
from src.components.detection.yolo import YoloDetector
from src.components.selection.heuristic import HeuristicSelector
from src.contracts import paths as ps_paths
from src.contracts.frozen_procedure import (
    FROZEN_ANCHOR_PRESETS,
    FROZEN_ANCHOR_QPS,
    FROZEN_RUNGS,
    check_adjacent_rungs,
    configure_frozen_rung,
    get_frozen_bounds,
    validate_ledger,
)
from experiments.tier.protocol import (
    ExperimentIdentity,
    ProtocolEvidence,
    capture_current_identity,
    evaluate_confirmation_protocol,
)
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner.config_io import load_tier


DEFAULT_MANIFEST = ps_paths.repo_root() / "manifests" / "gate_b_confirmation.json"
HEARTBEAT_INTERVAL_SECONDS = 600.0


@dataclass(frozen=True)
class ConfirmationClip:
    """Held-out confirmation clip loaded and tracked automatically."""

    source_id: str
    scene: str
    context_id: str
    frames: np.ndarray
    objects: tuple[ObjectRequest, ...]
    masks: np.ndarray
    paste_back_mae: float


def load_confirmation_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Gate B manifest not found at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def verify_source_integrity(manifest: dict[str, Any]) -> dict[str, Path]:
    data_root = ps_paths.data_root()
    paths: dict[str, Path] = {}
    for src in manifest.get("sources", []):
        sid = src["source_id"]
        rel_path = src["raw_file_path"]
        abs_path = data_root / rel_path
        if not abs_path.is_file():
            raise FileNotFoundError(f"Missing confirmation raw file for {sid} at {abs_path}")
        calculated_sha = hashlib.sha256(abs_path.read_bytes()).hexdigest()
        expected_sha = src["sha256"]
        if calculated_sha != expected_sha:
            raise ValueError(
                f"SHA256 mismatch for {sid}: expected {expected_sha}, got {calculated_sha}"
            )
        paths[sid] = abs_path
    return paths


def _fill_missing_track_frames(
    t_dict: dict[int, tuple[int, int, int, int]],
    T: int,
    width: int,
    height: int,
) -> dict[int, tuple[int, int, int, int]]:
    """Fill un-detected frames across [0, T-1] using linear interpolation and edge holding."""
    if not t_dict:
        return {}
    known_t = sorted(t_dict.keys())
    filled: dict[int, tuple[int, int, int, int]] = {}
    for t in range(T):
        if t in t_dict:
            b = t_dict[t]
        elif t <= known_t[0]:
            b = t_dict[known_t[0]]
        elif t >= known_t[-1]:
            b = t_dict[known_t[-1]]
        else:
            prev_t = max(k for k in known_t if k < t)
            next_t = min(k for k in known_t if k > t)
            alpha = (t - prev_t) / (next_t - prev_t)
            b_prev = np.array(t_dict[prev_t], dtype=float)
            b_next = np.array(t_dict[next_t], dtype=float)
            b_interp = (1.0 - alpha) * b_prev + alpha * b_next
            b = (
                int(round(b_interp[0])),
                int(round(b_interp[1])),
                int(round(b_interp[2])),
                int(round(b_interp[3])),
            )
        x1 = max(0, min(width - 1, b[0]))
        y1 = max(0, min(height - 1, b[1]))
        x2 = max(x1 + 1, min(width, b[2]))
        y2 = max(y1 + 1, min(height, b[3]))
        filled[t] = (x1, y1, x2, y2)
    return filled


def materialize_clip(
    source_meta: dict[str, Any],
    video_path: Path,
    out_dir: Path,
    *,
    n_frames: int = 96,
    ffmpeg: str | None = None,
) -> ConfirmationClip:
    """Extract frames and track players automatically without tuning."""
    ffmpeg_bin = ffmpeg or resolve_ffmpeg().path
    shot = source_meta["selected_shot"]
    start_s = float(shot["start_s"])
    duration_s = float(n_frames / 24.0)

    clip_frames_dir = out_dir / "frames"
    files = extract_24fps_pngs(
        video_path,
        t_start=start_s,
        duration=duration_s,
        out_dir=clip_frames_dir,
        ffmpeg=ffmpeg_bin,
    )
    if len(files) < n_frames:
        raise RuntimeError(
            f"Expected {n_frames} frames, ffmpeg extracted {len(files)} from {video_path}"
        )
    files = files[:n_frames]
    frames = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])
    T, H, W, _ = frames.shape

    detector = YoloDetector()
    selector = HeuristicSelector()

    player_tracks: dict[str, dict[int, tuple[int, int, int, int]]] = {
        "player_near": {},
        "player_far": {},
    }
    for t in range(T):
        dets = detector.detect(frames[t])
        sel = selector.select(dets, (H, W))
        for item in sel:
            pid = item.track_id
            if pid in player_tracks:
                b = item.bbox
                player_tracks[pid][t] = (
                    max(0, int(np.floor(b.x1))),
                    max(0, int(np.floor(b.y1))),
                    min(W, int(np.ceil(b.x2))),
                    min(H, int(np.ceil(b.y2))),
                )

    objects: list[ObjectRequest] = []
    union_mask = np.zeros((T, H, W), dtype=bool)

    for pid in ("player_near", "player_far"):
        t_dict = _fill_missing_track_frames(player_tracks[pid], T, width=W, height=H)
        if not t_dict:
            continue
        first_t = min(t_dict.keys())
        x1, y1, x2, y2 = t_dict[first_t]
        first_crop = frames[first_t, y1:y2, x1:x2]
        first_bbox = (x1, y1, x2, y2)
        mask = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            bx1, by1, bx2, by2 = t_dict[t]
            mask[t, by1:by2, bx1:bx2] = True
            union_mask[t, by1:by2, bx1:bx2] = True
        objects.append(
            ObjectRequest(
                object_id=pid,
                appearance=first_crop,
                bbox=first_bbox,
                mask=mask,
                frame_index=first_t,
            )
        )

    return ConfirmationClip(
        source_id=source_meta["source_id"],
        scene=shot["scene_label"],
        context_id=f"{source_meta['source_id']}_{shot['scene_label']}",
        frames=frames,
        objects=tuple(objects),
        masks=union_mask,
        paste_back_mae=0.0,
    )


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



def confirmation_verdict(
    sources: list[dict[str, Any]],
    alarms: list[str],
    *,
    identity: ExperimentIdentity | dict[str, Any] | None = None,
    expected_identity: ExperimentIdentity | dict[str, Any] | None = None,
    evidence: ProtocolEvidence | dict[str, Any] | None = None,
    is_pilot: bool = True,
) -> dict[str, Any]:
    """Report pilot completion without certifying an unimplemented protocol.

    This driver requires evidence of actual client-output scoring, full wire cost,
    calibrated metrics/nulls, source eligibility (six independent matches required for Gate B),
    source-level uncertainty, and full identity matching before any confirmation pass can be claimed.
    Runs without full evidence remain development pilots and cannot claim confirmation.
    Historical report JSONs remain immutable; re-adjudicate them separately.
    """
    if not sources:
        return {
            "execution_completed": False,
            "pilot_alarms_clear": False,
            "gate_b_passed": False,
            "confirmation_status": "incomplete_protocol",
            "confirmation_blockers": ["six independent sources required; only 0 reported"],
        }
    return evaluate_confirmation_protocol(
        sources,
        alarms,
        identity=identity,
        expected_identity=expected_identity,
        evidence=evidence,
        is_pilot=is_pilot,
        required_matches=6,
    )


def run_dry_run(destination: Path, n_frames: int = 96) -> dict[str, Any]:
    """Execute Gate B dry run verifying manifest, tool floor, bounds, and rungs."""
    print(f"=== Starting Gate B Dry Run (n_frames={n_frames}) ===")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_confirmation_manifest()
    source_paths = verify_source_integrity(manifest)
    tools = write_tool_identity(destination)

    bounds = get_frozen_bounds(n_frames=n_frames, height=1080, width=1920, n_scenes=1)
    (destination / "bounds-before-run.json").write_text(json.dumps(bounds, indent=2) + "\n")

    base = load_tier("balanced")
    identity = capture_current_identity(DEFAULT_MANIFEST, config=base)
    (destination / "experiment-identity.json").write_text(json.dumps(identity.to_dict(), indent=2) + "\n")
    ladder_plan = []
    for rung in FROZEN_RUNGS:
        cfg = configure_frozen_rung(base, rung, context_id="dryrun_context")
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
            }
        )

    summary = {
        "status": "gate_b_dry_run_complete",
        "n_frames": n_frames,
        "sources": list(source_paths.keys()),
        "tools_resolved": tools["tools"],
        "identity": identity.to_dict(),
        "ladder_plan": ladder_plan,
        "bounds": bounds,
    }
    (destination / "dry-run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=== Gate B Dry Run Complete ===")
    return summary


def run_confirmation(
    destination: Path,
    n_frames: int = 96,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Run Gate B confirmation sweep across held-out sources."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_confirmation_manifest(manifest_path)
    source_paths = verify_source_integrity(manifest)
    write_tool_identity(destination)
    base = load_tier("balanced")
    identity = capture_current_identity(manifest_path, config=base)
    (destination / "experiment-identity.json").write_text(json.dumps(identity.to_dict(), indent=2) + "\n")
    ffmpeg = resolve_ffmpeg().path

    points = destination / "points"
    points.mkdir(parents=True, exist_ok=True)

    per_source_reports: list[dict[str, Any]] = []
    all_alarms: list[str] = []

    for src_meta in manifest.get("sources", []):
        sid = src_meta["source_id"]
        vpath = source_paths[sid]
        print(f"\n--- Materializing held-out source: {sid} ---")
        clip_dir = destination / "clips" / sid
        clip = materialize_clip(src_meta, vpath, clip_dir, n_frames=n_frames, ffmpeg=ffmpeg)

        H, W = clip.frames.shape[1], clip.frames.shape[2]
        bounds = get_frozen_bounds(n_frames=n_frames, height=H, width=W, n_scenes=1)
        (clip_dir / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")

        print(f"Running anchors for {sid} (H={H}, W={W})...")
        anchors: dict[str, Any] = {}
        for codec in ("av1", "vvc"):
            preset = FROZEN_ANCHOR_PRESETS[codec]
            anchors[codec] = encode_reference_curve(
                [clip],
                codec=codec,
                preset=preset,
                qps=FROZEN_ANCHOR_QPS,
                fps=24.0,
                checkpoint_root=points / f"{sid}.{codec}.anchors",
            )

        print(f"Running PointStream C0–C3 confirmation ladder for {sid}...")
        base = load_tier("balanced")
        ps_rows: list[dict[str, Any]] = []
        for rung in FROZEN_RUNGS:
            cfg = configure_frozen_rung(base, rung, context_id=clip.context_id)
            payload = pointstream_e1([clip], cfg, checkpoint_dir=points / f"{sid}.{rung.name}.run")
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
                print(f"  ALARM on {rung.name}: {alarms}")
                all_alarms.extend(alarms)
            ps_rows.append(row)

        comparisons: dict[str, Any] = {}
        for codec in ("av1", "vvc"):
            comparisons[codec] = {
                pattern: compare_candidate_to_anchor(
                    ps_rows,
                    anchors[codec]["access_patterns"][pattern],
                )
                for pattern in ("continuous", "segmented")
            }

        per_source_reports.append(
            {
                "source_id": sid,
                "match_name": src_meta["match_name"],
                "resolution": f"{W}x{H}",
                "n_frames": n_frames,
                "fps": 24.0,
                "pointstream_rungs": ps_rows,
                "anchors": anchors,
                "comparisons": comparisons,
            }
        )

    final_report = {
        "status": "gate_b_pilot_complete",
        "n_sources": len(per_source_reports),
        "sources": per_source_reports,
        "alarms": all_alarms,
        "identity": identity.to_dict(),
        **confirmation_verdict(per_source_reports, all_alarms, identity=identity),
        "timestamp_unix": time.time(),
    }
    (destination / "report.json").write_text(json.dumps(final_report, indent=2) + "\n")
    print("\n=== Gate B Confirmation Report Complete ===")
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B confirmation driver")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight checks only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ps_paths.outputs() / "gate-b-confirmation",
        help="Destination directory for confirmation outputs",
    )
    parser.add_argument("--frames", type=int, default=96, help="Frame count (default: 96)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to Gate B confirmation manifest",
    )
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.output_dir, n_frames=args.frames)
    else:
        run_confirmation(args.output_dir, n_frames=args.frames, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
