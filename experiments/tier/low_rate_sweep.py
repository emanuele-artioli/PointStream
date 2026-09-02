"""Staged PointStream low-rate search. Not a Cartesian product.

Generation stays off. Each stage moves one rate-bearing family, records the
full byte ledger, and asserts that the intended categories moved — or records
why they did not. AV1 and VVC run on the same frames, size, frame rate and
colour convention, as one joint encode (continuous) and as independent
segments (segmented). The headline control is the one that matches the
product claim; the other is an access-pattern tradeoff.

E1 evidence still needs B1 (canonical canvas) and D1 (long eligible scenes).
This module is the search harness; a diagnostic run on current short clips
does not become a Gate-A result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.tier.ladder_scenes import (
    anchor_over_sequence,
    load_scene_sequence,
    pointstream_over_sequence,
)
from experiments.tier.low_rate_plan import (
    SweepPoint,
    intended_category,
    ledger_moved,
    points_for,
    stage_names,
)
from experiments.tier.low_rate_validate import BOUNDS_PATH, DECLARED_FPS, OUT_DIR, PROBE_PATH
from src.components.codec.measure import PRESETS
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import PointstreamConfig
from src.runner.config_io import load_tier


def apply_point(base: PointstreamConfig, point: SweepPoint, *, codec: str) -> PointstreamConfig:
    residual = replace(
        base.residual,
        codec=codec,
        preset=PRESETS[codec],
        rate_control=RateControl.QP,
        rate=int(point.residual_qp) if point.residual_qp is not None else 55,
    )
    background = replace(
        base.background,
        method=point.background_method,
        stream_codec="av1" if codec == "vvc" else codec,
        stream_crf=int(point.stream_crf),
        keyframe_interval=0,
        reference_mode="last",
    )
    appearance = replace(
        base.appearance,
        jpeg_quality=int(point.appearance_jpeg_quality),
        downscale=int(point.appearance_downscale),
    )
    motion = replace(base.motion, max_points=int(point.motion_max_points))
    lattice = replace(base.lattice, residual=bool(point.residual_on), generation=False)
    if not point.object_stream_on:
        lattice = replace(
            lattice,
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
    return base.with_(
        residual=residual,
        background=background,
        appearance=appearance,
        motion=motion,
        lattice=lattice,
    )


def run_point(
    clips: list[Any],
    base: PointstreamConfig,
    point: SweepPoint,
    *,
    codec: str,
    preset: str,
    fps: float,
) -> dict[str, Any]:
    """One operating point, both access-pattern controls, one PointStream run."""
    del fps  # recorded by the caller; coded_roundtrip uses the request, not this
    row: dict[str, Any] = {
        "name": point.name,
        "stage": point.stage,
        "config": {
            "stream_crf": point.stream_crf,
            "residual_on": point.residual_on,
            "residual_qp": point.residual_qp,
            "appearance_jpeg_quality": point.appearance_jpeg_quality,
            "appearance_downscale": point.appearance_downscale,
            "motion_max_points": point.motion_max_points,
            "object_stream_on": point.object_stream_on,
            "background_method": point.background_method,
        },
    }
    # Continuous = joint encode of the ordered scenes. Segmented = sum of
    # independent per-scene encodes. Both from ladder_scenes.anchor_over_sequence.
    request = EncodeRequest(
        codec_name=codec,
        rate_control=RateControl.QP,
        rate=int(point.residual_qp or 45),
        preset=preset,
        pix_fmt="yuv420p",
    )
    request.validate()
    try:
        anchor = anchor_over_sequence(clips, request)
        row["continuous"] = {
            "bytes": anchor["joint_bytes"],
            "psnr_y_dB": anchor["psnr_y_dB"],
            "seconds": anchor["seconds"],
            "psnr_y_by_frame": anchor["psnr_y_by_frame"],
        }
        row["segmented"] = {
            "bytes": anchor["separate_bytes"],
            "joint_over_separate": anchor["joint_over_separate"],
        }
    except Exception as exc:  # noqa: BLE001
        row["anchor_error"] = repr(exc)

    try:
        tuned = apply_point(base, point, codec=codec)
        row["pointstream"] = pointstream_over_sequence(clips, tuned)
        if not point.object_stream_on:
            row["pointstream"]["control"] = "object-stream-off"
    except Exception as exc:  # noqa: BLE001
        row["pointstream_error"] = repr(exc)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="alcaraz_highlights")
    parser.add_argument("--scenes", nargs="+", default=["scene_000", "scene_010"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--stage", default=None, help="run one named stage, or all")
    parser.add_argument("--fps", type=float, default=DECLARED_FPS)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-short-scenes",
        action="store_true",
        help="run on current BP21 windows. That is diagnostic, not E1 evidence.",
    )
    args = parser.parse_args(argv)

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds first.")
    if not PROBE_PATH.is_file():
        raise SystemExit(
            f"{PROBE_PATH} does not exist. Probe AV1/VVC floors before the sweep."
        )
    if args.frames > 16 and not args.allow_short_scenes:
        # Long scenes need the canonical canvas (B1). Refuse rather than hit
        # the unequal-canvas failure and call it a PointStream result.
        raise SystemExit(
            "frames > 16 need BP44 canonical canvases. Pass --allow-short-scenes "
            "only for a diagnostic on the 8/16-frame windows."
        )
    if not args.allow_short_scenes:
        raise SystemExit(
            "E1 is blocked on B1 and D1. For a diagnostic on existing 8-frame "
            "clips pass --allow-short-scenes and say so in the report."
        )

    stages = [args.stage] if args.stage else list(stage_names())
    clips = load_scene_sequence(args.video, list(args.scenes), n_frames=args.frames)
    base = load_tier(args.tier)
    preset = PRESETS[args.codec]
    rows: list[dict[str, Any]] = []
    category_notes: list[str] = []

    for stage in stages:
        points = points_for(stage)
        stage_rows: list[dict[str, Any]] = []
        print(f"stage {stage} ({len(points)} points)", flush=True)
        for point in points:
            row = run_point(
                clips, base, point, codec=args.codec, preset=preset, fps=args.fps
            )
            rows.append(row)
            stage_rows.append(row)
            ps = row.get("pointstream") or {}
            print(
                f"  {point.name:<16} {ps.get('coded_bytes', '—')} B  "
                f"{ps.get('psnr_y_dB', '—')}",
                flush=True,
            )
        key = intended_category(stage)
        if stage != "controls" and not ledger_moved(stage_rows, key=key):
            note = (
                f"stage {stage}: intended ledger key {key!r} did not move "
                "across points. The knob is not reaching the payload, or the "
                "points are not distinct."
            )
            category_notes.append(note)
            print(f"  ALARM {note}", flush=True)

    report = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "diagnostic": True,
        "e1_evidence": False,
        "reason_not_evidence": (
            "E1 requires B1 canonical canvases and D1 long eligible scenes. "
            "This run is a harness check or a short-scene diagnostic."
        ),
        "video": args.video,
        "scenes": list(args.scenes),
        "frames_per_scene": args.frames,
        "fps": args.fps,
        "codec": args.codec,
        "preset": preset,
        "generation": "off",
        "access_patterns": ["continuous", "segmented"],
        "headline_control": "undecided — follows the product claim, not this diagnostic",
        "tried": [row["name"] for row in rows],
        "ledger_notes": category_notes,
        "bounds_file": str(BOUNDS_PATH),
        "probe_file": str(PROBE_PATH),
        "rows": rows,
    }
    dest = Path(args.out) if args.out else OUT_DIR / f"sweep-{args.video}-{args.codec}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 1 if category_notes else 0


if __name__ == "__main__":
    raise SystemExit(main())
