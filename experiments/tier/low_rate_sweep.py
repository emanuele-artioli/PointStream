"""Staged PointStream low-rate search. Not a Cartesian product.

Generation stays off. Each stage moves one rate-bearing family, records the
full byte ledger, and asserts that the intended categories moved — or records
why they did not.

AV1 and VVC reference curves are encoded separately
(``python -m experiments.tier.low_rate_references``) on the same frames, size,
frame rate and colour, as continuous and segmented access patterns. This sweep
does not re-encode those anchors at the residual QP.
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

from experiments.tier.low_rate_canvas import (
    clip_context_ids,
    require_run_accepts_context_ids,
    with_canonical_background,
)
from experiments.tier.low_rate_checkpoint import load_checkpoint, save_checkpoint, write_json
from experiments.tier.low_rate_checkpoint import (
    completion_counts, fingerprint, guard_checkpoints, implementation_digest, source_identity,
)
from experiments.tier.low_rate_clips import (
    DEFAULT_SCENES,
    DEFAULT_SPAN_FRAMES,
    DEFAULT_VIDEO,
    load_e1_sequence,
)
from experiments.tier.low_rate_fallback import run_fallback_control
from experiments.tier.low_rate_identity import (
    checkpoint_dir,
    input_identity,
    references_path,
    sweep_path,
)
from experiments.tier.low_rate_measure import (
    late_frame_bound_alarms,
    late_frame_by_scene,
    late_frame_report,
    pointstream_timing,
    primary_preset,
    score_headlines,
    stream_codec_provenance,
)
from experiments.tier.low_rate_plan import (
    SweepPoint,
    intended_category,
    ledger_moved,
    points_for,
    select_work,
    stage_names,
)
from experiments.tier.low_rate_references import (
    ACCESS_PATTERNS,
    compare_candidate_to_anchor,
    load_reference_curve,
)
from experiments.tier.low_rate_validate import BOUNDS_PATH, DECLARED_FPS, PROBE_PATH
from src.contracts.codecs import RateControl
from src.contracts.config import PointstreamConfig
from src.pipeline.reconstruction.dispatch import GeneratorRef


def apply_point(
    base: PointstreamConfig,
    point: SweepPoint,
    *,
    codec: str,
    preset: str,
    context_id: str,
) -> PointstreamConfig:
    residual = replace(
        base.residual,
        codec=codec,
        preset=preset,
        rate_control=RateControl.QP,
        rate=int(point.residual_qp) if point.residual_qp is not None else 55,
    )
    background = with_canonical_background(
        base.background,
        method=point.background_method,
        stream_codec="av1" if codec == "vvc" else codec,
        stream_crf=int(point.stream_crf),
        context_id=context_id,
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
    fallback = replace(
        base.fallback,
        codec=codec,
        rate_control=RateControl.QP,
        rate=int(point.residual_qp) if point.residual_qp is not None else 55,
        preset=preset,
        pix_fmt="yuv420p",
    )
    return base.with_(
        residual=residual,
        background=background,
        appearance=appearance,
        motion=motion,
        lattice=lattice,
        fallback=fallback,
    )


def _no_generator() -> GeneratorRef:
    raise AssertionError("generation is off in every low-rate config used here")


def pointstream_e1(
    clips: list[Any],
    config: PointstreamConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """One ``run()`` over the long scenes, scored on delivered frames."""
    run = require_run_accepts_context_ids()
    ids = clip_context_ids(clips)
    shapes = [tuple(int(x) for x in np.asarray(clip.frames).shape) for clip in clips]
    print(
        f"pointstream_e1 start context_ids={list(ids)} shapes={shapes}",
        flush=True,
    )
    started = time.perf_counter()
    result = run(
        config,
        [np.asarray(clip.frames) for clip in clips],
        bind_generator_fn=_no_generator,
        objects=tuple(clip.objects for clip in clips),
        context_ids=ids,
        checkpoint_dir=checkpoint_dir,
    )
    wall = time.perf_counter() - started
    source = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
    delivered = result.delivered_frames
    if delivered.shape != source.shape:
        raise ValueError(f"PointStream delivered {delivered.shape}, expected {source.shape}")
    sizes = result.sizes
    total = int(sizes.transport_total)
    panorama = int(sizes.panorama)
    scores = score_headlines(source, delivered)
    per_scene = late_frame_by_scene(clips, source, delivered)
    bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))
    late_alarms = late_frame_bound_alarms(per_scene, bounds)
    return {
        "coded_bytes": total,
        "bytes": total,
        "usable": (isinstance(scores.get("vmaf"), float) and bool(sizes.is_rate)
                   and result.timing.get("hourly_checkpoint_budget_met", True) is True),
        "scores": scores,
        "late_frame": {
            "by_scene": per_scene,
            "joined_across_scenes": late_frame_report(source, delivered),
            "bound": bounds["bounds"]["late_frame_quality_change"],
            "alarms": late_alarms,
            "note": (
                "The rot bound is last-minus-first *per scene*. "
                "joined_across_scenes crosses a scene boundary and is diagnostic only."
            ),
        },
        "is_rate": bool(sizes.is_rate),
        "raw_parts": list(sizes.raw_parts),
        "parts": {
            "residual": int(sizes.residual),
            "panorama": panorama,
            "actor_reference": int(sizes.actor_reference),
            "metadata": int(sizes.metadata),
        },
        "background_share": round(panorama / total, 4) if total else None,
        "n_chunks": len(result.chunks),
        "n_frames": int(source.shape[0]),
        "context_ids": list(ids),
        "canvas": getattr(config.background, "canvas", None),
        "background_codec": stream_codec_provenance(
            getattr(config.background, "stream_codec", "av1")
        ),
        "residual_codec": {
            "role": "residual",
            "enabled": bool(config.lattice.residual),
            "codec": config.residual.codec,
            "preset": config.residual.preset,
        },
        "stage_seconds": list(result.stage_seconds),
        "invocation_phase_seconds": result.phase_seconds,
        "recovery_alarm": (
            None if result.timing.get("hourly_checkpoint_budget_met", True) is True
            else "hourly checkpoint budget exceeded or unverified after interruption; do not expand batch"
        ),
        **pointstream_timing(wall),
        **result.timing,
    }


def run_point(
    clips: list[Any],
    base: PointstreamConfig,
    point: SweepPoint,
    *,
    codec: str,
    preset: str,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """One PointStream operating point. Anchors are not encoded here."""
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
            "residual_codec": codec,
            "residual_preset": preset,
        },
    }
    try:
        tuned = apply_point(
            base, point, codec=codec, preset=preset, context_id=clip_context_ids(clips)[0]
        )
        row["pointstream"] = pointstream_e1(
            clips, tuned, checkpoint_dir=checkpoint_dir
        )
        if not point.object_stream_on:
            row["pointstream"]["control"] = "object-stream-off"
    except Exception as exc:  # noqa: BLE001
        row["pointstream_error"] = repr(exc)
    return row


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("pointstream")
        if not payload:
            continue
        out.append(
            {
                "name": row.get("name"),
                "bytes": payload.get("bytes") or payload.get("coded_bytes"),
                "usable": bool(payload.get("usable")),
                "scores": payload.get("scores") or {},
                "encode_seconds": payload.get("encode_seconds"),
                "decode_seconds": payload.get("decode_seconds"),
                "run_seconds": payload.get("run_seconds"),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--frames", type=int, default=DEFAULT_SPAN_FRAMES)
    parser.add_argument("--codec", default="av1")
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--stage", default=None, help="run one named stage, or all")
    parser.add_argument(
        "--point",
        default=None,
        help="run one named operating point (native-resolution preflight)",
    )
    parser.add_argument("--fps", type=float, default=DECLARED_FPS)
    parser.add_argument("--preset", default=None, help="override the slowest-preset rule")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="run PointStream without loading the independent reference curves",
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="skip the conventional-fallback equivalence control",
    )
    args = parser.parse_args(argv)
    if args.fps != DECLARED_FPS:
        raise SystemExit("BP46 inputs and the PointStream runner require 24 fps")

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds first.")
    if not PROBE_PATH.is_file():
        raise SystemExit(
            f"{PROBE_PATH} does not exist. Probe AV1/VVC floors before the sweep."
        )

    from src.runner.config_io import load_tier
    from experiments.tier.low_rate_validate import probe_qps

    identity = input_identity(
        video=args.video,
        scenes=list(args.scenes),
        frames_per_scene=args.frames,
        codec=args.codec,
        fps=args.fps,
    )
    dest = Path(args.out) if args.out else sweep_path(identity)
    points_dir = checkpoint_dir(dest)

    try:
        work = select_work(stage=args.stage, point=args.point)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    clips = load_e1_sequence(args.video, list(args.scenes), n_frames=args.frames)
    context_ids = clip_context_ids(clips)
    base = load_tier(args.tier)
    preset = primary_preset(args.codec, override=args.preset)
    identity["source"] = source_identity(clips)
    identity["preset"] = preset
    identity["implementation"] = implementation_digest()
    guard_checkpoints(points_dir, {
        "input": identity, "preset": preset, "config": fingerprint(base),
        "plan": fingerprint([points_for(name) for name in stage_names()]),
        "bounds": fingerprint(BOUNDS_PATH.read_text()),
        "probe": fingerprint(PROBE_PATH.read_text()),
    })
    rows: list[dict[str, Any]] = []
    category_notes: list[str] = []
    fallback_control: dict[str, Any] | None = None

    def _report() -> dict[str, Any]:
        return {
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "e1_evidence": False,
            "reason_not_evidence": (
                "E1 is diagnostic on two videos until the frozen rule passes E2 "
                "on at least six independent videos. This file is the harness output."
            ),
            "input": identity,
            "context_ids": list(context_ids),
            "canvas": "canonical",
            "video": args.video,
            "scenes": list(args.scenes),
            "frames_per_scene": args.frames,
            "fps": args.fps,
            "codec": args.codec,
            "preset": preset,
            "generation": "off",
            "access_patterns": list(ACCESS_PATTERNS),
            "headline_control": "undecided — follows the product claim after both curves exist",
            "reference_file": None if args.skip_compare else str(references_path(identity)),
            "checkpoint_dir": str(points_dir),
            "preflight_point": args.point,
            "tried": [row["name"] for row in rows],
            "ledger_notes": category_notes,
            "bounds_file": str(BOUNDS_PATH),
            "probe_file": str(PROBE_PATH),
            "fallback_control": fallback_control,
            "comparisons": comparisons,
            "rows": rows,
            "completion": completion_counts(rows),
        }

    comparisons: dict[str, Any] = {}

    if not args.skip_fallback:
        existing_fb = load_checkpoint(points_dir, "fallback-equivalence")
        if existing_fb is not None:
            fallback_control = existing_fb
            print("resume fallback-equivalence", flush=True)
        else:
            bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))
            band = bounds["bounds"]["fallback_reproduces_reference"]
            joined = np.concatenate([np.asarray(clip.frames) for clip in clips], axis=0)
            fallback_control = run_fallback_control(
                joined,
                base.fallback,
                codec=args.codec,
                qp=int(probe_qps(args.codec)[0]),
                preset=preset,
                fps=float(args.fps),
                rate_rel=(
                    float(band["rate_rel"]["low"]),
                    float(band["rate_rel"]["high"]),
                ),
                vmaf_abs=(
                    float(band["vmaf_abs"]["low"]),
                    float(band["vmaf_abs"]["high"]),
                ),
            )
            save_checkpoint(points_dir, "fallback-equivalence", fallback_control)
        if fallback_control and not (fallback_control.get("comparison") or {}).get("held"):
            category_notes.append(
                "conventional-fallback control did not reproduce the reference codec"
            )
        write_json(dest, _report())

    for stage, points in work:
        stage_rows: list[dict[str, Any]] = []
        print(f"stage {stage} ({len(points)} points) preset {preset}", flush=True)
        for point in points:
            existing = load_checkpoint(points_dir, point.name)
            if existing is not None:
                row = existing
                print(f"  resume {point.name}", flush=True)
            else:
                row = run_point(
                    clips,
                    base,
                    point,
                    codec=args.codec,
                    preset=preset,
                    checkpoint_dir=points_dir / f"{point.name}.run",
                )
                save_checkpoint(points_dir, point.name, row)
            rows.append(row)
            stage_rows.append(row)
            alarms = ((row.get("pointstream") or {}).get("late_frame") or {}).get("alarms") or []
            for alarm in alarms:
                category_notes.append(f"{point.name}: {alarm}")
                print(f"  ALARM {alarm}", flush=True)
            ps = row.get("pointstream") or {}
            print(
                f"  {point.name:<16} {ps.get('coded_bytes', row.get('pointstream_error', '—'))} B  "
                f"{(ps.get('scores') or {}).get('vmaf', '—')}",
                flush=True,
            )
            write_json(dest, _report())
        key = intended_category(stage)
        # A one-point preflight cannot show a knob moving; skip that assertion.
        if (
            stage != "controls"
            and len(stage_rows) > 1
            and not ledger_moved(stage_rows, key=key)
        ):
            note = (
                f"stage {stage}: intended ledger key {key!r} did not move "
                "across points. The knob is not reaching the payload, or the "
                "points are not distinct."
            )
            category_notes.append(note)
            print(f"  ALARM {note}", flush=True)

    if not args.skip_compare:
        ref_file = references_path(identity)
        if not ref_file.is_file():
            raise SystemExit(
                f"{ref_file} does not exist. Encode the independent "
                f"{args.codec} curve first: python -m experiments.tier.low_rate_references "
                f"--video {args.video} --scenes {' '.join(args.scenes)} "
                f"--frames {args.frames} --codec {args.codec}"
            )
        candidates = _candidate_rows(rows)
        for pattern in ACCESS_PATTERNS:
            comparisons[pattern] = compare_candidate_to_anchor(
                candidates,
                load_reference_curve(
                    ref_file, access_pattern=pattern, expected=identity
                ),
            )

    write_json(dest, _report())
    print(f"wrote {dest}", flush=True)
    return 1 if category_notes or completion_counts(rows)["failed"] or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
