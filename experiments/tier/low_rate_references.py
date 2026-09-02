"""Independent AV1/VVC reference curves for the low-rate search.

The reference QP walk is ``probe_qps``, not PointStream's residual knob. Each
codec encodes the same frames, size, frame rate and colour, once as a joint
continuous sequence and once as independently decodable segments. Generation
is not involved.

Run::

    python -m experiments.tier.low_rate_references \\
        --video alcaraz_highlights --scenes scene_000 scene_028 --frames 48
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.low_rate_checkpoint import load_checkpoint, save_checkpoint, write_json
from experiments.tier.low_rate_checkpoint import guard_checkpoints, implementation_digest, source_identity
from experiments.tier.low_rate_clips import (
    DEFAULT_SCENES,
    DEFAULT_SPAN_FRAMES,
    DEFAULT_VIDEO,
    load_e1_sequence,
)
from experiments.tier.low_rate_identity import (
    assert_same_input,
    checkpoint_dir,
    input_identity,
    references_path,
)
from experiments.tier.low_rate_measure import (
    late_frame_report,
    primary_preset,
    reference_request,
    score_headlines,
    timed_roundtrip,
    timing_record,
)
from experiments.tier.low_rate_plan import all_points
from experiments.tier.low_rate_validate import (
    BOUNDS_PATH,
    DECLARED_FPS,
    OUT_DIR,
    PRIMARY_ANCHORS,
    decode_rejections,
    probe_qps,
)
from src.components.codec.frames import even_size
from src.components.metrics.bd_rate import (
    InsufficientOverlapError,
    OperatingPoint,
    RDCurve,
    compare_rd_curves,
    meets_or_beats_floor,
)
from src.contracts.codecs import EncodeRequest
from src.contracts.metrics import metric as metric_spec


ACCESS_PATTERNS: tuple[str, ...] = ("continuous", "segmented")


def residual_qps_in_plan() -> frozenset[int]:
    return frozenset(
        int(point.residual_qp) for point in all_points() if point.residual_qp is not None
    )


def reference_qps(codec: str) -> tuple[int, ...]:
    """The independent reference walk. Not the residual-QP column of the plan."""
    qps = probe_qps(codec)
    residual = residual_qps_in_plan()
    if residual and set(qps) <= residual:
        raise ValueError(
            f"{codec} reference QPs {qps} are a subset of the residual plan "
            f"{sorted(residual)}. The reference curve would be slaved to the "
            "correction knob."
        )
    return qps


def _joined_frames(clips: list[Any]) -> np.ndarray:
    return even_size(
        np.concatenate([np.asarray(clip.frames, dtype=np.uint8) for clip in clips], axis=0)
    )


def _usable_point(
    *,
    source: np.ndarray,
    decoded: np.ndarray,
    size_bytes: int,
) -> tuple[bool, list[str], dict[str, float | str], dict[str, Any]]:
    reasons = decode_rejections(
        bitstream_bytes=int(size_bytes),
        source_shape=(
            int(source.shape[0]),
            int(source.shape[1]),
            int(source.shape[2]),
            int(source.shape[3]),
        ),
        decoded_shape=tuple(int(dim) for dim in decoded.shape),
    )
    if reasons:
        return False, reasons, {}, {}
    scores = score_headlines(source, decoded)
    late = late_frame_report(source, decoded)
    usable = isinstance(scores.get("vmaf"), float)
    if not usable:
        reasons = [str(scores.get("vmaf_error", "VMAF missing"))]
    return usable, reasons, scores, late


def encode_continuous(
    clips: list[Any],
    request: EncodeRequest,
    *,
    fps: float,
) -> dict[str, Any]:
    source = _joined_frames(clips)
    trip = timed_roundtrip(source, request=request, fps=fps)
    usable, reasons, scores, late = _usable_point(
        source=source, decoded=trip.frames, size_bytes=trip.size_bytes
    )
    row: dict[str, Any] = {
        "access_pattern": "continuous",
        "qp": int(request.rate or 0),
        "preset": request.preset,
        "bytes": int(trip.size_bytes),
        "n_frames": int(source.shape[0]),
        "usable": usable,
        "rejections": reasons,
        "scores": scores,
        "late_frame": late,
        "tool_path": trip.tool_path,
        "tool_version": trip.tool_version,
        **timing_record(trip),
    }
    return row


def encode_segmented(
    clips: list[Any],
    request: EncodeRequest,
    *,
    fps: float,
) -> dict[str, Any]:
    decoded_parts: list[np.ndarray] = []
    total_bytes = 0
    encode_seconds = 0.0
    decode_seconds = 0.0
    parts: list[dict[str, Any]] = []
    for clip in clips:
        source = even_size(np.asarray(clip.frames, dtype=np.uint8))
        trip = timed_roundtrip(source, request=request, fps=fps)
        total_bytes += int(trip.size_bytes)
        encode_seconds += float(trip.encode_seconds)
        decode_seconds += float(trip.decode_seconds)
        decoded_parts.append(trip.frames)
        parts.append(
            {
                "scene": getattr(clip, "scene", None),
                "bytes": int(trip.size_bytes),
                "n_frames": int(source.shape[0]),
                **timing_record(trip),
            }
        )
    source = _joined_frames(clips)
    decoded = np.concatenate(decoded_parts, axis=0)
    usable, reasons, scores, late = _usable_point(
        source=source, decoded=decoded, size_bytes=total_bytes
    )
    return {
        "access_pattern": "segmented",
        "qp": int(request.rate or 0),
        "preset": request.preset,
        "bytes": int(total_bytes),
        "n_frames": int(source.shape[0]),
        "usable": usable,
        "rejections": reasons,
        "scores": scores,
        "late_frame": late,
        "parts": parts,
        "encode_seconds": round(encode_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
    }


def encode_reference_curve(
    clips: list[Any],
    *,
    codec: str,
    preset: str,
    qps: tuple[int, ...],
    fps: float,
    checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    curves: dict[str, list[dict[str, Any]]] = {name: [] for name in ACCESS_PATTERNS}
    encoders = {
        "continuous": encode_continuous,
        "segmented": encode_segmented,
    }
    for qp in qps:
        request = reference_request(codec, qp, preset)
        for pattern, encode_fn in encoders.items():
            name = f"{pattern}-qp{qp}"
            existing = (
                load_checkpoint(checkpoint_root, name) if checkpoint_root is not None else None
            )
            if existing is not None:
                print(f"  resume {name}", flush=True)
                curves[pattern].append(existing)
                continue
            print(f"  {codec} qp{qp} preset {preset} {pattern}", flush=True)
            row = encode_fn(clips, request, fps=fps)
            if checkpoint_root is not None:
                save_checkpoint(checkpoint_root, name, row)
            curves[pattern].append(row)
    return {
        "codec": codec,
        "preset": preset,
        "qps": list(qps),
        "residual_qps_in_plan": sorted(residual_qps_in_plan()),
        "qps_are_independent_of_residual": True,
        "access_patterns": curves,
    }


def _vmaf_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("usable") and isinstance((row.get("scores") or {}).get("vmaf"), float)
    ]


def curve_from_rows(rows: list[dict[str, Any]], *, label: str) -> RDCurve | None:
    usable = _vmaf_points(rows)
    if len(usable) < 2:
        return None
    spec = metric_spec("vmaf")
    return RDCurve(
        rates=tuple(float(row["bytes"]) for row in usable),
        qualities=tuple(float(row["scores"]["vmaf"]) for row in usable),
        label=label,
        quality_spec=spec,
    )


def compare_candidate_to_anchor(
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """BD-rate on VMAF, or the floor test when the curves do not overlap.

    A missing BD-rate is not filled in. ``meets_or_beats_floor`` is the Gate A
    substitute at the anchor's smallest valid point.
    """
    spec = metric_spec("vmaf")
    anchor_usable = _vmaf_points(anchor_rows)
    candidate_usable = _vmaf_points(candidate_rows)
    report: dict[str, Any] = {
        "quality_metric": spec.name,
        "bd_rate_percent": None,
        "meets_or_beats_floor": None,
        "n_anchor": len(anchor_usable),
        "n_candidate": len(candidate_usable),
    }
    if not anchor_usable:
        report["reason"] = "anchor has no usable VMAF points"
        return report
    floor_row = min(anchor_usable, key=lambda row: int(row["bytes"]))
    floor = OperatingPoint(
        rate=float(floor_row["bytes"]),
        quality=float(floor_row["scores"]["vmaf"]),
    )
    report["anchor_floor"] = {
        "bytes": floor_row["bytes"],
        "vmaf": floor_row["scores"]["vmaf"],
        "qp": floor_row.get("qp"),
    }
    beats = [
        meets_or_beats_floor(
            OperatingPoint(
                rate=float(row["bytes"]),
                quality=float(row["scores"]["vmaf"]),
            ),
            floor,
            spec,
        )
        for row in candidate_usable
    ]
    report["meets_or_beats_floor"] = bool(beats) and any(beats)

    anchor_curve = curve_from_rows(anchor_rows, label="anchor")
    candidate_curve = curve_from_rows(candidate_rows, label="pointstream")
    if anchor_curve is None or candidate_curve is None:
        report["reason"] = "need at least two usable VMAF points on both curves"
        return report
    try:
        comparison = compare_rd_curves(anchor_curve, candidate_curve)
    except InsufficientOverlapError as exc:
        report["reason"] = str(exc)
        report["overlap"] = list(exc.overlap)
        report["overlap_fraction"] = exc.overlap_fraction
        return report
    report.update(
        {
            "bd_rate_percent": comparison.bd_rate_percent,
            "bd_quality": comparison.bd_quality,
            "overlap": list(comparison.overlap),
            "overlap_fraction": comparison.overlap_fraction,
            "reason": None,
        }
    )
    return report


def load_reference_curve(
    path: Path,
    *,
    access_pattern: str,
    expected: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected is not None:
        found = payload.get("input")
        if not isinstance(found, dict):
            raise SystemExit(
                f"{path} has no input identity. Re-run "
                "python -m experiments.tier.low_rate_references."
            )
        assert_same_input(found, expected)
    try:
        return list(payload["curve"]["access_patterns"][access_pattern])
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"{path} has no curve.access_patterns.{access_pattern}. "
            "Re-run python -m experiments.tier.low_rate_references."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--frames", type=int, default=DEFAULT_SPAN_FRAMES)
    parser.add_argument("--codecs", nargs="+", default=list(PRIMARY_ANCHORS))
    parser.add_argument(
        "--codec",
        default=None,
        help="encode one codec; overrides --codecs when set",
    )
    parser.add_argument("--preset", default=None, help="override the slowest-preset rule")
    parser.add_argument("--fps", type=float, default=DECLARED_FPS)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.fps != DECLARED_FPS:
        raise SystemExit("BP46 inputs and the PointStream runner require 24 fps")

    if not BOUNDS_PATH.is_file():
        raise SystemExit(f"{BOUNDS_PATH} does not exist. Bounds first.")

    clips = load_e1_sequence(args.video, list(args.scenes), n_frames=args.frames)
    joined = _joined_frames(clips)
    print(
        f"{args.video}: {len(clips)} scenes, {joined.shape[0]} frames "
        f"{joined.shape[2]}x{joined.shape[1]} @ {args.fps} fps",
        flush=True,
    )

    dest_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    codecs = [args.codec] if args.codec else list(args.codecs)
    for codec in codecs:
        identity = input_identity(
            video=args.video,
            scenes=list(args.scenes),
            frames_per_scene=args.frames,
            codec=codec,
            fps=args.fps,
        )
        dest = references_path(identity, root=dest_dir)
        points_dir = checkpoint_dir(dest)
        preset = primary_preset(codec, override=args.preset)
        qps = reference_qps(codec)
        identity["source"] = source_identity(clips)
        identity["preset"] = preset
        identity["implementation"] = implementation_digest()
        guard_checkpoints(points_dir, {"input": identity, "preset": preset, "qps": qps})
        print(f"{codec}: preset {preset}  qps {list(qps)}  {dest.name}", flush=True)
        curve = encode_reference_curve(
            clips,
            codec=codec,
            preset=preset,
            qps=qps,
            fps=float(args.fps),
            checkpoint_root=points_dir,
        )
        report = {
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": identity,
            "video": args.video,
            "scenes": list(args.scenes),
            "frames_per_scene": args.frames,
            "fps": args.fps,
            "resolution_policy": "native; no downscale, no frame drop",
            "colour": "yuv420p encode, RGB round-trip quality",
            "generation": "off",
            "preset_policy": (
                "slowest valid preset from codec-floor.json"
                if args.preset is None
                else f"OVERRIDDEN to {args.preset}"
            ),
            "checkpoint_dir": str(points_dir),
            "curve": curve,
        }
        write_json(dest, report)
        print(f"wrote {dest}", flush=True)
        n_bad = sum(
            1
            for pattern in ACCESS_PATTERNS
            for row in curve["access_patterns"][pattern]
            if not row.get("usable")
        )
        if n_bad:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
