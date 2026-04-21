from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transport.disk import DiskTransport  # noqa: E402

DEFAULT_DETECTORS = ("yolo26", "yoloe")
DEFAULT_SEGMENTERS = ("yolo", "yoloe", "sam3", "none")


@dataclass
class AblationRun:
    detector: str
    segmenter: str
    run_index: int
    status: str
    elapsed_seconds: float
    output_dir: str
    chunk_id: str
    metadata_size_bytes: int | None = None
    transport_total_size_bytes: int | None = None
    residual_size_bytes: int | None = None
    panorama_size_bytes: int | None = None
    actor_reference_size_bytes: int | None = None
    num_actor_packets: int | None = None
    actor_reference_count: int | None = None
    actor_reference_fragmentation: float | None = None
    keyframe_events: int | None = None
    interpolate_events: int | None = None
    mask_frame_count: int | None = None
    mask_frame_coverage: float | None = None
    error_message: str | None = None


def _find_default_input() -> str | None:
    candidate = PROJECT_ROOT / "assets" / "real_tennis.mp4"
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return None


def _parse_csv_values(raw: str) -> list[str]:
    items = [part.strip() for part in raw.split(",")]
    return [item for item in items if item]


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_command(
    *,
    input_video: str | None,
    num_frames: int,
    detector: str,
    detector_caption: str,
    segmenter: str,
    segmenter_caption: str,
    ball_extractor: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--num-frames",
        str(num_frames),
        "--disable-genai",
        "--disable-debug-keyframes",
        "--execution-pool",
        "inline",
        "--actor-extractor",
        "real",
        "--detector",
        detector,
        "--detector-caption",
        detector_caption,
        "--pose-estimator",
        "yolo",
        "--segmenter",
        segmenter,
        "--segmenter-caption",
        segmenter_caption,
        "--ball-extractor",
        ball_extractor,
        "--compositing-mask-mode",
        "metadata-source-mask",
        "--metadata-mask-codec",
        "segmenter-native",
    ]
    if input_video is not None:
        cmd.extend(["--input", input_video])
    return cmd


def _extract_summary_from_stdout(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            parsed, _consumed = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "chunk_id" in parsed and "run_output_root" in parsed:
            candidates.append(parsed)

    if not candidates:
        raise ValueError("Could not parse run summary JSON from src.main stdout")
    return candidates[-1]


def _collect_payload_metrics(output_dir: Path, chunk_id: str, num_frames: int) -> dict[str, Any]:
    payload = DiskTransport(root_dir=output_dir).receive(chunk_id)

    keyframe_events = 0
    interpolate_events = 0
    mask_frame_count = 0
    for actor_packet in payload.actors:
        for event in actor_packet.events:
            event_type = str(getattr(event, "event_type", ""))
            if event_type == "keyframe":
                keyframe_events += 1
            elif event_type == "interpolate":
                interpolate_events += 1
        mask_frame_count += int(len(actor_packet.mask_frames))

    actor_packets = int(len(payload.actors))
    actor_reference_count = int(len(payload.actor_references))

    denominator = max(1, actor_packets)
    fragmentation = float(actor_reference_count) / float(denominator)

    coverage_denominator = max(1, int(num_frames) * denominator)
    mask_coverage = float(mask_frame_count) / float(coverage_denominator)

    return {
        "num_actor_packets": actor_packets,
        "actor_reference_count": actor_reference_count,
        "actor_reference_fragmentation": fragmentation,
        "keyframe_events": keyframe_events,
        "interpolate_events": interpolate_events,
        "mask_frame_count": mask_frame_count,
        "mask_frame_coverage": mask_coverage,
    }


def _run_one(
    *,
    output_dir: Path,
    chunk_id: str,
    input_video: str | None,
    num_frames: int,
    detector: str,
    detector_caption: str,
    segmenter: str,
    segmenter_caption: str,
    ball_extractor: str,
    run_index: int,
    allow_auto_model_download: bool,
) -> AblationRun:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(
        input_video=input_video,
        num_frames=num_frames,
        detector=detector,
        detector_caption=detector_caption,
        segmenter=segmenter,
        segmenter_caption=segmenter_caption,
        ball_extractor=ball_extractor,
    )

    env = os.environ.copy()
    env["POINTSTREAM_ALLOW_AUTO_MODEL_DOWNLOAD"] = "1" if allow_auto_model_download else "0"

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        tail_stdout = completed.stdout[-1800:]
        tail_stderr = completed.stderr[-1800:]
        return AblationRun(
            detector=detector,
            segmenter=segmenter,
            run_index=run_index,
            status="failed",
            elapsed_seconds=float(elapsed),
            output_dir=str(output_dir),
            chunk_id=chunk_id,
            error_message=(
                f"return_code={completed.returncode}; stdout_tail={tail_stdout!r}; stderr_tail={tail_stderr!r}"
            ),
        )

    summary = _extract_summary_from_stdout(completed.stdout)
    run_output_root = Path(str(summary["run_output_root"]))
    summary_chunk_id = str(summary["chunk_id"])
    payload_metrics = _collect_payload_metrics(
        output_dir=run_output_root,
        chunk_id=summary_chunk_id,
        num_frames=num_frames,
    )

    return AblationRun(
        detector=detector,
        segmenter=segmenter,
        run_index=run_index,
        status="ok",
        elapsed_seconds=float(elapsed),
        output_dir=str(run_output_root),
        chunk_id=summary_chunk_id,
        metadata_size_bytes=_safe_int(summary.get("metadata_size_bytes")),
        transport_total_size_bytes=_safe_int(summary.get("transport_total_size_bytes")),
        residual_size_bytes=_safe_int(summary.get("residual_size_bytes")),
        panorama_size_bytes=_safe_int(summary.get("panorama_size_bytes")),
        actor_reference_size_bytes=_safe_int(summary.get("actor_reference_size_bytes")),
        num_actor_packets=payload_metrics["num_actor_packets"],
        actor_reference_count=payload_metrics["actor_reference_count"],
        actor_reference_fragmentation=payload_metrics["actor_reference_fragmentation"],
        keyframe_events=payload_metrics["keyframe_events"],
        interpolate_events=payload_metrics["interpolate_events"],
        mask_frame_count=payload_metrics["mask_frame_count"],
        mask_frame_coverage=payload_metrics["mask_frame_coverage"],
    )


def _write_runs_csv(path: Path, rows: list[AblationRun]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(AblationRun(
        detector="",
        segmenter="",
        run_index=0,
        status="",
        elapsed_seconds=0.0,
        output_dir="",
        chunk_id="",
    )).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _aggregate(rows: list[AblationRun]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[AblationRun]] = {}
    for row in rows:
        grouped.setdefault((row.detector, row.segmenter), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (detector, segmenter), entries in grouped.items():
        ok_entries = [entry for entry in entries if entry.status == "ok"]
        fail_count = len(entries) - len(ok_entries)

        if ok_entries:
            avg_elapsed = float(statistics.fmean(entry.elapsed_seconds for entry in ok_entries))
            avg_fragmentation = float(
                statistics.fmean(float(entry.actor_reference_fragmentation or 0.0) for entry in ok_entries)
            )
            avg_mask_coverage = float(
                statistics.fmean(float(entry.mask_frame_coverage or 0.0) for entry in ok_entries)
            )
            avg_transport = float(
                statistics.fmean(float(entry.transport_total_size_bytes or 0) for entry in ok_entries)
            )
        else:
            avg_elapsed = float("inf")
            avg_fragmentation = float("inf")
            avg_mask_coverage = 0.0
            avg_transport = float("inf")

        aggregates.append(
            {
                "detector": detector,
                "segmenter": segmenter,
                "runs": len(entries),
                "successes": len(ok_entries),
                "failures": fail_count,
                "avg_elapsed_seconds": None if not ok_entries else avg_elapsed,
                "avg_actor_reference_fragmentation": None if not ok_entries else avg_fragmentation,
                "avg_mask_frame_coverage": None if not ok_entries else avg_mask_coverage,
                "avg_transport_total_size_bytes": None if not ok_entries else int(round(avg_transport)),
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[int, float, float, float]:
        has_success = 0 if item["successes"] > 0 else 1
        frag = float(item["avg_actor_reference_fragmentation"] or float("inf"))
        elapsed = float(item["avg_elapsed_seconds"] or float("inf"))
        mask_cov = -float(item["avg_mask_frame_coverage"] or 0.0)
        return (has_success, frag, elapsed, mask_cov)

    return sorted(aggregates, key=_sort_key)


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "detector",
        "segmenter",
        "runs",
        "successes",
        "failures",
        "avg_elapsed_seconds",
        "avg_actor_reference_fragmentation",
        "avg_mask_frame_coverage",
        "avg_transport_total_size_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nPlayer backend ablation summary")
    print("detector | segmenter | successes/runs | avg_fragmentation | avg_mask_coverage | avg_elapsed_s")
    print("---------|-----------|----------------|-------------------|-------------------|-------------")
    for row in rows:
        success_ratio = f"{row['successes']}/{row['runs']}"
        frag = "n/a" if row["avg_actor_reference_fragmentation"] is None else f"{row['avg_actor_reference_fragmentation']:.3f}"
        mask_cov = "n/a" if row["avg_mask_frame_coverage"] is None else f"{row['avg_mask_frame_coverage']:.3f}"
        elapsed = "n/a" if row["avg_elapsed_seconds"] is None else f"{row['avg_elapsed_seconds']:.2f}"
        print(f"{row['detector']} | {row['segmenter']} | {success_ratio} | {frag} | {mask_cov} | {elapsed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark detector/segmenter backend combinations for player tracking stability and mask quality.",
    )
    parser.add_argument("--input", type=str, default=_find_default_input(), help="Input video path for ablations.")
    parser.add_argument("--num-frames", type=int, default=48, help="Number of frames to process per run.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per backend combination.")
    parser.add_argument(
        "--detectors",
        type=str,
        default=",".join(DEFAULT_DETECTORS),
        help="Comma-separated detector backends (for example: yolo26,yoloe).",
    )
    parser.add_argument(
        "--segmenters",
        type=str,
        default=",".join(DEFAULT_SEGMENTERS),
        help="Comma-separated segmenter backends (for example: yolo,yoloe,sam3,none).",
    )
    parser.add_argument(
        "--detector-caption",
        type=str,
        default="tennis player",
        help="Detector caption prompt for open-vocabulary backends such as YOLOE.",
    )
    parser.add_argument(
        "--segmenter-caption",
        type=str,
        default="tennis player",
        help="Segmenter caption prompt for open-vocabulary backends such as YOLOE.",
    )
    parser.add_argument(
        "--ball-extractor",
        choices=("difference", "mock"),
        default="difference",
        help="Ball extractor backend used during each ablation run.",
    )
    parser.add_argument(
        "--allow-auto-model-download",
        action="store_true",
        help="Allow ultralytics to auto-download missing model weights.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/bench_player_backends",
        help="Output directory root for per-run artifacts and reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be a positive integer")
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")

    input_video = args.input
    if input_video is None:
        raise FileNotFoundError("No default input found. Provide --input with a valid video path.")

    input_path = Path(input_video).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    detectors = _parse_csv_values(args.detectors)
    segmenters = _parse_csv_values(args.segmenters)
    if not detectors:
        raise ValueError("No detector backends requested")
    if not segmenters:
        raise ValueError("No segmenter backends requested")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_root = Path(args.output_root).expanduser() / timestamp
    bench_root.mkdir(parents=True, exist_ok=True)

    rows: list[AblationRun] = []
    combo_index = 0
    for detector in detectors:
        for segmenter in segmenters:
            for run_idx in range(1, int(args.repeats) + 1):
                combo_index += 1
                slug = f"det_{detector}__seg_{segmenter}__run_{run_idx:02d}"
                output_dir = bench_root / slug
                chunk_id = f"bench_{combo_index:04d}"
                row = _run_one(
                    output_dir=output_dir,
                    chunk_id=chunk_id,
                    input_video=str(input_path),
                    num_frames=int(args.num_frames),
                    detector=detector,
                    detector_caption=str(args.detector_caption),
                    segmenter=segmenter,
                    segmenter_caption=str(args.segmenter_caption),
                    ball_extractor=str(args.ball_extractor),
                    run_index=int(run_idx),
                    allow_auto_model_download=bool(args.allow_auto_model_download),
                )
                rows.append(row)
                print(
                    f"[{len(rows):03d}] detector={detector} segmenter={segmenter} run={run_idx} "
                    f"status={row.status} elapsed={row.elapsed_seconds:.2f}s"
                )

    runs_csv = bench_root / "player_backend_ablation_runs.csv"
    _write_runs_csv(runs_csv, rows)

    aggregate_rows = _aggregate(rows)
    summary_csv = bench_root / "player_backend_ablation_summary.csv"
    _write_aggregate_csv(summary_csv, aggregate_rows)

    summary_json = bench_root / "player_backend_ablation_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "input": str(input_path),
                "num_frames": int(args.num_frames),
                "repeats": int(args.repeats),
                "detectors": detectors,
                "segmenters": segmenters,
                "allow_auto_model_download": bool(args.allow_auto_model_download),
                "runs_csv": str(runs_csv),
                "summary_csv": str(summary_csv),
                "rows": aggregate_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _print_summary(aggregate_rows)
    print(f"\nWrote reports to: {bench_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
