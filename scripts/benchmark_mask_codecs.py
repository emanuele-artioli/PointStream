from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transport.disk import DiskTransport  # noqa: E402

DEFAULT_CODECS = (
    "auto",
    "rle-v1",
    "bitpack-z1",
    "png",
    "segmenter-native",
    "yolo-native",
)


@dataclass
class RunMetrics:
    requested_codec: str
    run_index: int
    elapsed_seconds: float
    metadata_size_bytes: int | None
    transport_total_size_bytes: int | None
    residual_size_bytes: int | None
    panorama_size_bytes: int | None
    source_size_bytes: int | None
    mask_frame_count: int
    actual_codec_counts: dict[str, int]
    output_dir: str
    chunk_id: str


def _slugify(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    safe = [ch if ch.isalnum() else "-" for ch in normalized]
    compact = "".join(safe).strip("-")
    return compact.replace("-", "_") or "codec"


def _find_default_input() -> str | None:
    candidate = PROJECT_ROOT / "assets" / "real_tennis.mp4"
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return None


def _parse_codecs(raw: str) -> list[str]:
    items = [piece.strip() for piece in raw.split(",")]
    return [item for item in items if item]


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_mask_codec_counts(output_dir: Path, chunk_id: str) -> tuple[int, dict[str, int]]:
    payload = DiskTransport(root_dir=output_dir).receive(chunk_id)
    counts: dict[str, int] = {}
    num_frames = 0
    for actor_packet in payload.actors:
        for frame_mask in actor_packet.mask_frames:
            codec_name = str(frame_mask.mask_codec)
            counts[codec_name] = counts.get(codec_name, 0) + 1
            num_frames += 1
    return num_frames, counts


def _build_main_command(
    *,
    output_dir: Path | None = None,
    chunk_id: str | None = None,
    summary_path: Path | None = None,
    input_video: str | None,
    num_frames: int,
    requested_codec: str,
) -> list[str]:
    _ = output_dir
    _ = chunk_id
    _ = summary_path

    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--num-frames",
        str(num_frames),
        "--disable-genai",
        "--disable-debug-keyframes",
        "--compositing-mask-mode",
        "metadata-source-mask",
        "--metadata-mask-codec",
        requested_codec,
        "--actor-extractor",
        "real",
        "--pose-estimator",
        "yolo",
        "--segmenter",
        "yolo",
        "--ball-extractor",
        "mock",
        "--execution-pool",
        "inline",
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


def _run_single_ablation(
    *,
    output_dir: Path | None = None,
    chunk_id: str | None = None,
    input_video: str | None,
    num_frames: int,
    requested_codec: str,
    run_index: int,
) -> RunMetrics:
    summary_path = (output_dir / "run_summary.json") if output_dir is not None else None
    run_output_root: Path
    summary_chunk_id: str

    cmd = _build_main_command(
        output_dir=output_dir,
        chunk_id=chunk_id,
        summary_path=summary_path,
        input_video=input_video,
        num_frames=num_frames,
        requested_codec=requested_codec,
    )

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        msg = [
            f"Ablation run failed for codec '{requested_codec}' (run {run_index}).",
            f"Command: {' '.join(cmd)}",
            f"Exit code: {completed.returncode}",
            "--- STDOUT ---",
            completed.stdout[-4000:],
            "--- STDERR ---",
            completed.stderr[-4000:],
        ]
        raise RuntimeError("\n".join(msg))

    if summary_path is not None:
        try:
            summary = _load_summary(summary_path)
            if output_dir is None:
                raise RuntimeError("output_dir must be provided when summary_path is set")
            run_output_root = output_dir
            summary_chunk_id = chunk_id or "0001"
        except FileNotFoundError:
            summary = _extract_summary_from_stdout(completed.stdout)
            run_output_root = Path(str(summary["run_output_root"]))
            summary_chunk_id = str(summary["chunk_id"])
    else:
        summary = _extract_summary_from_stdout(completed.stdout)
        run_output_root = Path(str(summary["run_output_root"]))
        summary_chunk_id = str(summary["chunk_id"])

    mask_frame_count, actual_codec_counts = _load_mask_codec_counts(
        output_dir=run_output_root,
        chunk_id=summary_chunk_id,
    )

    return RunMetrics(
        requested_codec=requested_codec,
        run_index=run_index,
        elapsed_seconds=float(elapsed),
        metadata_size_bytes=_coerce_int(summary.get("metadata_size_bytes")),
        transport_total_size_bytes=_coerce_int(summary.get("transport_total_size_bytes")),
        residual_size_bytes=_coerce_int(summary.get("residual_size_bytes")),
        panorama_size_bytes=_coerce_int(summary.get("panorama_size_bytes")),
        source_size_bytes=_coerce_int(summary.get("source_size_bytes")),
        mask_frame_count=mask_frame_count,
        actual_codec_counts=actual_codec_counts,
        output_dir=str(run_output_root),
        chunk_id=summary_chunk_id,
    )


def _mean_int(values: list[int | None]) -> int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return int(round(statistics.fmean(filtered)))


def _format_codec_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    parts = [f"{name}:{count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return ",".join(parts)


def _aggregate_results(rows: list[RunMetrics]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunMetrics]] = {}
    for row in rows:
        grouped.setdefault(row.requested_codec, []).append(row)

    ordered_codecs = []
    for codec in DEFAULT_CODECS:
        if codec in grouped:
            ordered_codecs.append(codec)
    ordered_codecs.extend([codec for codec in grouped if codec not in ordered_codecs])

    aggregates: list[dict[str, Any]] = []
    for codec in ordered_codecs:
        entries = grouped[codec]
        elapsed = [entry.elapsed_seconds for entry in entries]
        codec_counts: dict[str, int] = {}
        for entry in entries:
            for actual_codec, count in entry.actual_codec_counts.items():
                codec_counts[actual_codec] = codec_counts.get(actual_codec, 0) + int(count)

        aggregate = {
            "requested_codec": codec,
            "runs": len(entries),
            "avg_elapsed_seconds": float(statistics.fmean(elapsed)),
            "stdev_elapsed_seconds": float(statistics.pstdev(elapsed)) if len(elapsed) > 1 else 0.0,
            "avg_metadata_size_bytes": _mean_int([entry.metadata_size_bytes for entry in entries]),
            "avg_transport_total_size_bytes": _mean_int([entry.transport_total_size_bytes for entry in entries]),
            "avg_residual_size_bytes": _mean_int([entry.residual_size_bytes for entry in entries]),
            "avg_panorama_size_bytes": _mean_int([entry.panorama_size_bytes for entry in entries]),
            "avg_source_size_bytes": _mean_int([entry.source_size_bytes for entry in entries]),
            "avg_mask_frame_count": float(statistics.fmean([float(entry.mask_frame_count) for entry in entries])),
            "actual_codec_counts": codec_counts,
            "actual_codec_counts_str": _format_codec_counts(codec_counts),
        }
        aggregates.append(aggregate)

    baseline = next((entry for entry in aggregates if entry["requested_codec"] == "rle-v1"), None)
    baseline_metadata = cast(int | None, baseline["avg_metadata_size_bytes"]) if baseline is not None else None
    baseline_elapsed = cast(float | None, baseline["avg_elapsed_seconds"]) if baseline is not None else None

    for aggregate in aggregates:
        metadata = cast(int | None, aggregate["avg_metadata_size_bytes"])
        if baseline_metadata is not None and baseline_metadata > 0 and metadata is not None:
            aggregate["metadata_vs_rle_percent"] = ((float(metadata) - float(baseline_metadata)) / float(baseline_metadata)) * 100.0
        else:
            aggregate["metadata_vs_rle_percent"] = None

        elapsed_avg = cast(float, aggregate["avg_elapsed_seconds"])
        if baseline_elapsed is not None and baseline_elapsed > 0:
            aggregate["elapsed_vs_rle_percent"] = ((elapsed_avg - float(baseline_elapsed)) / float(baseline_elapsed)) * 100.0
        else:
            aggregate["elapsed_vs_rle_percent"] = None

    return aggregates


def _write_per_run_csv(path: Path, rows: list[RunMetrics]) -> None:
    fieldnames = [
        "requested_codec",
        "run_index",
        "elapsed_seconds",
        "metadata_size_bytes",
        "transport_total_size_bytes",
        "residual_size_bytes",
        "panorama_size_bytes",
        "source_size_bytes",
        "mask_frame_count",
        "actual_codec_counts",
        "output_dir",
        "chunk_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = asdict(row)
            row_dict["actual_codec_counts"] = _format_codec_counts(row.actual_codec_counts)
            writer.writerow(row_dict)


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "requested_codec",
        "runs",
        "avg_elapsed_seconds",
        "stdev_elapsed_seconds",
        "avg_metadata_size_bytes",
        "avg_transport_total_size_bytes",
        "avg_residual_size_bytes",
        "avg_panorama_size_bytes",
        "avg_source_size_bytes",
        "avg_mask_frame_count",
        "metadata_vs_rle_percent",
        "elapsed_vs_rle_percent",
        "actual_codec_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "requested_codec": row["requested_codec"],
                    "runs": row["runs"],
                    "avg_elapsed_seconds": row["avg_elapsed_seconds"],
                    "stdev_elapsed_seconds": row["stdev_elapsed_seconds"],
                    "avg_metadata_size_bytes": row["avg_metadata_size_bytes"],
                    "avg_transport_total_size_bytes": row["avg_transport_total_size_bytes"],
                    "avg_residual_size_bytes": row["avg_residual_size_bytes"],
                    "avg_panorama_size_bytes": row["avg_panorama_size_bytes"],
                    "avg_source_size_bytes": row["avg_source_size_bytes"],
                    "avg_mask_frame_count": row["avg_mask_frame_count"],
                    "metadata_vs_rle_percent": row["metadata_vs_rle_percent"],
                    "elapsed_vs_rle_percent": row["elapsed_vs_rle_percent"],
                    "actual_codec_counts": row["actual_codec_counts_str"],
                }
            )


def _human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < (1024 * 1024):
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _print_aggregate_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "requested_codec",
        "avg_metadata",
        "vs_rle(meta)",
        "avg_elapsed_s",
        "vs_rle(time)",
        "actual_mask_codecs",
    )
    print("\nMask codec ablation summary")
    print(" | ".join(header))
    print(" | ".join(["-" * len(piece) for piece in header]))
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["requested_codec"]),
                    _human_bytes(row["avg_metadata_size_bytes"]),
                    _format_delta(row.get("metadata_vs_rle_percent")),
                    f"{float(row['avg_elapsed_seconds']):.3f}",
                    _format_delta(row.get("elapsed_vs_rle_percent")),
                    str(row["actual_codec_counts_str"]),
                ]
            )
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mask codec ablation benchmarks over the PointStream metadata-source-mask path.",
    )
    parser.add_argument(
        "--input",
        dest="input_video",
        default=None,
        help="Input video path. Defaults to assets/real_tennis.mp4 when available.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of frames processed per ablation run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per codec.",
    )
    parser.add_argument(
        "--codecs",
        default=",".join(DEFAULT_CODECS),
        help="Comma-separated codec list. Example: auto,rle-v1,bitpack-z1,png,segmenter-native.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/bench_mask_codecs",
        help="Root folder where benchmark artifacts and reports are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be a positive integer")
    if args.repeats <= 0:
        raise ValueError("--repeats must be a positive integer")

    input_video = args.input_video
    if input_video is None:
        input_video = _find_default_input()
    if input_video is not None:
        resolved_input = Path(input_video).expanduser().resolve()
        if not resolved_input.exists() or not resolved_input.is_file():
            raise FileNotFoundError(f"Input video does not exist: {resolved_input}")
        input_video = str(resolved_input)

    codecs = _parse_codecs(str(args.codecs))
    if not codecs:
        raise ValueError("--codecs must include at least one codec")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(str(args.output_root)).expanduser().resolve() / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark output root: {output_root}")
    if input_video is not None:
        print(f"Input video: {input_video}")
    else:
        print("Input video: synthetic fallback clip generated by src.main")
    print(f"Codecs: {', '.join(codecs)}")
    print(f"Repeats per codec: {int(args.repeats)}")

    all_runs: list[RunMetrics] = []
    for codec in codecs:
        codec_slug = _slugify(codec)
        for run_idx in range(1, int(args.repeats) + 1):
            run_dir = output_root / f"{codec_slug}_run{run_idx}"
            chunk_id = f"bench_{codec_slug}_{run_idx}"
            print(f"Running codec={codec} repeat={run_idx} ...")
            metrics = _run_single_ablation(
                output_dir=run_dir,
                chunk_id=chunk_id,
                input_video=input_video,
                num_frames=int(args.num_frames),
                requested_codec=codec,
                run_index=run_idx,
            )
            all_runs.append(metrics)
            print(
                "  done: "
                f"elapsed={metrics.elapsed_seconds:.3f}s, "
                f"metadata={_human_bytes(metrics.metadata_size_bytes)}, "
                f"actual={_format_codec_counts(metrics.actual_codec_counts)}"
            )

    aggregates = _aggregate_results(all_runs)

    runs_csv_path = output_root / "mask_codec_ablation_runs.csv"
    aggregate_csv_path = output_root / "mask_codec_ablation_summary.csv"
    summary_json_path = output_root / "mask_codec_ablation_summary.json"

    _write_per_run_csv(path=runs_csv_path, rows=all_runs)
    _write_aggregate_csv(path=aggregate_csv_path, rows=aggregates)
    summary_json_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "input_video": input_video,
                "num_frames": int(args.num_frames),
                "repeats": int(args.repeats),
                "codecs": codecs,
                "runs": [asdict(row) for row in all_runs],
                "aggregate": aggregates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _print_aggregate_table(aggregates)
    print("\nArtifacts written:")
    print(f"- {runs_csv_path}")
    print(f"- {aggregate_csv_path}")
    print(f"- {summary_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
