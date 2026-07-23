"""Direct-codec anchor sweep: AV1/HEVC on the raw source video, no semantics.

Answers the reviewer-critical "missing baselines" gap (R2, R5,
reports/6_action_matrix.md): what does a conventional codec achieve on the
same tennis footage, at the same encoder settings PointStream already uses
for its own residual stream (see `ffmpeg-codec`/`codec-preset` in
config/default.yaml)? This is a *whole-video* anchor — no actor extraction,
no panorama, no metadata — so it sits at the opposite end of the spectrum
from the pipeline's Whole-Frame Residual Baseline (which still pays for
panorama + actor references). Compare the two together to place PointStream
on the rate-distortion map.

Reuses `src.encoder.video_io.encode_video_frames_ffmpeg` (the exact FFmpeg
wrapper the pipeline uses to write residual.mp4) and
`src.shared.experiment_evaluation.evaluate_run_summary` (the same PSNR/SSIM/VMAF
code the pipeline evaluation uses), so numbers are apples-to-apples with any
`run_summary.json`.

Usage:
    # Full sweep, both codecs, default CRF ladder, default preset ("slow"):
    python -m scripts.codec_baseline_sweep --input assets/real_tennis.mp4

    # Multiple presets per codec (cross-producted with CRF), side-by-side
    # with a PointStream run:
    python -m scripts.codec_baseline_sweep \
        --preset libsvtav1=slow,veryslow --preset libx265=slow,veryslow \
        --pointstream-run outputs/20260710_120000_123456

    # Custom CRF ladder for one codec:
    python -m scripts.codec_baseline_sweep \
        --codecs libsvtav1 --crf libsvtav1=25,35,45

    # Quick smoke test (first 30 frames only):
    python -m scripts.codec_baseline_sweep --max-frames 30
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder.video_io import (  # noqa: E402
    encode_video_frames_ffmpeg,
    iter_video_frames_ffmpeg,
    probe_video_metadata,
)
from src.shared.experiment_evaluation import evaluate_run_summary  # noqa: E402

DEFAULT_CODECS: tuple[str, ...] = ("libsvtav1", "libx265")
DEFAULT_CRFS: dict[str, tuple[int, ...]] = {
    "libsvtav1": (20, 30, 40, 50),
    "libx265": (18, 23, 28, 33),
}
CODEC_LABELS: dict[str, str] = {"libsvtav1": "AV1", "libx265": "HEVC"}
# A research-anchor baseline needs each codec at its own well-effort operating
# point, not the real-time preset PointStream uses for its own residual
# stream (config/default.yaml's codec-preset: fast). Naming asymmetry across
# encoders makes exact effort-parity impossible, but "slow" is the
# conventional x264-derived name both libsvtav1's preset-alias table and
# libx265 recognize as "thorough, well past real-time" — the standard choice
# for RD-curve baselines in the literature (see 2026-07-10 finding in
# reports/9_codec_baselines_report.md: preset "fast" on both codecs made
# HEVC dominate AV1, which is not the expected shape).
DEFAULT_PRESET = "slow"


@dataclass
class SweepPoint:
    codec: str
    codec_label: str
    crf: int
    preset: str
    elapsed_sec: float
    output_bytes: int
    source_bytes: int
    psnr_mean: float | None
    ssim_mean: float | None
    vmaf_mean: float | None
    output_path: str


EncodeFn = Callable[[Path, Path, str, int, str, "int | None"], float]
EvaluateFn = Callable[[Path, Path, "int | None"], dict[str, Any]]


def _find_default_input() -> Path | None:
    candidate = PROJECT_ROOT / "assets" / "real_tennis.mp4"
    return candidate if candidate.is_file() else None


def _parse_kv_list(raw: list[str] | None) -> dict[str, str]:
    """Parse repeated `KEY=VALUE` CLI args (e.g. `--crf libsvtav1=20,30,40`)."""
    result: dict[str, str] = {}
    for entry in raw or []:
        if "=" not in entry:
            raise ValueError(f"Expected KEY=VALUE, got: {entry!r}")
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in: {entry!r}")
        result[key] = value.strip()
    return result


def _parse_crf_overrides(raw: list[str] | None) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for codec, value in _parse_kv_list(raw).items():
        overrides[codec] = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    return overrides


def _parse_preset_overrides(raw: list[str] | None) -> dict[str, list[str]]:
    """Parse `--preset CODEC=p1,p2,...` into a per-codec preset ladder (cross-producted with CRF)."""
    overrides: dict[str, list[str]] = {}
    for codec, value in _parse_kv_list(raw).items():
        overrides[codec] = [piece.strip() for piece in value.split(",") if piece.strip()]
    return overrides


def _encode_point(
    input_path: Path,
    output_path: Path,
    codec: str,
    crf: int,
    preset: str,
    max_frames: int | None,
) -> float:
    metadata = probe_video_metadata(input_path)
    frames = iter_video_frames_ffmpeg(input_path, width=metadata.width, height=metadata.height)
    if max_frames is not None:
        frames = itertools.islice(frames, int(max_frames))

    started = time.perf_counter()
    encode_video_frames_ffmpeg(
        output_path,
        frames,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        codec=codec,
        crf=crf,
        preset=preset,
    )
    return time.perf_counter() - started


def _evaluate_point(input_path: Path, output_path: Path, max_frames: int | None) -> dict[str, Any]:
    summary = {
        "source_uri": str(input_path),
        "decoded_uri": str(output_path),
        "evaluation": {"sizes_bytes": {}},
    }
    return evaluate_run_summary(
        summary,
        experiment_dir=output_path.parent,
        max_frames=max_frames,
        metrics=["psnr", "ssim", "vmaf"],
    )


def run_sweep(
    input_path: Path,
    output_root: Path,
    *,
    codecs: list[str],
    crf_overrides: dict[str, list[int]],
    preset_overrides: dict[str, list[str]],
    max_frames: int | None = None,
    encode_fn: EncodeFn = _encode_point,
    evaluate_fn: EvaluateFn = _evaluate_point,
) -> list[SweepPoint]:
    output_root.mkdir(parents=True, exist_ok=True)
    source_bytes = input_path.stat().st_size

    points: list[SweepPoint] = []
    for codec in codecs:
        crfs = crf_overrides.get(codec) or list(DEFAULT_CRFS.get(codec, (30,)))
        presets = preset_overrides.get(codec) or [DEFAULT_PRESET]
        for preset in presets:
            for crf in crfs:
                label = f"{codec}_{preset}_crf{crf}"
                output_path = output_root / f"{label}.mp4"
                print(f"[{label}] encoding...", flush=True)
                elapsed = encode_fn(input_path, output_path, codec, crf, preset, max_frames)
                output_bytes = output_path.stat().st_size
                print(
                    f"[{label}] encoded in {elapsed:.1f}s -> {output_bytes:,} bytes; evaluating quality...",
                    flush=True,
                )
                evaluation = evaluate_fn(input_path, output_path, max_frames)
                point = SweepPoint(
                    codec=codec,
                    codec_label=CODEC_LABELS.get(codec, codec),
                    crf=crf,
                    preset=preset,
                    elapsed_sec=elapsed,
                    output_bytes=output_bytes,
                    source_bytes=source_bytes,
                    psnr_mean=evaluation.get("psnr_mean"),
                    ssim_mean=evaluation.get("ssim_mean"),
                    vmaf_mean=evaluation.get("vmaf_mean"),
                    output_path=str(output_path),
                )
                points.append(point)
                print(
                    f"[{label}] psnr={point.psnr_mean} ssim={point.ssim_mean} vmaf={point.vmaf_mean}",
                    flush=True,
                )
    return points


def _load_pointstream_point(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = summary.get("evaluation") or {}
    sizes = evaluation.get("sizes_bytes") or {}
    return {
        "label": "pointstream (semantic decomposition)",
        "transport_total_bytes": sizes.get("transport_total"),
        "source_bytes": sizes.get("source"),
        "psnr_mean": evaluation.get("psnr_mean"),
        "ssim_mean": evaluation.get("ssim_mean"),
        "vmaf_mean": evaluation.get("vmaf_mean"),
        "run_dir": str(run_dir),
    }


def _fmt_ratio(output_bytes: int, source_bytes: int) -> str:
    return f"{output_bytes / source_bytes:.4f}" if source_bytes else "—"


def _fmt_quality(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "null"


def build_report(
    points: list[SweepPoint],
    *,
    input_path: Path,
    pointstream_point: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    lines: list[str] = [
        "# Codec baseline sweep (AV1 / HEVC anchors, whole video, no semantics)",
        "",
        f"Source: `{input_path}`",
        "",
        "| codec | crf | preset | bytes | ratio-to-source | psnr | ssim | vmaf | wall (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(points, key=lambda p: (p.codec_label, p.preset, p.crf))
    for p in ordered:
        lines.append(
            f"| {p.codec_label} | {p.crf} | {p.preset} | {p.output_bytes:,} "
            f"| {_fmt_ratio(p.output_bytes, p.source_bytes)} | {_fmt_quality(p.psnr_mean)} "
            f"| {_fmt_quality(p.ssim_mean)} | {_fmt_quality(p.vmaf_mean)} | {p.elapsed_sec:.0f} |"
        )

    if pointstream_point is not None:
        lines.append("")
        lines.append("## PointStream anchor (for comparison)")
        lines.append("")
        lines.append("| label | transport_total bytes | ratio-to-source | psnr | ssim | vmaf | run_dir |")
        lines.append("|---|---|---|---|---|---|---|")
        transport = pointstream_point.get("transport_total_bytes")
        source = pointstream_point.get("source_bytes")
        ratio = _fmt_ratio(int(transport), int(source)) if transport is not None and source else "—"
        lines.append(
            f"| {pointstream_point['label']} | {transport if transport is not None else '—'} | {ratio} "
            f"| {_fmt_quality(pointstream_point.get('psnr_mean'))} "
            f"| {_fmt_quality(pointstream_point.get('ssim_mean'))} "
            f"| {_fmt_quality(pointstream_point.get('vmaf_mean'))} | `{pointstream_point['run_dir']}` |"
        )

    markdown = "\n".join(lines) + "\n"
    payload: dict[str, Any] = {
        "source": str(input_path),
        "points": [asdict(p) for p in ordered],
        "pointstream_point": pointstream_point,
    }
    return markdown, payload


def _write_csv(path: Path, points: list[SweepPoint]) -> None:
    fieldnames = list(asdict(points[0]).keys()) if points else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(asdict(p))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=None, help="Input video. Defaults to assets/real_tennis.mp4.")
    parser.add_argument(
        "--codecs",
        default=",".join(DEFAULT_CODECS),
        help="Comma-separated FFmpeg codec names, e.g. libsvtav1,libx265.",
    )
    parser.add_argument(
        "--crf",
        action="append",
        default=None,
        help="Per-codec CRF ladder override: CODEC=V1,V2,... (repeatable).",
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=None,
        help=(
            "Per-codec preset ladder override: CODEC=p1,p2,... (repeatable). "
            f"Cross-producted with --crf. Default: [{DEFAULT_PRESET!r}] for all codecs."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit encode+evaluation to the first N frames (smoke testing).",
    )
    parser.add_argument(
        "--pointstream-run",
        default=None,
        help="A PointStream outputs/<timestamp> run dir to report side-by-side.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/codec_baselines",
        help="Root folder for encoded anchors and reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_video = args.input or (_find_default_input() and str(_find_default_input()))
    if input_video is None:
        raise FileNotFoundError("No --input given and assets/real_tennis.mp4 not found.")
    input_path = Path(input_video).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    codecs = [c.strip() for c in str(args.codecs).split(",") if c.strip()]
    if not codecs:
        raise ValueError("--codecs must include at least one codec")

    crf_overrides = _parse_crf_overrides(args.crf)
    preset_overrides = _parse_preset_overrides(args.preset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(str(args.output_root)).expanduser().resolve() / timestamp

    print(f"Input: {input_path}")
    print(f"Codecs: {codecs}")
    print(f"Output root: {output_root}")

    points = run_sweep(
        input_path,
        output_root,
        codecs=codecs,
        crf_overrides=crf_overrides,
        preset_overrides=preset_overrides,
        max_frames=args.max_frames,
    )

    pointstream_point = None
    if args.pointstream_run:
        pointstream_point = _load_pointstream_point(Path(args.pointstream_run).expanduser().resolve())
        if pointstream_point is None:
            print(f"WARNING: no run_summary.json found under {args.pointstream_run}")

    markdown, payload = build_report(points, input_path=input_path, pointstream_point=pointstream_point)
    print()
    print(markdown)

    (output_root / "report.md").write_text(markdown, encoding="utf-8")
    (output_root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if points:
        _write_csv(output_root / "sweep.csv", points)

    print(f"Report written to {output_root / 'report.md'} and {output_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
