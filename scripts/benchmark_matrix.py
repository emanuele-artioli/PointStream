"""Residual-Guarantee benchmark harness.

Runs a matrix of pipeline configs against a shared baseline and reports, for
each variant, whether the component under test pays for itself: does it
shrink the residual payload by more than the semantic metadata it adds?
(See reports/7_implementation_plan.md §1 and reports/REPORTS.md.)

Usage:
    # Run a matrix spec (materializes configs, runs variants, writes report):
    python -m scripts.benchmark_matrix run config/benchmarks/example.yaml

    # Resume an interrupted matrix (completed variants are skipped):
    python -m scripts.benchmark_matrix run config/benchmarks/example.yaml \
        --dir outputs/benchmarks/example_20260709_120000

    # Re-report an existing benchmark dir:
    python -m scripts.benchmark_matrix report outputs/benchmarks/example_20260709_120000

    # Ad-hoc comparison of existing run dirs (first dir is the baseline):
    python -m scripts.benchmark_matrix report outputs/<ts_baseline> outputs/<ts_variant>

Matrix spec YAML:
    name: example                     # optional; defaults to spec filename stem
    input: assets/real_tennis.mp4     # required: real runs must pass an input
    base-config: config/default.yaml
    common-overrides:                 # optional, applied to every variant
      num-frames: null
    variants:
      - name: baseline
        baseline: true                # exactly one (defaults to the first)
        overrides:
          genai-backend: null
      - name: canny
        overrides:
          genai-backend: canny-controlnet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

# Config keys that must match across variants for the byte comparison to be
# like-with-like; differing values don't abort the run but are flagged in the
# report (a CRF sweep, for instance, deliberately varies them).
COMPARABILITY_KEYS = (
    "source-uri",
    "num-frames",
    "seed",
    "ffmpeg-codec",
    "codec-crf",
    "codec-preset",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    overrides: dict[str, Any]
    is_baseline: bool = False


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    input_video: str
    base_config: str
    common_overrides: dict[str, Any]
    variants: tuple[VariantSpec, ...]

    @property
    def baseline(self) -> VariantSpec:
        return next(v for v in self.variants if v.is_baseline)


@dataclass
class VariantResult:
    name: str
    is_baseline: bool
    status: str  # "ok" | "failed" | "skipped-existing"
    run_dir: str | None = None
    config_path: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    summary: dict[str, Any] | None = field(default=None, repr=False)


def _normalize_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    # Config loader accepts both spellings; normalize so overrides always win.
    return {str(k).replace("_", "-"): v for k, v in mapping.items()}


def load_matrix_spec(spec_path: str | Path) -> MatrixSpec:
    path = Path(spec_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Matrix spec root must be a mapping: {path}")
    raw = _normalize_keys(raw)

    input_video = raw.get("input")
    if not input_video:
        raise ValueError("Matrix spec requires 'input' (real runs must pass an input video)")

    raw_variants = raw.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) < 2:
        raise ValueError("Matrix spec requires at least two 'variants' (a baseline and a variant)")

    variants: list[VariantSpec] = []
    for entry in raw_variants:
        if not isinstance(entry, dict):
            raise ValueError(f"Each variant must be a mapping, got: {entry!r}")
        entry = _normalize_keys(entry)
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("Every variant needs a non-empty 'name'")
        overrides = _normalize_keys(entry.get("overrides") or {})
        variants.append(VariantSpec(name=name, overrides=overrides, is_baseline=bool(entry.get("baseline", False))))

    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"Variant names must be unique, got: {names}")

    baseline_count = sum(1 for v in variants if v.is_baseline)
    if baseline_count > 1:
        raise ValueError("At most one variant may set 'baseline: true'")
    if baseline_count == 0:
        variants[0] = VariantSpec(name=variants[0].name, overrides=variants[0].overrides, is_baseline=True)

    return MatrixSpec(
        name=str(raw.get("name") or path.stem),
        input_video=str(input_video),
        base_config=str(raw.get("base-config") or "config/default.yaml"),
        common_overrides=_normalize_keys(raw.get("common-overrides") or {}),
        variants=tuple(variants),
    )


def materialize_config(spec: MatrixSpec, variant: VariantSpec) -> dict[str, Any]:
    base_path = (PROJECT_ROOT / spec.base_config).resolve() if not Path(spec.base_config).is_absolute() else Path(spec.base_config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Base config root must be a mapping: {base_path}")

    merged = _normalize_keys(base)
    merged.update(spec.common_overrides)
    merged.update(variant.overrides)
    merged["summary-file"] = True  # the harness needs run_summary.json to exist
    return merged


def comparability_issues(configs: dict[str, dict[str, Any]]) -> list[str]:
    """Flag comparability keys whose values differ across materialized configs."""
    issues: list[str] = []
    for key in COMPARABILITY_KEYS:
        values = {name: cfg.get(key) for name, cfg in configs.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            rendered = ", ".join(f"{name}={value!r}" for name, value in values.items())
            issues.append(f"'{key}' differs across variants ({rendered}) — byte comparison is not like-with-like")
    return issues


def extract_summary_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Find the run summary dict printed by run_cli in (possibly noisy) stdout."""
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    index = 0
    while True:
        start = stdout.find("{", index)
        if start < 0:
            break
        try:
            candidate, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = start + max(end, 1)
        if isinstance(candidate, dict) and "run_output_root" in candidate and "evaluation" in candidate:
            result = candidate
    return result


def _run_variant_subprocess(config_path: Path, input_video: str, log_path: Path) -> tuple[int, str]:
    """Run one pipeline variant; returns (returncode, stdout). Full output is teed to log_path."""
    cmd = [sys.executable, "src/main.py", "--input", input_video, "--config", str(config_path)]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n=== STDOUT ===\n{completed.stdout}\n=== STDERR ===\n{completed.stderr}\n",
        encoding="utf-8",
    )
    return completed.returncode, completed.stdout


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_matrix(spec: MatrixSpec, bench_dir: Path, runner=_run_variant_subprocess) -> list[VariantResult]:
    configs_dir = bench_dir / "configs"
    logs_dir = bench_dir / "logs"
    runs_dir = bench_dir / "runs"
    for directory in (configs_dir, logs_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    input_path = (PROJECT_ROOT / spec.input_video) if not Path(spec.input_video).is_absolute() else Path(spec.input_video)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    materialized = {v.name: materialize_config(spec, v) for v in spec.variants}
    for issue in comparability_issues(materialized):
        print(f"WARNING: {issue}")

    results: list[VariantResult] = []
    for variant in spec.variants:
        config = materialized[variant.name]
        config_path = configs_dir / f"{variant.name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

        result = VariantResult(name=variant.name, is_baseline=variant.is_baseline, status="failed", config_path=str(config_path))
        results.append(result)

        run_link = runs_dir / variant.name
        if run_link.is_dir() and (run_link / "run_summary.json").is_file():
            result.status = "skipped-existing"
            result.run_dir = str(run_link.resolve())
            result.summary = _load_run_summary(run_link)
            print(f"[{variant.name}] already complete at {result.run_dir} — skipping")
            _write_manifest(bench_dir, spec, results)
            continue

        num_frames = config.get("num-frames")
        frames_note = "all frames" if num_frames is None else f"{num_frames} frames"
        print(f"[{variant.name}] running ({frames_note})...", flush=True)
        started = datetime.now()
        try:
            returncode, stdout = runner(config_path, str(input_path), logs_dir / f"{variant.name}.log")
        except Exception as exc:  # runner infrastructure failure, not a pipeline error
            result.error = f"runner raised: {exc}"
            print(f"[{variant.name}] FAILED: {result.error}")
            _write_manifest(bench_dir, spec, results)
            continue
        result.duration_sec = (datetime.now() - started).total_seconds()

        if returncode != 0:
            result.error = f"pipeline exited with code {returncode}; see {logs_dir / f'{variant.name}.log'}"
            print(f"[{variant.name}] FAILED: {result.error}")
            _write_manifest(bench_dir, spec, results)
            continue

        summary = extract_summary_from_stdout(stdout)
        if summary is None:
            result.error = f"could not locate run summary in pipeline stdout; see {logs_dir / f'{variant.name}.log'}"
            print(f"[{variant.name}] FAILED: {result.error}")
            _write_manifest(bench_dir, spec, results)
            continue

        run_dir = Path(str(summary["run_output_root"]))
        result.status = "ok"
        result.run_dir = str(run_dir)
        result.summary = _load_run_summary(run_dir) if (run_dir / "run_summary.json").is_file() else summary
        if not run_link.exists():
            run_link.symlink_to(run_dir, target_is_directory=True)
        print(f"[{variant.name}] done in {result.duration_sec:.0f}s → {run_dir}")
        _write_manifest(bench_dir, spec, results)

    return results


def _write_manifest(bench_dir: Path, spec: MatrixSpec, results: list[VariantResult]) -> None:
    manifest = {
        "name": spec.name,
        "input": spec.input_video,
        "base_config": spec.base_config,
        "common_overrides": spec.common_overrides,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "variants": [
            {
                "name": r.name,
                "baseline": r.is_baseline,
                "status": r.status,
                "run_dir": r.run_dir,
                "config": r.config_path,
                "duration_sec": r.duration_sec,
                "error": r.error,
                "overrides": next(v.overrides for v in spec.variants if v.name == r.name),
            }
            for r in results
        ],
    }
    (bench_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _sizes(summary: dict[str, Any]) -> dict[str, int]:
    sizes = (summary.get("evaluation") or {}).get("sizes_bytes") or {}
    return {k: int(v) for k, v in sizes.items() if isinstance(v, (int, float)) and k != "transport_to_source_ratio"}


def semantic_bytes(sizes: dict[str, int]) -> int:
    return sizes.get("metadata", 0) + sizes.get("actor_reference", 0) + sizes.get("panorama", 0)


def _quality(summary: dict[str, Any]) -> dict[str, float | None]:
    evaluation = summary.get("evaluation") or {}
    return {metric: evaluation.get(f"{metric}_mean") for metric in ("psnr", "ssim", "vmaf")}


def _fmt_bytes(value: int | None) -> str:
    return f"{value:,}" if value is not None else "—"


def _fmt_delta_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:+,}"


def _fmt_quality(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "null"


def build_report(results: list[VariantResult], *, matrix_name: str, warnings: list[str]) -> tuple[str, dict[str, Any]]:
    """Render the Residual-Guarantee comparison as (markdown, machine-readable dict)."""
    usable = [r for r in results if r.summary is not None]
    baseline = next((r for r in usable if r.is_baseline), None)

    lines: list[str] = [f"# Benchmark: {matrix_name}", ""]
    report_rows: list[dict[str, Any]] = []

    if baseline is None:
        lines.append("**No usable baseline run — cannot evaluate the Residual Guarantee.**")
        failed = [r for r in results if r.summary is None]
        for r in failed:
            lines.append(f"- `{r.name}`: {r.status} ({r.error or 'no summary'})")
        return "\n".join(lines) + "\n", {"name": matrix_name, "rows": [], "warnings": warnings}

    base_sizes = _sizes(baseline.summary or {})
    base_semantic = semantic_bytes(base_sizes)
    base_residual = base_sizes.get("residual", 0)
    base_total = base_sizes.get("transport_total", 0)
    base_quality = _quality(baseline.summary or {})

    for warning in warnings:
        lines.append(f"> ⚠️ {warning}")
    if warnings:
        lines.append("")

    num_frames = (baseline.summary or {}).get("num_frames")
    source = (baseline.summary or {}).get("source_uri")
    lines.append(f"Baseline: `{baseline.name}` | source: `{source}` | frames: {num_frames}")
    lines.append("")
    lines.append(
        "Verdict rule: a component **pays for itself** iff it cuts residual bytes by more "
        "than the semantic bytes (metadata + references + panorama) it adds, i.e. "
        "`transport_total` drops below the baseline's."
    )
    lines.append("")
    header = (
        "| variant | semantic | residual | total | Δsemantic | Δresidual saved | net vs baseline | "
        "ratio-to-source | psnr | ssim | vmaf | wall (s) | verdict |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 13)

    ordered = [baseline] + [r for r in usable if r is not baseline]
    for r in ordered:
        sizes = _sizes(r.summary or {})
        quality = _quality(r.summary or {})
        semantic = semantic_bytes(sizes)
        residual = sizes.get("residual", 0)
        total = sizes.get("transport_total", 0)
        source_bytes = sizes.get("source", 0)
        ratio = (total / source_bytes) if source_bytes else None
        wall = ((r.summary or {}).get("evaluation") or {}).get("timings_sec", {}).get("pipeline_total")

        if r is baseline:
            delta_semantic = delta_residual_saved = net = None
            verdict = "(baseline)"
        else:
            delta_semantic = semantic - base_semantic
            delta_residual_saved = base_residual - residual
            net = base_total - total
            verdict = "PAYS" if net > 0 else "DOES NOT PAY"
            quality_drop = (
                base_quality["vmaf"] is not None
                and quality["vmaf"] is not None
                and quality["vmaf"] < base_quality["vmaf"] - 1.0
            )
            if quality_drop:
                verdict += " ⚠️ quality drop"

        row = {
            "variant": r.name,
            "baseline": r.is_baseline,
            "status": r.status,
            "run_dir": r.run_dir,
            "semantic_bytes": semantic,
            "residual_bytes": residual,
            "transport_total_bytes": total,
            "delta_semantic_bytes": delta_semantic,
            "delta_residual_saved_bytes": delta_residual_saved,
            "net_bytes_vs_baseline": net,
            "transport_to_source_ratio": ratio,
            "psnr_mean": quality["psnr"],
            "ssim_mean": quality["ssim"],
            "vmaf_mean": quality["vmaf"],
            "pipeline_total_sec": wall,
            "verdict": verdict,
        }
        report_rows.append(row)

        lines.append(
            f"| {r.name} | {_fmt_bytes(semantic)} | {_fmt_bytes(residual)} | {_fmt_bytes(total)} "
            f"| {_fmt_delta_bytes(delta_semantic)} | {_fmt_delta_bytes(delta_residual_saved)} | {_fmt_delta_bytes(net)} "
            f"| {f'{ratio:.4f}' if ratio is not None else '—'} "
            f"| {_fmt_quality(quality['psnr'])} | {_fmt_quality(quality['ssim'])} | {_fmt_quality(quality['vmaf'])} "
            f"| {f'{wall:.0f}' if wall is not None else '—'} | {verdict} |"
        )

    failed = [r for r in results if r.summary is None]
    if failed:
        lines.append("")
        lines.append("## Failed / missing variants")
        for r in failed:
            lines.append(f"- `{r.name}`: {r.status} ({r.error or 'no summary'})")

    null_metrics = [r.name for r in usable if _quality(r.summary or {})["psnr"] is None]
    if null_metrics:
        lines.append("")
        lines.append(
            f"> ⚠️ PSNR is null for: {', '.join(null_metrics)} — evaluation found no valid frame "
            "pairs; treat those quality columns as a failure to investigate, not as a metric."
        )

    return "\n".join(lines) + "\n", {"name": matrix_name, "rows": report_rows, "warnings": warnings}


def results_from_bench_dir(bench_dir: Path) -> tuple[list[VariantResult], list[str], str]:
    manifest_path = bench_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a benchmark dir (no manifest.json): {bench_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results: list[VariantResult] = []
    configs: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("variants", []):
        result = VariantResult(
            name=entry["name"],
            is_baseline=bool(entry.get("baseline")),
            status=entry.get("status", "unknown"),
            run_dir=entry.get("run_dir"),
            config_path=entry.get("config"),
            duration_sec=entry.get("duration_sec"),
            error=entry.get("error"),
        )
        if result.run_dir and Path(result.run_dir).is_dir():
            try:
                result.summary = _load_run_summary(Path(result.run_dir))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                result.error = result.error or str(exc)
        if result.config_path and Path(result.config_path).is_file():
            configs[result.name] = _normalize_keys(yaml.safe_load(Path(result.config_path).read_text(encoding="utf-8")) or {})
        results.append(result)

    warnings = comparability_issues(configs) if len(configs) > 1 else []
    return results, warnings, str(manifest.get("name") or bench_dir.name)


def results_from_run_dirs(run_dirs: list[Path]) -> tuple[list[VariantResult], list[str]]:
    results: list[VariantResult] = []
    for index, run_dir in enumerate(run_dirs):
        result = VariantResult(name=run_dir.name, is_baseline=index == 0, status="ok", run_dir=str(run_dir))
        result.summary = _load_run_summary(run_dir)
        results.append(result)

    warnings = [
        "ad-hoc run dirs: codec/CRF/preset/seed cannot be verified from run summaries — "
        "confirm these runs used identical settings before trusting the byte comparison"
    ]
    sources = {(r.summary or {}).get("source_uri") for r in results}
    if len(sources) > 1:
        warnings.append(f"source videos differ across runs: {sorted(str(s) for s in sources)}")
    frames = {(r.summary or {}).get("num_frames") for r in results}
    if len(frames) > 1:
        warnings.append(f"frame counts differ across runs: {sorted(str(f) for f in frames)}")
    return results, warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_report(results: list[VariantResult], *, matrix_name: str, warnings: list[str], out_dir: Path | None) -> None:
    markdown, payload = build_report(results, matrix_name=matrix_name, warnings=warnings)
    print()
    print(markdown)
    if out_dir is not None:
        (out_dir / "report.md").write_text(markdown, encoding="utf-8")
        (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {out_dir / 'report.md'} and {out_dir / 'report.json'}")


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_matrix_spec(args.spec)
    if args.dir:
        bench_dir = Path(args.dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bench_dir = PROJECT_ROOT / "outputs" / "benchmarks" / f"{spec.name}_{timestamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark dir: {bench_dir}")

    results = run_matrix(spec, bench_dir)
    materialized = {v.name: materialize_config(spec, v) for v in spec.variants}
    _emit_report(results, matrix_name=spec.name, warnings=comparability_issues(materialized), out_dir=bench_dir)
    return 0 if all(r.status in {"ok", "skipped-existing"} for r in results) else 1


def _cmd_report(args: argparse.Namespace) -> int:
    paths = [Path(p).expanduser().resolve() for p in args.paths]
    if len(paths) == 1 and (paths[0] / "manifest.json").is_file():
        results, warnings, name = results_from_bench_dir(paths[0])
        _emit_report(results, matrix_name=name, warnings=warnings, out_dir=paths[0])
    else:
        results, warnings = results_from_run_dirs(paths)
        _emit_report(results, matrix_name="ad-hoc comparison", warnings=warnings, out_dir=None)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark matrix spec")
    run_parser.add_argument("spec", type=str, help="Path to the matrix spec YAML")
    run_parser.add_argument("--dir", type=str, default=None, help="Existing benchmark dir to resume into")
    run_parser.set_defaults(func=_cmd_run)

    report_parser = subparsers.add_parser("report", help="Report on a benchmark dir or ad-hoc run dirs")
    report_parser.add_argument("paths", nargs="+", type=str, help="One benchmark dir, or 2+ run dirs (first = baseline)")
    report_parser.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
