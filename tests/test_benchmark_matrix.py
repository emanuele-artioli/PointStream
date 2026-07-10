from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import benchmark_matrix as bench


def _write_spec(tmp_path: Path, body: dict[str, Any]) -> Path:
    spec_path = tmp_path / "matrix.yaml"
    spec_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return spec_path


def _base_config(tmp_path: Path, body: dict[str, Any] | None = None) -> Path:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(body or {"num-frames": 10, "codec-crf": 35}), encoding="utf-8")
    return config_path


def _spec_body(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "input": "assets/real_tennis.mp4",
        "base-config": str(_base_config(tmp_path)),
        "variants": [
            {"name": "baseline", "baseline": True, "overrides": {"genai-backend": None}},
            {"name": "canny", "overrides": {"genai-backend": "canny-controlnet"}},
        ],
    }
    body.update(overrides)
    return body


def _summary(
    *,
    metadata: int = 1000,
    actor_reference: int = 100,
    panorama: int = 400,
    residual: int = 5000,
    source: int = 100_000,
    psnr: float | None = 30.0,
    ssim: float | None = 0.9,
    vmaf: float | None = 60.0,
    num_frames: int = 10,
    run_root: str = "/tmp/run",
) -> dict[str, Any]:
    total = metadata + actor_reference + panorama + residual
    return {
        "run_output_root": run_root,
        "source_uri": "/data/real_tennis.mp4",
        "num_frames": num_frames,
        "evaluation": {
            "sizes_bytes": {
                "source": source,
                "metadata": metadata,
                "actor_reference": actor_reference,
                "panorama": panorama,
                "residual": residual,
                "transport_total": total,
                "transport_to_source_ratio": total / source,
            },
            "timings_sec": {"pipeline_total": 42.0},
            "psnr_mean": psnr,
            "ssim_mean": ssim,
            "vmaf_mean": vmaf,
        },
    }


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def test_load_matrix_spec_roundtrip(tmp_path: Path) -> None:
    spec = bench.load_matrix_spec(_write_spec(tmp_path, _spec_body(tmp_path, name="my-bench")))
    assert spec.name == "my-bench"
    assert spec.input_video == "assets/real_tennis.mp4"
    assert [v.name for v in spec.variants] == ["baseline", "canny"]
    assert spec.baseline.name == "baseline"


def test_load_matrix_spec_defaults_name_and_baseline(tmp_path: Path) -> None:
    body = _spec_body(tmp_path)
    body["variants"][0].pop("baseline")
    spec = bench.load_matrix_spec(_write_spec(tmp_path, body))
    assert spec.name == "matrix"  # spec filename stem
    assert spec.baseline.name == "baseline"  # first variant becomes baseline


def test_load_matrix_spec_normalizes_underscore_keys(tmp_path: Path) -> None:
    body = _spec_body(tmp_path)
    body["variants"][1]["overrides"] = {"genai_backend": "canny-controlnet"}
    spec = bench.load_matrix_spec(_write_spec(tmp_path, body))
    assert spec.variants[1].overrides == {"genai-backend": "canny-controlnet"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda body: body.pop("input"), "input"),
        (lambda body: body.update(variants=body["variants"][:1]), "at least two"),
        (lambda body: body["variants"].append({"name": "canny", "overrides": {}}), "unique"),
        (lambda body: body["variants"][1].update(baseline=True), "At most one"),
        (lambda body: body["variants"][1].update(name="  "), "non-empty"),
    ],
)
def test_load_matrix_spec_rejects_invalid(tmp_path: Path, mutate, message: str) -> None:
    body = _spec_body(tmp_path)
    mutate(body)
    with pytest.raises(ValueError, match=message):
        bench.load_matrix_spec(_write_spec(tmp_path, body))


# ---------------------------------------------------------------------------
# Config materialization & comparability
# ---------------------------------------------------------------------------


def test_materialize_config_merge_order_and_summary_forced(tmp_path: Path) -> None:
    base = _base_config(tmp_path, {"num-frames": 10, "codec-crf": 35, "summary-file": False})
    body = _spec_body(tmp_path)
    body["base-config"] = str(base)
    body["common-overrides"] = {"num-frames": None, "codec-crf": 40}
    body["variants"][1]["overrides"] = {"codec_crf": 45}
    spec = bench.load_matrix_spec(_write_spec(tmp_path, body))

    baseline_cfg = bench.materialize_config(spec, spec.variants[0])
    variant_cfg = bench.materialize_config(spec, spec.variants[1])

    assert baseline_cfg["num-frames"] is None  # common override beats base
    assert baseline_cfg["codec-crf"] == 40
    assert variant_cfg["codec-crf"] == 45  # variant override beats common, despite underscore spelling
    assert baseline_cfg["summary-file"] is True and variant_cfg["summary-file"] is True


def test_comparability_issues_flags_differing_keys() -> None:
    configs = {
        "baseline": {"codec-crf": 35, "seed": 1337},
        "variant": {"codec-crf": 45, "seed": 1337},
    }
    issues = bench.comparability_issues(configs)
    assert len(issues) == 1
    assert "codec-crf" in issues[0]
    assert not bench.comparability_issues({"a": {"codec-crf": 35}, "b": {"codec-crf": 35}})


# ---------------------------------------------------------------------------
# Stdout summary extraction
# ---------------------------------------------------------------------------


def test_extract_summary_from_noisy_stdout() -> None:
    summary = _summary(run_root="/out/x")
    stdout = "loading {weights}\n" + json.dumps({"other": 1}) + "\nnoise\n" + json.dumps(summary, indent=2) + "\ntrailer"
    extracted = bench.extract_summary_from_stdout(stdout)
    assert extracted is not None
    assert extracted["run_output_root"] == "/out/x"


def test_extract_summary_returns_none_when_absent() -> None:
    assert bench.extract_summary_from_stdout("no json here { broken") is None


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def _result(name: str, summary: dict[str, Any] | None, *, baseline: bool = False, status: str = "ok") -> bench.VariantResult:
    return bench.VariantResult(name=name, is_baseline=baseline, status=status, run_dir=f"/runs/{name}", summary=summary)


def test_build_report_verdicts() -> None:
    results = [
        _result("baseline", _summary(metadata=1000, residual=10_000), baseline=True),
        # +500 semantic, -2000 residual → net +1500 → pays
        _result("good", _summary(metadata=1500, residual=8_000)),
        # +3000 semantic, -2000 residual → net -1000 → does not pay
        _result("bad", _summary(metadata=4000, residual=8_000)),
    ]
    markdown, payload = bench.build_report(results, matrix_name="t", warnings=[])
    rows = {row["variant"]: row for row in payload["rows"]}

    assert rows["baseline"]["verdict"] == "(baseline)"
    assert rows["good"]["verdict"] == "PAYS"
    assert rows["good"]["net_bytes_vs_baseline"] == 1500
    assert rows["good"]["delta_semantic_bytes"] == 500
    assert rows["good"]["delta_residual_saved_bytes"] == 2000
    assert rows["bad"]["verdict"] == "DOES NOT PAY"
    assert "| PAYS |" in markdown and "DOES NOT PAY" in markdown


def test_build_report_flags_quality_drop_and_null_psnr() -> None:
    results = [
        _result("baseline", _summary(residual=10_000, vmaf=60.0), baseline=True),
        _result("cheap", _summary(residual=1_000, vmaf=40.0, psnr=None)),
    ]
    markdown, payload = bench.build_report(results, matrix_name="t", warnings=["like-for-like warning"])
    row = payload["rows"][1]
    assert row["verdict"].startswith("PAYS")
    assert "quality drop" in row["verdict"]
    assert "like-for-like warning" in markdown
    assert "PSNR is null" in markdown


def test_build_report_without_baseline_lists_failures() -> None:
    results = [_result("baseline", None, baseline=True, status="failed")]
    markdown, payload = bench.build_report(results, matrix_name="t", warnings=[])
    assert "No usable baseline" in markdown
    assert payload["rows"] == []


def test_build_report_lists_failed_variants_alongside_table() -> None:
    results = [
        _result("baseline", _summary(), baseline=True),
        bench.VariantResult(name="broken", is_baseline=False, status="failed", error="pipeline exited with code 1"),
    ]
    markdown, _payload = bench.build_report(results, matrix_name="t", warnings=[])
    assert "Failed / missing variants" in markdown
    assert "pipeline exited with code 1" in markdown


# ---------------------------------------------------------------------------
# run_matrix orchestration (runner injected — no real pipeline)
# ---------------------------------------------------------------------------


def _fake_runner_factory(outputs_root: Path, *, fail_names: set[str] | None = None):
    calls: list[str] = []

    def runner(config_path: Path, input_video: str, log_path: Path) -> tuple[int, str]:
        name = config_path.stem
        calls.append(name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log", encoding="utf-8")
        if fail_names and name in fail_names:
            return 1, "boom"
        run_dir = outputs_root / f"run_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = _summary(run_root=str(run_dir), residual=4000 if name != "baseline" else 5000)
        (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return 0, "prelude\n" + json.dumps(summary, indent=2)

    return runner, calls


def _runnable_spec(tmp_path: Path) -> bench.MatrixSpec:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake")
    body = _spec_body(tmp_path, input=str(input_video))
    return bench.load_matrix_spec(_write_spec(tmp_path, body))


def test_run_matrix_produces_manifest_symlinks_and_results(tmp_path: Path) -> None:
    spec = _runnable_spec(tmp_path)
    bench_dir = tmp_path / "bench"
    runner, calls = _fake_runner_factory(tmp_path / "outputs")

    results = bench.run_matrix(spec, bench_dir, runner=runner)

    assert calls == ["baseline", "canny"]
    assert [r.status for r in results] == ["ok", "ok"]
    assert (bench_dir / "runs" / "canny").is_symlink()
    assert (bench_dir / "configs" / "baseline.yaml").is_file()

    manifest = json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {v["name"]: v["status"] for v in manifest["variants"]} == {"baseline": "ok", "canny": "ok"}
    assert manifest["variants"][1]["overrides"] == {"genai-backend": "canny-controlnet"}

    loaded, warnings, name = bench.results_from_bench_dir(bench_dir)
    assert name == "matrix"
    assert all(r.summary is not None for r in loaded)
    assert warnings == []  # identical comparability keys


def test_run_matrix_resumes_completed_variants(tmp_path: Path) -> None:
    spec = _runnable_spec(tmp_path)
    bench_dir = tmp_path / "bench"
    runner, calls = _fake_runner_factory(tmp_path / "outputs")
    bench.run_matrix(spec, bench_dir, runner=runner)

    runner2, calls2 = _fake_runner_factory(tmp_path / "outputs")
    results = bench.run_matrix(spec, bench_dir, runner=runner2)
    assert calls2 == []  # nothing re-run
    assert all(r.status == "skipped-existing" for r in results)
    assert all(r.summary is not None for r in results)


def test_run_matrix_continues_past_failures(tmp_path: Path) -> None:
    spec = _runnable_spec(tmp_path)
    bench_dir = tmp_path / "bench"
    runner, calls = _fake_runner_factory(tmp_path / "outputs", fail_names={"baseline"})

    results = bench.run_matrix(spec, bench_dir, runner=runner)
    assert calls == ["baseline", "canny"]  # canny still ran
    assert results[0].status == "failed" and "code 1" in (results[0].error or "")
    assert results[1].status == "ok"


def test_run_matrix_missing_input_raises(tmp_path: Path) -> None:
    body = _spec_body(tmp_path, input=str(tmp_path / "missing.mp4"))
    spec = bench.load_matrix_spec(_write_spec(tmp_path, body))
    with pytest.raises(FileNotFoundError, match="Input video"):
        bench.run_matrix(spec, tmp_path / "bench", runner=lambda *a: (0, ""))


# ---------------------------------------------------------------------------
# Ad-hoc report over run dirs
# ---------------------------------------------------------------------------


def test_results_from_run_dirs_warns_on_mismatch(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    for run_dir, frames in ((dir_a, 10), (dir_b, 300)):
        run_dir.mkdir()
        (run_dir / "run_summary.json").write_text(json.dumps(_summary(num_frames=frames)), encoding="utf-8")

    results, warnings = bench.results_from_run_dirs([dir_a, dir_b])
    assert results[0].is_baseline and not results[1].is_baseline
    assert any("frame counts differ" in w for w in warnings)
    assert any("cannot be verified" in w for w in warnings)


def test_main_report_cli_on_bench_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = _runnable_spec(tmp_path)
    bench_dir = tmp_path / "bench"
    runner, _calls = _fake_runner_factory(tmp_path / "outputs")
    bench.run_matrix(spec, bench_dir, runner=runner)

    exit_code = bench.main(["report", str(bench_dir)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PAYS" in captured.out
    assert (bench_dir / "report.md").is_file()
    assert (bench_dir / "report.json").is_file()
