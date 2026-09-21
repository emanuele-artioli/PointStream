"""Tests for PointStream Neural Generative Benchmark Runner."""

from pathlib import Path
import tempfile

from experiments.modular.neural_benchmark import (
    evaluate_candidate,
    run_neural_benchmark,
)


def test_benchmark_dry_run_executes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "output"
        report = run_neural_benchmark(
            output_dir=out_dir,
            dry_run=True,
        )

        assert "candidates" in report
        assert len(report["candidates"]) >= 3
        assert (out_dir / "benchmark_report.json").exists()
        assert report["overall_status"] == "ALL_CANDIDATES_FALSIFIED"


def test_falsification_ceiling_enforced():
    ceilings = {
        "wire_budget_bytes": 12000,
        "anatomical_oks_min": 0.90,
        "fg_psnr_min": 35.8,
    }

    # Case 1: Wire violation
    c1 = {
        "model_id": "heavy_model",
        "conditioning_wire_bytes": 15000,
        "pose_oks": 0.95,
        "fg_psnr": 38.0,
    }
    ev1 = evaluate_candidate(c1, ceilings)
    assert ev1.verdict == "FALSIFIED"
    assert not ev1.wire_passed
    assert ev1.oks_passed
    assert any("Wire budget exceeded" in r for r in ev1.falsification_reasons)

    # Case 2: OKS violation
    c2 = {
        "model_id": "hallucinating_model",
        "conditioning_wire_bytes": 8000,
        "pose_oks": 0.82,
        "fg_psnr": 38.0,
    }
    ev2 = evaluate_candidate(c2, ceilings)
    assert ev2.verdict == "FALSIFIED"
    assert ev2.wire_passed
    assert not ev2.oks_passed
    assert any("Anatomical drift" in r for r in ev2.falsification_reasons)

    # Case 3: Ideal hypothetical model that clears all
    c3 = {
        "model_id": "perfect_model",
        "conditioning_wire_bytes": 9000,
        "pose_oks": 0.94,
        "fg_psnr": 37.0,
    }
    ev3 = evaluate_candidate(c3, ceilings)
    assert ev3.verdict == "PASSED"
    assert ev3.wire_passed
    assert ev3.oks_passed
    assert ev3.quality_passed
    assert len(ev3.falsification_reasons) == 0


def test_baseline_clears_ceiling():
    spec_path = Path(__file__).resolve().parents[2] / "manifests" / "neural_generative_benchmark_spec.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = run_neural_benchmark(
            spec_path=spec_path,
            output_dir=Path(tmp_dir),
            dry_run=True,
        )

        baseline = report["operational_baseline"]
        # Baseline must clear the ceilings
        assert baseline["short_horizon_wire_bytes"] <= report["falsifiable_ceilings"]["wire_budget_bytes"]
        assert baseline["long_horizon_wire_bytes"] <= report["falsifiable_ceilings"]["wire_budget_bytes"]
        assert baseline["pose_oks"] >= report["falsifiable_ceilings"]["anatomical_oks_min"]
        assert baseline["fg_psnr"] >= report["falsifiable_ceilings"]["fg_psnr_min"]
