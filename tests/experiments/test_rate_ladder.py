"""Tests for PointStream Modular Rate Ladder Runner."""

from pathlib import Path
import tempfile
import pytest

from experiments.modular.rate_ladder import run_rate_ladder


def test_rate_ladder_dry_run_executes_successfully():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "output"
        visuals_dir = Path(tmp_dir) / "visuals"

        report = run_rate_ladder(
            output_dir=out_dir,
            visuals_dir=visuals_dir,
            dry_run=True,
            generate_visuals=True,
            enforce_gpu=False,
        )

        assert "horizons" in report
        assert len(report["horizons"]) == 2

        # Check output file written
        assert (out_dir / "results.json").exists()
        # Check carousel generated
        assert (visuals_dir / "carousel.md").exists()
        assert (visuals_dir / "comparison_short.png").exists()
        assert (visuals_dir / "comparison_long.png").exists()


def test_rungs_ledger_and_byte_accounting():
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = run_rate_ladder(
            output_dir=Path(tmp_dir),
            dry_run=True,
            enforce_gpu=False,
        )

        for horizon in report["horizons"]:
            for rung in horizon["rungs"]:
                # Byte ledger invariant: T = B + F + M + R + H
                total_computed = (
                    rung["bytes_background"]
                    + rung["bytes_appearance"]
                    + rung["bytes_metadata"]
                    + rung["bytes_residual"]
                    + rung["bytes_container"]
                )
                assert rung["total_bytes"] == total_computed

                # Saliency weighting invariant: 0.7 * FG + 0.3 * BG
                expected_weighted = round(0.70 * rung["psnr_fg"] + 0.30 * rung["psnr_bg"], 2)
                assert abs(rung["psnr_weighted"] - expected_weighted) < 0.05


def test_pointstream_beats_vvc_on_long_horizon():
    with tempfile.TemporaryDirectory() as tmp_dir:
        report = run_rate_ladder(
            output_dir=Path(tmp_dir),
            dry_run=True,
            enforce_gpu=False,
        )

        long_h = next(h for h in report["horizons"] if h["id"] == "long")
        vvc_bytes = long_h["anchor_vvc_bytes"]

        # All four rungs must beat VVC rate on the 192f long horizon
        for rung in long_h["rungs"]:
            assert rung["beats_vvc_rate"], f"Rung {rung['rung_id']} failed to beat VVC"
            assert rung["total_bytes"] < vvc_bytes

        # C1 sweet spot check: >60% savings
        c1 = next(r for r in long_h["rungs"] if r["rung_id"] == "C1_adaptive_keyframes")
        saving_pct = (1.0 - c1["total_bytes"] / vvc_bytes) * 100.0
        assert saving_pct >= 60.0
        assert c1["pose_oks"] >= 0.90


def test_missing_manifest_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        run_rate_ladder(
            manifest_path=Path("/nonexistent/manifest.json"),
            dry_run=True,
            enforce_gpu=False,
        )
