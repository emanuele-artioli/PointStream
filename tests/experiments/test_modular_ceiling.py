"""Tests for modular oracle ceiling and segmentation impact experiments."""

from __future__ import annotations

from pathlib import Path

from experiments.modular.eval_segmentation_impact import run_segmentation_impact_eval
from experiments.modular.oracle_ceiling import DEFAULT_MANIFEST, run_ceiling_analysis


def test_oracle_ceiling_analysis_runs_cleanly() -> None:
    report = run_ceiling_analysis(manifest_path=DEFAULT_MANIFEST, dry_run=True)
    assert report["schema"] == "pointstream.modular_oracle_ceiling_report.v1"
    assert "conclusions" in report
    assert "short_horizon_48f" in report["conclusions"]
    assert "long_horizon_192f" in report["conclusions"]

    # Short horizon is blocked
    assert report["conclusions"]["short_horizon_48f"]["verdict"] == "CEILING_BLOCKED"
    # Long horizon is cleared by oracle
    assert report["conclusions"]["long_horizon_192f"]["verdict"] == "CEILING_CLEARED_BY_ORACLE"

    # Check 2 horizons returned
    horizons = report["horizons"]
    assert len(horizons) == 2
    short_h = next(h for h in horizons if h["horizon_id"] == "short_48f")
    long_h = next(h for h in horizons if h["horizon_id"] == "long_192f")

    # Short horizon: Current and Oracle both exceed VVC anchor bytes (21,288 B)
    assert short_h["total_bytes_current"] > short_h["anchor_vvc_bytes"]

    # Long horizon: Oracle clears VVC anchor bytes (77,228 B)
    assert long_h["total_bytes_oracle"] < long_h["anchor_vvc_bytes"]
    assert long_h["anchor_cleared_by_oracle"] is True


def test_segmentation_impact_eval_runs_cleanly(tmp_path: Path) -> None:
    report = run_segmentation_impact_eval(scene="test_scene", n_frames=8, dry_run=True)
    assert report["schema"] == "pointstream.segmentation_downstream_impact.v1"
    assert "arms" in report
    assert "null_bbox" in report["arms"]
    assert "current" in report["arms"]
    assert "oracle_sam3" in report["arms"]

    # Check that oracle has smaller or equal mask area than null bbox
    null_area = report["arms"]["null_bbox"]["mean_mask_area_px"]
    oracle_area = report["arms"]["oracle_sam3"]["mean_mask_area_px"]
    assert oracle_area < null_area

    # Check headroom fields
    assert "crop_bytes_saved" in report["headroom"]
    assert "ghost_mad_reduction" in report["headroom"]
    assert report["verdict"] in {"ACTIVE_SEARCH", "SATISFIED_FREEZE"}


def test_oracle_ceiling_generate_visuals(tmp_path: Path) -> None:
    visuals_dir = tmp_path / "visuals"
    report = run_ceiling_analysis(
        manifest_path=DEFAULT_MANIFEST,
        dry_run=True,
        generate_visuals=True,
        visuals_dir=visuals_dir,
    )
    assert visuals_dir.exists()
    carousel_md = visuals_dir / "carousel.md"
    assert carousel_md.exists()
    carousel_content = carousel_md.read_text(encoding="utf-8")
    assert "````carousel" in carousel_content
    assert "<!-- slide -->" in carousel_content

    horizons = report["horizons"]
    assert len(horizons) == 2
    for h in horizons:
        scene_slug = h["scene"].replace("/", "_")
        assert "arms" in h
        for arm_name in ("null", "current", "oracle"):
            assert arm_name in h["arms"]
            arm_data = h["arms"][arm_name]
            assert "pose_oks" in arm_data
            assert 0.0 <= arm_data["pose_oks"] <= 1.0

            strip_file = visuals_dir / f"{scene_slug}_{arm_name}_strip.png"
            assert strip_file.exists()
            assert strip_file.stat().st_size > 0
            assert str(strip_file.resolve()) in carousel_content

    # Check anatomical fidelity progression across arms: oracle >= current > null
    for h in horizons:
        arms = h["arms"]
        assert arms["oracle"]["pose_oks"] >= arms["current"]["pose_oks"]
        assert arms["current"]["pose_oks"] > arms["null"]["pose_oks"]
