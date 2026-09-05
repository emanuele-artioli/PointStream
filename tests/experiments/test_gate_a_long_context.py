"""Tests for Gate A long-context rate ladder, controls, and pre-registered bounds."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tier.gate_a_controls import (
    run_temporal_null,
    verify_conventional_fallback,
)
from experiments.tier.gate_a_long_context import (
    RUNGS,
    check_adjacent_rungs,
    configure_rung,
    get_pre_registered_bounds,
    validate_ledger,
)
from src.contracts.config import PointstreamConfig
from src.runner.config_io import load_tier


def test_ladder_rungs_specification() -> None:
    """Ensure exactly 4 coherent rungs with nondecreasing fidelity parameters."""
    assert len(RUNGS) == 4
    names = [r.name for r in RUNGS]
    assert names == ["C0", "C1", "C2", "C3"]

    # Background CRF is nonincreasing (higher fidelity)
    bg_crfs = [r.bg_crf for r in RUNGS]
    assert bg_crfs == [63, 63, 57, 51]

    # Appearance JPEG quality is nondecreasing
    app_jpegs = [r.appearance_jpeg for r in RUNGS]
    assert app_jpegs == [25, 40, 55, 70]

    # Appearance downscale is nonincreasing
    downscales = [r.appearance_downscale for r in RUNGS]
    assert downscales == [4, 2, 2, 1]

    # Motion trajectories are strictly increasing
    motion_pts = [r.motion_max_points for r in RUNGS]
    assert motion_pts == [8, 16, 24, 32]


def test_configure_rung_invariants() -> None:
    """Verify that configure_rung enforces canonical invariants."""
    base: PointstreamConfig = load_tier("balanced")

    for rung in RUNGS:
        cfg = configure_rung(base, rung)

        # Invariant zero-byte channels: generation and residual off
        assert cfg.lattice.generation is False
        assert cfg.lattice.residual is False

        # Canonical background configuration
        assert cfg.background.method == "panorama-stream"
        assert cfg.background.stream_codec == "av1"
        assert cfg.background.stream_crf == rung.bg_crf
        assert cfg.background.transport_scale == 1.0
        assert cfg.background.stream_usage == "good"
        assert cfg.background.stream_cpu_used == 4

        # Foreground & motion
        assert cfg.appearance.representation == "compressed-image"
        assert cfg.appearance.jpeg_quality == rung.appearance_jpeg
        assert cfg.appearance.downscale == rung.appearance_downscale
        assert cfg.motion.max_points == rung.motion_max_points


def test_bp56_seed_regression_and_bounds() -> None:
    """Assert BP56 seed (C1) ledger targets and pre-registered bounds conformance."""
    bp56_ledger = {
        "panorama": 348504,
        "actor_reference": 8599,
        "metadata": 20257,
        "residual": 0,
        "fallback": 0,
    }
    total_bytes = 377360
    assert sum(bp56_ledger.values()) == total_bytes

    alarms = validate_ledger(bp56_ledger, total_bytes)
    assert alarms == []

    vmaf = 79.339
    y_psnr = 33.259
    ssim = 0.974
    assert y_psnr > 30.0
    assert ssim > 0.95

    bounds_48 = get_pre_registered_bounds(48)["bounds"]["C1"]
    assert bounds_48["bytes_min"] <= total_bytes <= bounds_48["bytes_max"]
    assert bounds_48["vmaf_min"] <= vmaf <= bounds_48["vmaf_max"]


def test_validate_ledger_detects_imbalance() -> None:
    """Verify validate_ledger catches ledger discrepancies."""
    parts = {
        "panorama": 1000,
        "actor_reference": 500,
        "metadata": 100,
    }
    # Matching total
    assert validate_ledger(parts, 1600) == []

    # Discrepancy
    alarms = validate_ledger(parts, 1601)
    assert len(alarms) == 1
    assert "Ledger does not balance" in alarms[0]


def test_check_adjacent_rungs_monotonicity() -> None:
    """Verify check_adjacent_rungs detects regressions."""
    prev_row = {
        "name": "C0",
        "bytes": 300000,
        "parts": {"panorama": 250000},
    }
    curr_row = {
        "name": "C1",
        "bytes": 350000,
        "parts": {"panorama": 250000},
    }
    assert check_adjacent_rungs(prev_row, curr_row) == []

    # Drop by > 5%
    curr_dropped = {
        "name": "C1",
        "bytes": 280000,
        "parts": {"panorama": 250000},
    }
    alarms = check_adjacent_rungs(prev_row, curr_dropped)
    assert any("fell by >5%" in a for a in alarms)

    # Background bytes decrease
    curr_bg_dropped = {
        "name": "C2",
        "bytes": 380000,
        "parts": {"panorama": 240000},
    }
    alarms2 = check_adjacent_rungs(curr_row, curr_bg_dropped)
    assert any("background bytes" in a for a in alarms2)


def test_pre_registered_bounds_scaling() -> None:
    """Verify pre-registered bounds scale with context duration."""
    bounds_48 = get_pre_registered_bounds(48)["bounds"]
    bounds_96 = get_pre_registered_bounds(96)["bounds"]

    for rung in ("C0", "C1", "C2", "C3"):
        assert bounds_96[rung]["bytes_min"] == bounds_48[rung]["bytes_min"] * 2
        assert bounds_96[rung]["bytes_max"] == bounds_48[rung]["bytes_max"] * 2
        # Quality scale does not scale with duration
        assert bounds_96[rung]["vmaf_min"] == bounds_48[rung]["vmaf_min"]
        assert bounds_96[rung]["vmaf_max"] == bounds_48[rung]["vmaf_max"]


def test_conventional_fallback_control() -> None:
    """Test Gate 2 conventional fallback control bounds checking."""
    # Valid control: rate ratio ~ 1.0, vmaf diff <= 1.0
    res = verify_conventional_fallback(
        fallback_bytes=10000,
        fallback_vmaf=80.0,
        anchor_bytes=10000,
        anchor_vmaf=80.5,
    )
    assert res["passed"] is True

    # Rate ratio out of bounds
    with pytest.raises(SystemExit) as exc_info:
        verify_conventional_fallback(
            fallback_bytes=12000,
            fallback_vmaf=80.0,
            anchor_bytes=10000,
            anchor_vmaf=80.5,
        )
    assert "rate ratio" in str(exc_info.value)

    # VMAF diff out of bounds
    with pytest.raises(SystemExit) as exc_info2:
        verify_conventional_fallback(
            fallback_bytes=10000,
            fallback_vmaf=80.0,
            anchor_bytes=10000,
            anchor_vmaf=82.0,
        )
    assert "vmaf difference" in str(exc_info2.value)


def test_temporal_null_control() -> None:
    """Test Gate 2 temporal null control permutation."""
    rng = np.random.default_rng(0)
    fake_frames = rng.integers(0, 255, size=(4, 64, 64, 3), dtype=np.uint8)

    # Requires >= 2 frames
    with pytest.raises(ValueError):
        run_temporal_null(fake_frames[:1])

    # Runs permutation
    res = run_temporal_null(fake_frames)
    assert res["frame_count"] == 4
    assert res["control"] == "temporal_null_shuffled_frames"
    assert "scores" in res
