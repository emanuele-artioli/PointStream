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
    assert bg_crfs == [63, 55, 48, 42]

    # Appearance quality is nondecreasing
    app_jpegs = [r.appearance_jpeg for r in RUNGS]
    assert app_jpegs == [30, 45, 60, 75]

    # Appearance downscale is nonincreasing
    downscales = [r.appearance_downscale for r in RUNGS]
    assert downscales == [2, 1, 1, 1]

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
        assert cfg.background.stream_codec == rung.stream_codec
        assert cfg.background.stream_crf == rung.bg_crf
        assert cfg.background.transport_scale == 1.0

        # Foreground & motion
        assert cfg.appearance.representation == "compressed-image"
        assert cfg.appearance.jpeg_quality == rung.appearance_jpeg
        assert cfg.appearance.downscale == rung.appearance_downscale
        assert cfg.appearance.format == rung.appearance_format
        assert cfg.motion.max_points == rung.motion_max_points


def test_bp56_seed_configuration_is_driven_by_current_code() -> None:
    """C1 is constructed by the current driver, not a hard-coded result row."""
    base = load_tier("balanced")
    c1 = configure_rung(base, RUNGS[1])
    assert c1.background.stream_codec == "vvc"
    assert c1.background.stream_crf == 55
    assert c1.appearance.format == "webp"
    assert c1.appearance.jpeg_quality == 45
    assert c1.appearance.downscale == 1
    assert c1.motion.max_points == 16
    assert c1.lattice.residual is False
    assert c1.lattice.generation is False


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
    """Raw-byte alarm ceiling follows the exact number of source frames."""
    bounds_48 = get_pre_registered_bounds(48)
    bounds_96 = get_pre_registered_bounds(96)
    assert bounds_96["coded_bytes"]["high_inclusive"] - 1048576 == 2 * (
        bounds_48["coded_bytes"]["high_inclusive"] - 1048576
    )
    assert bounds_96["quality"] == bounds_48["quality"]


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


def test_codec_floor_probe_executes_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace
    import experiments.tier.gate_a_tools as tools

    binary = tmp_path / "encoder"
    binary.write_bytes(b"real tool")
    called = []

    monkeypatch.setattr(
        tools,
        "resolve_tool_specs",
        lambda: {"av1": {"available": True, "slowest_preset": "0"}},
    )

    def fake_roundtrip(frames, *, request, fps):
        called.append((request.codec_name, request.preset, request.rate, fps))
        return SimpleNamespace(
            size_bytes=17,
            frames=frames.copy(),
            tool_path=str(binary),
            tool_version="test-version",
            preset=request.preset,
            qp=request.rate,
        )

    monkeypatch.setattr(tools, "timed_roundtrip", fake_roundtrip)
    result = tools.probe_codec_floor("av1")
    assert called == [("av1", "0", 63, 24.0)]
    assert result["verified"] is True
    assert result["probe_bytes"] == 17
    assert result["binary_sha256"]


def test_frozen_bounds_match_gate_a_contract() -> None:
    bounds = get_pre_registered_bounds(48)
    assert bounds["bd_rate_vmaf_percent"] == [-90.0, 300.0]
    assert bounds["late_frame_last_minus_first"]["vmaf"] == [-25.0, 8.0]
    assert bounds["timing"]["ranked_encode_decode_must_be_non_null"] is True
    assert bounds["curve"]["minimum_usable_points"] == 4
    assert bounds["nonresumable_operation_timeout_seconds"] == 3300.0


def test_dry_controls_cannot_claim_native_controls_valid(tmp_path, monkeypatch) -> None:
    import experiments.tier.gate_a_controls as controls

    monkeypatch.setattr(
        controls, "verify_metric_anchors", lambda reference, destination: {"valid": True}
    )
    monkeypatch.setattr(controls, "run_temporal_null", lambda reference: {"scores": {}})
    result = controls.run_gate_a_controls(np.zeros((2, 2, 2, 3), dtype=np.uint8), tmp_path)
    assert result["valid"] is False
    assert result["pending"] == ["object_stream_off", "conventional_fallback"]


def test_budget_checkpoint_resume_does_not_rerun_operation(tmp_path) -> None:
    from experiments.tier.gate_a_long_context import _checkpointed

    points = tmp_path / "points"
    budget = tmp_path / "budget.json"
    calls = []

    def operation():
        calls.append("run")
        return {"usable": True}

    first = _checkpointed(points, budget, "C0", "pointstream", operation)
    second = _checkpointed(points, budget, "C0", "pointstream", operation)
    assert first["usable"] is True
    assert second["usable"] is True
    assert calls == ["run"]


def test_native_mode_requires_explicit_authorization(tmp_path) -> None:
    from experiments.tier.gate_a_long_context import main

    with pytest.raises(SystemExit, match="requires --authorize-native"):
        main(["--native", "--frames", "48", "--out-dir", str(tmp_path)])
