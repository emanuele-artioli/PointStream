"""Tests for the frozen Gate B codec procedure and metric bounds."""

from __future__ import annotations

import sqlite3  # noqa: F401


from src.contracts.frozen_procedure import (
    FROZEN_ANCHOR_PRESETS,
    FROZEN_ANCHOR_QPS,
    FROZEN_RUNGS,
    check_adjacent_rungs,
    configure_frozen_rung,
    get_frozen_bounds,
    validate_ledger,
)
from src.runner.config_io import load_tier


def test_frozen_rungs_structure() -> None:
    assert len(FROZEN_RUNGS) == 4
    names = [r.name for r in FROZEN_RUNGS]
    assert names == ["C0", "C1", "C2", "C3"]

    for r in FROZEN_RUNGS:
        assert r.stream_codec == "vvc"
        assert r.appearance_format == "webp"
        assert r.bg_crf > 0
        assert r.appearance_jpeg > 0
        assert r.motion_max_points > 0

    # Monotonicity of rungs
    crfs = [r.bg_crf for r in FROZEN_RUNGS]
    assert crfs == sorted(crfs, reverse=True), "CRF must decrease (quality increase) from C0 to C3"

    jpegs = [r.appearance_jpeg for r in FROZEN_RUNGS]
    assert jpegs == sorted(jpegs), "Appearance quality must increase from C0 to C3"

    points = [r.motion_max_points for r in FROZEN_RUNGS]
    assert points == sorted(points), "Motion points must increase from C0 to C3"


def test_configure_frozen_rung() -> None:
    base = load_tier("balanced")
    c0_cfg = configure_frozen_rung(base, FROZEN_RUNGS[0], context_id="test_ctx")

    # Invariants
    assert c0_cfg.lattice.generation is False
    assert c0_cfg.lattice.residual is False
    assert c0_cfg.lattice.pose is False

    assert c0_cfg.background.method == "panorama-stream"
    assert c0_cfg.background.stream_codec == "vvc"
    assert c0_cfg.background.stream_crf == 63
    assert c0_cfg.background.transport_scale == 1.0

    assert c0_cfg.appearance.representation == "compressed-image"
    assert c0_cfg.appearance.format == "webp"
    assert c0_cfg.appearance.jpeg_quality == 30
    assert c0_cfg.appearance.downscale == 2

    assert c0_cfg.motion.max_points == 8


def test_get_frozen_bounds() -> None:
    bounds = get_frozen_bounds(n_frames=96, height=1080, width=1920, n_scenes=2)
    assert bounds["n_frames_per_scene"] == 96
    assert bounds["decoded_shape_required"] == [192, 1080, 1920, 3]
    assert bounds["quality"]["vmaf"] == [0.0, 98.0]
    assert bounds["quality"]["psnr_y_db"] == [8.0, 55.0]
    assert bounds["quality"]["ssim"] == [0.0, 1.0]
    assert bounds["late_frame_last_minus_first"]["vmaf"] == [-25.0, 8.0]
    assert bounds["late_frame_last_minus_first"]["psnr_y_db"] == [-8.0, 3.0]


def test_validate_ledger() -> None:
    parts = {"background": 50000, "appearance": 10000, "motion": 1000, "metadata": 500}
    assert validate_ledger(parts, 61500) == []
    alarms = validate_ledger(parts, 60000)
    assert len(alarms) == 1
    assert "Ledger does not balance" in alarms[0]


def test_check_adjacent_rungs() -> None:
    prev = {"name": "C0", "bytes": 50000, "parts": {"panorama": 30000}}
    curr_ok = {"name": "C1", "bytes": 65000, "parts": {"panorama": 35000}}
    assert check_adjacent_rungs(prev, curr_ok) == []

    # Drop > 5%
    curr_drop = {"name": "C1", "bytes": 45000, "parts": {"panorama": 35000}}
    alarms = check_adjacent_rungs(prev, curr_drop)
    assert len(alarms) == 1
    assert "fell by >5%" in alarms[0]

    # Background bytes inverted
    curr_bg_invert = {"name": "C1", "bytes": 65000, "parts": {"panorama": 25000}}
    alarms = check_adjacent_rungs(prev, curr_bg_invert)
    assert len(alarms) == 1
    assert "background bytes" in alarms[0]


def test_frozen_anchors() -> None:
    assert FROZEN_ANCHOR_QPS == (63, 55, 47, 39)
    assert FROZEN_ANCHOR_PRESETS["av1"] == "8"
    assert FROZEN_ANCHOR_PRESETS["vvc"] == "medium"
