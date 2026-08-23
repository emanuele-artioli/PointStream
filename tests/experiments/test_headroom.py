"""Headroom measurement: player cost and panorama cost, with bounds written first."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.components.metrics.bd_rate import RDCurve
from experiments.headroom.measure import (
    FG_MODEST,
    FG_STRONG,
    bg_headroom,
    declared_bounds,
    fg_headroom,
    fg_verdict,
)
from experiments.headroom.remove import flat_fill, plate_fill, player_fraction
from experiments.headroom.synthetic import handheld_clip, tennis_clip


def test_empty_mask_plate_fill_keeps_the_pixels() -> None:
    frames, _masks = tennis_clip(n_frames=4, seed=2)
    empty = np.zeros(frames.shape[:3], dtype=bool)
    filled, _plate, _h = plate_fill(frames, empty)
    assert filled.shape == frames.shape
    assert np.mean(np.abs(filled.astype(int) - frames.astype(int))) < 8


def test_flat_fill_erases_the_player_region() -> None:
    frames, masks = tennis_clip(n_frames=4)
    filled = flat_fill(frames, masks, value=128)
    assert np.all(filled[masks] == 128)
    assert not np.all(filled[~masks] == 128)


def test_plate_fill_removes_player_stripes() -> None:
    frames, masks = tennis_clip(n_frames=8, seed=0)
    filled, _plate, _h = plate_fill(frames, masks)
    original_std = float(frames[masks].std())
    filled_std = float(filled[masks].std())
    assert original_std > filled_std
    assert player_fraction(masks) > 0.04
    assert player_fraction(masks) < 0.25


def test_bounds_are_the_prewritten_bars() -> None:
    bounds = declared_bounds()
    assert bounds["written_before_measurement"] is True
    assert bounds["fg_strong_saving"] == FG_STRONG == 0.25
    assert bounds["fg_modest_saving"] == FG_MODEST == 0.10
    assert fg_verdict(0.30) == "strong"
    assert fg_verdict(0.15) == "modest"
    assert fg_verdict(0.05) == "weak"


def _stub_encode(frames, *, work_dir, qps, masks=None, label=""):
    """Rate follows spatial gradient energy — what a codec spends bits on."""
    del work_dir, masks
    luma = frames.astype(np.float64).mean(axis=-1)
    grad_y, grad_x = np.gradient(luma, axis=(1, 2))
    energy = float(np.abs(grad_x).mean() + np.abs(grad_y).mean())
    rates = tuple(max(10.0, 5000.0 * energy / qp) for qp in qps)
    qualities = tuple(45.0 - 0.15 * qp for qp in qps)
    return {"curve": RDCurve(rates=rates, qualities=qualities, label=label), "qps": qps}


def test_fg_plate_saving_is_below_flat_saving(tmp_path: Path) -> None:
    """A flat hole overstates the prize; plate inpaint is the honest arm."""
    frames, masks = tennis_clip(n_frames=8, seed=4)
    result = fg_headroom(
        frames, masks, work_dir=tmp_path, encode_curve=_stub_encode, qps=(32, 40)
    )
    plate = result["plate_vs_original"]["saving"]
    flat = result["flat_vs_original"]["saving"]
    assert plate is not None and flat is not None
    assert flat >= plate
    # A gradient stub can give a slightly negative plate saving when warp
    # seams add edges; the real encoder run is the measurement. The stub
    # only has to keep flat from understating relative to plate.
    assert plate <= 1.0
    assert plate > -0.05


def test_fg_null_empty_mask_saves_almost_nothing(tmp_path: Path) -> None:
    frames, _ = tennis_clip(n_frames=6, seed=5)
    empty = np.zeros(frames.shape[:3], dtype=bool)
    result = fg_headroom(
        frames, empty, work_dir=tmp_path, encode_curve=_stub_encode, qps=(32, 40)
    )
    saving = result["plate_vs_original"]["saving"]
    assert saving is not None
    assert abs(saving) < 0.05


def test_bg_panorama_is_orders_cheaper_on_a_static_court() -> None:
    frames, masks = tennis_clip(n_frames=16, seed=0)
    result = bg_headroom(frames, masks, panorama_valid=True)
    assert result["panorama_valid"] is True
    assert result["conventional_over_panorama"] is not None
    assert result["conventional_over_panorama"] >= 10.0
    assert result["orders_of_magnitude"] is True
    assert result["plate_bytes"] > 0
    assert result["homography_bytes"] > 0
    assert result["jpeg_quality_moves_size"] is True


def test_bg_panorama_is_marked_invalid_on_handheld() -> None:
    frames, masks = handheld_clip()
    result = bg_headroom(frames, masks, panorama_valid=False)
    assert result["panorama_valid"] is False
    assert result["orders_of_magnitude"] is False
    assert result["note"]
