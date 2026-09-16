"""E06 probe-card controls: paste vs warped-reference charging, no encodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.tier.e06_probe import (
    LEDGER_PARTS,
    charge_controls,
    main,
    paste_placements,
    refuse_overwrite,
    required_ledger_keys,
    warped_reference_placements,
)


def _toy_clip() -> tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros((4, 32, 32), dtype=bool)
    for index in range(4):
        frames[index] = index * 40
        y0, x0 = 4 + index, 4 + index
        masks[index, y0 : y0 + 8, x0 : x0 + 8] = True
        frames[index, y0 : y0 + 8, x0 : x0 + 8] = (20 + index * 30, 80, 200)
    return frames, masks


def test_warped_reference_reuses_one_crop() -> None:
    frames, masks = _toy_clip()
    paste = paste_placements(frames, masks)
    warped = warped_reference_placements(frames, masks)
    assert len(paste) == 4
    assert len(warped) == 4
    assert not np.array_equal(paste[0].crop, paste[-1].crop)
    for item in warped:
        assert np.array_equal(item.crop, warped[0].crop)
        assert item.bbox != warped[0].bbox or item.frame_index == 0


def test_charge_residual_off_zeros_r_and_warp_sends_one_appearance() -> None:
    frames, masks = _toy_clip()
    paste = charge_controls(frames, masks, residual_on=False, predictor="paste")
    warp = charge_controls(frames, masks, residual_on=False, predictor="warped_reference")
    required_ledger_keys(paste["parts"])
    assert paste["parts"]["residual"] == 0
    assert warp["parts"]["residual"] == 0
    assert paste["n_appearance_payloads"] == 4
    assert warp["n_appearance_payloads"] == 1
    assert warp["parts"]["actor_reference"] < paste["parts"]["actor_reference"]
    assert warp["metadata_subledger"]["pose_present"] is False
    assert set(LEDGER_PARTS) <= set(paste["parts"])


def test_residual_on_leaves_r_unmeasured_until_encode() -> None:
    frames, masks = _toy_clip()
    charged = charge_controls(frames, masks, residual_on=True, predictor="paste")
    assert charged["parts"]["residual"] is None
    assert charged["generation"] == "off"


def test_refuse_overwrite_on_probe_report(tmp_path: Path) -> None:
    (tmp_path / "probe_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refuse_overwrite(tmp_path)


def test_launch_flag_is_refused() -> None:
    with pytest.raises(SystemExit, match="not authorized"):
        main(["--launch"])
