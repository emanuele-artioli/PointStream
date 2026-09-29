"""Mask feather and hold-out window."""

from __future__ import annotations

import numpy as np

from demo.experiments.neural_bg import feather_mask, holdout_windows


def test_feather_softens_the_edge_and_keeps_the_hole() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 1
    soft = feather_mask(mask, dilate_px=2, blur_px=5)
    assert soft[16, 16] == 1.0
    assert soft[0, 0] == 0.0
    assert 0.0 < soft[10, 16] < 1.0


def test_diffueraser_is_the_kept_fill_and_stills_are_archived() -> None:
    import demo.experiments.archive.neural_bg_still_fills as archived
    import demo.experiments.inpaint_background_smoke as smoke

    assert hasattr(smoke, "_fill_diffueraser")
    assert not hasattr(smoke, "_fill_sdxl")
    assert not hasattr(smoke, "_fill_flux")
    assert not hasattr(smoke, "_fill_qwen")
    assert callable(archived.fill_sdxl)
    assert callable(archived.fill_flux)
    assert callable(archived.fill_qwen)


def test_composite_keeps_unmasked_pixels() -> None:
    from demo.experiments.inpaint_background_smoke import _composite

    original = np.full((4, 4, 3), 10, dtype=np.uint8)
    filled = np.full((4, 4, 3), 200, dtype=np.uint8)
    soft = np.zeros((4, 4), dtype=np.float32)
    soft[1:3, 1:3] = 1.0
    mixed = _composite(original, filled, soft)
    assert mixed[0, 0].tolist() == [10, 10, 10]
    assert mixed[2, 2].tolist() == [200, 200, 200]
    windows = holdout_windows(100.0, holdout_s=10.0, gap_s=2.0)
    start, length = windows["holdout"]
    assert length == 10.0
    assert abs(start - 45.0) < 1e-6
    assert windows["train_left"][1] == start - 2.0
    assert windows["train_right"][0] == start + 12.0
