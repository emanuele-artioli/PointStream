"""Decision rules for the warp-residual probe. No encodes."""

from __future__ import annotations

import math

import numpy as np

from experiments.modular.warp_residual_probe import (
    admit_keyframe,
    dominates,
    foreground_error_split,
)


def test_pareto_win_requires_both_axes() -> None:
    assert dominates(100, 30.0, 200, 29.0)
    assert dominates(100, 30.0, 100, 29.0)
    assert dominates(100, 30.0, 200, 30.0)
    assert not dominates(200, 31.0, 100, 24.0)
    assert not dominates(100, 20.0, 200, 30.0)
    assert not dominates(100, 30.0, 100, 30.0)
    assert not dominates(100, None, 200, 30.0)


def test_keyframe_budget_blocks_a_crop_the_threshold_would_allow() -> None:
    assert admit_keyframe(spent=2_000, next_cost=2_000, budget=12_000, mse=51.0, threshold=50.0)
    assert not admit_keyframe(spent=2_000, next_cost=2_000, budget=12_000, mse=50.0, threshold=50.0)
    assert not admit_keyframe(spent=11_000, next_cost=2_000, budget=12_000, mse=1_000.0, threshold=50.0)


def test_foreground_split_scores_only_the_uncovered_error() -> None:
    reference = np.full((2, 2, 3), 10, dtype=np.uint8)
    reconstruction = reference.copy()
    reconstruction[:, 1] = 0
    mask = np.ones((2, 2), dtype=bool)
    covered = np.zeros((2, 2), dtype=bool)
    covered[:, 0] = True
    split = foreground_error_split(reference, reconstruction, mask, covered)
    assert split["covered_fraction"] == 0.5
    assert split["covered_psnr"] is None
    expected = 10.0 * math.log10((255.0 ** 2) / 100.0)
    assert split["uncovered_psnr"] == expected
