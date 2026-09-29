"""Oracle composite, rate sum, and the rule that a worse LPIPS stays reported."""

from __future__ import annotations

import numpy as np

from demo.experiments.compare_mask_oracles import (
    build_report,
    composite_oracle,
    flat_fill,
    select_background,
    total_kbps,
)


def test_composite_keeps_reference_on_support() -> None:
    reference = np.zeros((4, 4, 3), dtype=np.uint8)
    reference[:, :] = (10, 20, 30)
    background = np.zeros((4, 4, 3), dtype=np.uint8)
    background[:, :] = (200, 200, 200)
    support = np.zeros((4, 4), dtype=np.uint8)
    support[0, 0] = 1
    out = composite_oracle(reference, background, support)
    assert tuple(out[0, 0]) == (10, 20, 30)
    assert tuple(out[1, 1]) == (200, 200, 200)


def test_hole_fill_kept_only_when_smaller() -> None:
    assert select_background(100, 80) == ("hole", 80)
    assert select_background(100, 100) == ("clean", 100)
    assert select_background(100, 140) == ("clean", 100)


def test_rate_is_metadata_plus_background() -> None:
    assert total_kbps(1000, 3000, 1.0) == 32.0


def test_flat_fill_paints_only_the_support() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :] = (1, 2, 3)
    support = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    out = flat_fill(frame, support, (9, 9, 9))
    assert tuple(out[0, 0]) == (9, 9, 9)
    assert tuple(out[0, 1]) == (1, 2, 3)


def test_worse_lpips_at_lower_rate_is_kept() -> None:
    report = build_report(
        [
            {
                "clip": "clip_01",
                "candidate": "dino",
                "kbps": 40.0,
                "lpips": 0.40,
                "av1_kbps": 80.0,
                "av1_lpips": 0.20,
                "pose_pck": 0.4,
            }
        ]
    )
    assert report["dropped"] == []
    assert report["rows"][0]["kept"] is True
    assert report["rows"][0]["worse_lpips_lower_rate"] is True
    assert "dino" in report["recommendation"]
