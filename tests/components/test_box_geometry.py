"""Box area, centre, and clip. Ported from the pre-rewrite actor helpers.

The old helpers quietly accepted inverted boxes and clipped them back to a
positive extent. ``Box`` refuses inversion at construction — that case is
not ported, because the new type makes it an error instead of a crop of the
wrong pixels.
"""

from __future__ import annotations

import pytest

from src.components.detection.geometry import Box


def test_area_of_a_known_box() -> None:
    assert Box.from_xyxy([10.0, 20.0, 30.0, 50.0]).area == pytest.approx(600.0)


def test_an_inverted_box_is_rejected() -> None:
    """A negative area used to sort as the smallest actor. Construction fails instead."""
    with pytest.raises(ValueError, match="inverted"):
        Box.from_xyxy([30.0, 50.0, 10.0, 20.0])


def test_a_degenerate_box_has_no_area() -> None:
    assert Box.from_xyxy([10.0, 10.0, 10.0, 10.0]).area == 0.0


def test_center_of_a_known_box() -> None:
    assert Box.from_xyxy([0.0, 0.0, 10.0, 20.0]).center == (5.0, 10.0)


def test_center_handles_negative_coordinates() -> None:
    assert Box.from_xyxy([-10.0, -20.0, 10.0, 20.0]).center == (0.0, 0.0)


def test_clipping_leaves_an_inside_box_alone() -> None:
    clipped = Box.from_xyxy([10.0, 20.0, 30.0, 40.0]).clip(width=100, height=100)
    assert clipped.xyxy == (10.0, 20.0, 30.0, 40.0)


def test_clipping_pulls_a_box_inside_the_frame() -> None:
    clipped = Box.from_xyxy([-50.0, -50.0, 500.0, 500.0]).clip(width=100, height=80)
    assert clipped.x1 >= 0.0 and clipped.y1 >= 0.0
    assert clipped.x2 <= 100.0 and clipped.y2 <= 80.0


def test_a_clipped_box_always_has_positive_extent() -> None:
    """The width/height floor is what stops a zero-size crop reaching a model."""
    clipped = Box.from_xyxy([500.0, 500.0, 600.0, 600.0]).clip(width=100, height=80)
    assert clipped.x2 > clipped.x1
    assert clipped.y2 > clipped.y1
