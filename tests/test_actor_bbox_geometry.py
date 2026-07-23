"""Bounding-box geometry shared by every actor backend.

These three functions sit under detection, tracking, segmentation and payload
packing alike, and they fail quietly: a box clipped wrongly still produces a
crop, just of the wrong pixels. Splitting actor_components.py is what made them
reachable without model weights.
"""

from __future__ import annotations

import pytest

from src.encoder.actors.weights import _bbox_area, _bbox_center, _clip_bbox


def test_area_of_a_known_box():
    assert _bbox_area([10.0, 20.0, 30.0, 50.0]) == pytest.approx(600.0)


def test_an_inverted_box_has_no_area():
    """Never negative — a negative area would sort as the *smallest* actor.

    The player heuristic ranks candidates by area, so a negative value would
    quietly push a malformed detection to the bottom instead of flagging it.
    """
    assert _bbox_area([30.0, 50.0, 10.0, 20.0]) == 0.0


def test_a_degenerate_box_has_no_area():
    assert _bbox_area([10.0, 10.0, 10.0, 10.0]) == 0.0


def test_center_of_a_known_box():
    assert _bbox_center([0.0, 0.0, 10.0, 20.0]) == (5.0, 10.0)


def test_center_handles_negative_coordinates():
    assert _bbox_center([-10.0, -20.0, 10.0, 20.0]) == (0.0, 0.0)


def test_clipping_leaves_an_inside_box_alone():
    assert _clip_bbox([10.0, 20.0, 30.0, 40.0], width=100, height=100) == [10.0, 20.0, 30.0, 40.0]


def test_clipping_pulls_a_box_inside_the_frame():
    clipped = _clip_bbox([-50.0, -50.0, 500.0, 500.0], width=100, height=80)

    assert clipped[0] >= 0.0 and clipped[1] >= 0.0
    assert clipped[2] <= 100.0 and clipped[3] <= 80.0


def test_a_clipped_box_always_has_positive_extent():
    """The width/height floor is what stops a zero-size crop reaching a model.

    A detection entirely outside the frame would otherwise clip to an empty box,
    and an empty crop is the kind of thing that surfaces as a shape error deep
    inside a backend rather than here.
    """
    clipped = _clip_bbox([500.0, 500.0, 600.0, 600.0], width=100, height=80)

    assert clipped[2] > clipped[0]
    assert clipped[3] > clipped[1]


def test_clipping_an_inverted_box_still_yields_a_valid_box():
    clipped = _clip_bbox([80.0, 60.0, 10.0, 20.0], width=100, height=80)

    assert clipped[2] > clipped[0]
    assert clipped[3] > clipped[1]
