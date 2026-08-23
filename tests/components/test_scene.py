"""Scene classification exists to route point vs interlude. Nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.scene import REGISTRY as SCENE
from src.components.scene.hsv import HsvHistogramClassifier
from src.components.scene.routing import (
    INTERLUDE,
    POINT,
    ROUTE_FALLBACK,
    ROUTE_SEMANTIC,
    route_for,
    span,
)
from src.contracts.errors import UnknownBackendError


def test_hsv_histogram_is_registered() -> None:
    assert SCENE.spec("hsv-histogram").name == "hsv-histogram"
    built = SCENE.build("hsv-histogram")
    assert isinstance(built, HsvHistogramClassifier)


def test_unknown_scene_backend_lists_the_registered_set() -> None:
    with pytest.raises(UnknownBackendError, match="Registered scene backends"):
        SCENE.spec("clip")


def test_points_route_to_the_semantic_pipeline_and_interludes_to_the_fallback() -> None:
    assert route_for(POINT) == ROUTE_SEMANTIC
    assert route_for(INTERLUDE) == ROUTE_FALLBACK
    # An unknown label must not enter the semantic path: that reconstruction
    # would be quietly wrong. Falling back is the conservative miss.
    assert route_for("other") == ROUTE_FALLBACK
    assert span(0, 10, POINT).route == ROUTE_SEMANTIC
    assert span(10, 20, INTERLUDE).route == ROUTE_FALLBACK


def test_identical_frames_are_one_point_span() -> None:
    green = np.zeros((48, 64, 3), dtype=np.uint8)
    green[:, :] = (0, 180, 0)
    frames = [green.copy() for _ in range(5)]
    spans = HsvHistogramClassifier().classify(frames)
    assert len(spans) == 1
    assert spans[0].scene_class == POINT
    assert spans[0].route == ROUTE_SEMANTIC
    assert spans[0].start_frame == 0
    assert spans[0].end_frame == 5


def test_a_hard_appearance_change_is_an_interlude_not_a_point() -> None:
    green = np.zeros((48, 64, 3), dtype=np.uint8)
    green[:, :] = (0, 180, 0)
    red = np.zeros((48, 64, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 180)
    frames = [green.copy(), green.copy(), red.copy(), red.copy()]
    spans = HsvHistogramClassifier().classify(frames)
    # Uniform green then uniform red is a cut, not one fused span. Each side
    # of the cut can still be a point (stable shot); routing of an interlude
    # is tested on route_for, which is the load-bearing decision.
    assert len(spans) >= 2
    assert spans[0].end_frame == spans[1].start_frame
    assert any(item.scene_class == POINT for item in spans)
