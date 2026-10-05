"""Segmenter registry for the runner's config axis."""

from __future__ import annotations

import pytest

from src.components.segmentation import REGISTRY as SEGMENTERS
from src.contracts.errors import UnknownBackendError
from src.segmentation import BACKENDS
from src.segmentation.yoloe import YoloeSegmenter


def test_registry_names_every_segmentation_backend() -> None:
    assert set(SEGMENTERS.names()) == set(BACKENDS)
    assert SEGMENTERS.spec("sam3.1-multiplex").name == "sam31"
    assert SEGMENTERS.spec("yoloe-26x").defaults == {"size": "x", "weights": "yoloe-26x-seg.pt"}


def test_registry_builds_the_yoloe_backend_without_loading_a_model() -> None:
    backend = SEGMENTERS.build("yoloe-26s", model=object())
    assert isinstance(backend, YoloeSegmenter)
    assert backend.name == "yoloe-26s" and backend.weights_name == "yoloe-26s-seg.pt"


def test_unknown_segmenter_lists_the_registered_set() -> None:
    with pytest.raises(UnknownBackendError, match="Registered segmenter backends"):
        SEGMENTERS.spec("mask-rcnn")
