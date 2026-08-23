"""Segmenter registry and mocked mask parsing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.segmentation import REGISTRY as SEGMENTERS
from src.components.segmentation.sam3 import Sam3Segmenter
from src.components.segmentation.yolo import YoloSegmenter
from src.contracts.errors import UnknownBackendError


def test_yolo_and_sam3_are_registered() -> None:
    assert SEGMENTERS.spec("yolo").name == "yolo"
    assert SEGMENTERS.spec("yolo-seg").name == "yolo"
    assert SEGMENTERS.spec("sam3").name == "sam3"


def test_unknown_segmenter_lists_the_registered_set() -> None:
    with pytest.raises(UnknownBackendError, match="Registered segmenter backends"):
        SEGMENTERS.spec("mask-rcnn")


def test_yolo_segmenter_returns_a_crop_sized_mask_from_a_mock() -> None:
    mask = np.zeros((10, 12), dtype=np.float32)
    mask[2:8, 3:9] = 1.0
    result = SimpleNamespace(masks=SimpleNamespace(data=mask[None, ...]))
    model = SimpleNamespace(predict=lambda **_kwargs: [result])
    segmenter = YoloSegmenter(model=model)
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    out = segmenter.segment(frame, Detection("player", Box(5, 5, 17, 15)))
    assert out is not None
    assert out.shape == (10, 12)
    assert out.dtype == np.uint8
    assert int(out.max()) == 255


def test_sam3_segmenter_crops_a_full_frame_mask_to_the_box() -> None:
    full = np.zeros((40, 50), dtype=np.float32)
    full[6:14, 8:18] = 1.0
    result = SimpleNamespace(masks=SimpleNamespace(data=full[None, ...]))
    calls: list[dict[str, object]] = []

    def predict(**kwargs: object) -> list[SimpleNamespace]:
        calls.append(kwargs)
        return [result]

    segmenter = Sam3Segmenter(model=SimpleNamespace(predict=predict))
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    out = segmenter.segment(frame, Detection("player", Box(8, 6, 18, 14)))
    assert out is not None
    assert out.shape == (8, 10)
    assert calls[0]["bboxes"] == [[8.0, 6.0, 18.0, 14.0]]
