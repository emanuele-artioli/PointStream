"""Exact-name lookup, unknown-name errors, mocks, and the weight-resolution check."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.components.detection import REGISTRY as DETECTORS
from src.components.detection.geometry import Box
from src.components.detection.rfdetr import MISSING_MESSAGE, RfDetrDetector
from src.components.detection.sam3 import Sam3Detector
from src.components.detection.types import Detection
from src.components.detection.weights import (
    WeightResolutionError,
    assert_named_weights_resolve,
    named_weights,
    resolve_weight,
)
from src.components.detection.yolo import YoloDetector
from src.components.pose import REGISTRY as POSE
from src.components.segmentation import REGISTRY as SEGMENTERS
from src.components.selection import REGISTRY as SELECTION
from src.components.tracking import REGISTRY as TRACKING
from src.contracts.config import default, validate_backends
from src.contracts.errors import ConfigValueError, UnknownBackendError


def _boxes(xyxy: list[list[float]], cls: list[int], conf: list[float] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        cls=np.asarray(cls, dtype=np.float32),
        conf=np.asarray(conf or [1.0] * len(xyxy), dtype=np.float32),
    )


def _result(boxes: SimpleNamespace, names: dict[int, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=boxes,
        names=names or {0: "person", 32: "sports ball", 38: "tennis racket"},
    )


class _PredictModel:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return [self.result]


def test_default_config_backends_are_registered_by_exact_name() -> None:
    """Substring dispatch is the thing being retired; names match exactly."""
    config = default()
    validate_backends(
        config,
        registries={
            "detector": DETECTORS,
            "selection": SELECTION,
            "tracking": TRACKING,
            "pose": POSE,
            "segmenter": SEGMENTERS,
        },
    )
    assert DETECTORS.spec("yolo").name == "yolo"
    assert DETECTORS.spec("yolo26").name == "yolo"
    assert SELECTION.spec("heuristic").name == "heuristic"
    assert TRACKING.spec("tracker").name == "tracker"
    assert POSE.spec("yolo").name == "yolo"
    assert SEGMENTERS.spec("yolo").name == "yolo"


def test_unknown_detector_name_lists_the_registered_set_and_suggests_a_close_match() -> None:
    with pytest.raises(UnknownBackendError, match="Did you mean") as caught:
        DETECTORS.spec("yolo3")
    assert "yolo" in caught.value.known
    assert "sam3" in caught.value.known
    assert "rf-detr" in caught.value.known


def test_sam3_and_rf_detr_are_registered_under_their_exact_names() -> None:
    assert DETECTORS.spec("sam3").name == "sam3"
    assert DETECTORS.spec("rf-detr").name == "rf-detr"
    assert SEGMENTERS.spec("sam3").name == "sam3"


def test_yolo_detector_parses_mocked_boxes_and_drops_unrelated_classes() -> None:
    model = _PredictModel(
        _result(
            _boxes(
                [[10, 10, 50, 80], [100, 20, 140, 70], [5, 5, 20, 20]],
                [0, 38, 2],
            )
        )
    )
    detector = YoloDetector(model=model)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    names = {item.class_name for item in detections}
    assert names == {"person", "tennis racket"}
    assert all(item.bbox.x2 <= 160 and item.bbox.y2 <= 120 for item in detections)


def test_sam3_detector_forwards_the_class_prompt_to_the_mocked_model() -> None:
    model = _PredictModel(_result(_boxes([[1, 1, 10, 20]], [0]), names={0: "tennis player"}))
    detector = Sam3Detector(model=model, prompt="tennis player")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert detections[0].class_name == "tennis player"
    assert model.calls[0]["text"] == ["tennis player"]


def test_rfdetr_without_an_injected_model_explains_why_it_is_not_installed() -> None:
    with pytest.raises(RuntimeError, match="transformers") as caught:
        RfDetrDetector()
    assert "4.46.3" in str(caught.value)
    assert MISSING_MESSAGE.split(":")[0] in str(caught.value)


def test_rfdetr_with_an_injected_model_does_not_need_the_package() -> None:
    model = _PredictModel(_result(_boxes([[2, 2, 8, 16]], [0])))
    detector = RfDetrDetector(model=model)
    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "person"


def test_a_dangling_symlink_is_rejected_rather_than_downloaded(tmp_path: Path) -> None:
    weights = tmp_path / "assets" / "weights"
    weights.mkdir(parents=True)
    link = weights / "yolo26n.pt"
    link.symlink_to(tmp_path / "does-not-exist.pt")
    with pytest.raises(WeightResolutionError, match="dangling symlink"):
        resolve_weight("yolo26n.pt", root=tmp_path)


def test_a_missing_weight_is_rejected_without_downloading(tmp_path: Path) -> None:
    (tmp_path / "assets" / "weights").mkdir(parents=True)
    with pytest.raises(WeightResolutionError, match="not found"):
        resolve_weight("yolo26n.pt", root=tmp_path)


def test_shipped_default_weight_names_resolve_in_a_planted_tree(tmp_path: Path) -> None:
    """Every checkpoint the default config names must be a real file, not a wish."""
    config = default()
    names = named_weights(config)
    assert names == {
        "detector": "yolo26n.pt",
        "pose": "yolo26n-pose.pt",
        "segmenter": "yolo26n-seg.pt",
    }
    planted = tmp_path / "assets" / "weights"
    planted.mkdir(parents=True)
    for filename in names.values():
        (planted / filename).write_bytes(b"stub")
    assert_named_weights_resolve(config, root=tmp_path)


def test_shipped_default_weight_names_fail_when_any_link_is_dangling(tmp_path: Path) -> None:
    config = default()
    planted = tmp_path / "assets" / "weights"
    planted.mkdir(parents=True)
    (planted / "yolo26n.pt").write_bytes(b"ok")
    (planted / "yolo26n-pose.pt").write_bytes(b"ok")
    (planted / "yolo26n-seg.pt").symlink_to(tmp_path / "missing-seg.pt")
    with pytest.raises(ConfigValueError, match="dangling"):
        assert_named_weights_resolve(config, root=tmp_path)


def test_roi_predictor_maps_a_crop_box_back_to_the_frame() -> None:
    crop_result = _result(_boxes([[2, 3, 6, 9]], [0]))
    model = _PredictModel(crop_result)
    detector = YoloDetector(model=model)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mapped = detector.predict_roi(frame, Box(10, 10, 30, 40), "person")
    assert mapped is not None
    # crop origin is the padded box, at least shifted from (0, 0)
    assert mapped.x1 >= 2.0


@pytest.mark.integration
def test_yolo_detector_loads_the_shipped_checkpoint() -> None:
    pytest.importorskip("ultralytics")
    from src.components.detection.weights import repo_root, resolve_weight

    try:
        resolve_weight("yolo26n.pt", root=repo_root())
    except WeightResolutionError as exc:
        pytest.skip(str(exc))
    detector = YoloDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    assert all(isinstance(item, Detection) for item in detections)
