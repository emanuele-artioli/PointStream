"""Parse detector/pose/segmenter outputs without importing ultralytics.

Mocks in tests expose the same attributes real Results objects do (``.boxes``,
``.xyxy``, ``.cls``, ``.conf``, ``.id``, ``.masks``, ``.keypoints``). The
parsers never call into a model, so a unit test can feed them by hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import COCO_ID_TO_NAME, Detection


def as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_boxes(
    results: Any,
    *,
    frame_width: int,
    frame_height: int,
    allowed_class_ids: frozenset[int] | None = None,
    names: Mapping[int, str] | None = None,
) -> list[Detection]:
    """Turn an ultralytics-style result (or a list of them) into detections."""
    result = _first_result(results)
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []

    xyxy = as_numpy(boxes.xyxy)
    if xyxy.size == 0:
        return []
    if xyxy.ndim == 1:
        xyxy = xyxy.reshape(1, -1)

    cls = _optional_column(getattr(boxes, "cls", None), xyxy.shape[0], dtype=np.int32)
    conf = _optional_column(getattr(boxes, "conf", None), xyxy.shape[0], dtype=np.float32, fill=1.0)
    result_names = names or getattr(result, "names", None) or {}

    detections: list[Detection] = []
    for index in range(xyxy.shape[0]):
        class_id = int(cls[index]) if index < cls.shape[0] else None
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        class_name = _class_name(class_id, result_names)
        bbox = Box.from_xyxy(xyxy[index].tolist()).clip(frame_width, frame_height)
        detections.append(
            Detection(
                class_name=class_name,
                bbox=bbox,
                score=float(conf[index]) if index < conf.shape[0] else 1.0,
                class_id=class_id,
            )
        )
    return detections


def parse_supervision(
    raw: Any,
    *,
    frame_width: int,
    frame_height: int,
    names: Mapping[int, str] | None = None,
    allowed_class_ids: frozenset[int] | None = None,
) -> list[Detection]:
    """Parse RF-DETR / supervision ``Detections`` (``.xyxy``, ``.class_id``)."""
    xyxy = as_numpy(getattr(raw, "xyxy", None))
    if xyxy.size == 0:
        return []
    if xyxy.ndim == 1:
        xyxy = xyxy.reshape(1, -1)
    class_ids = _optional_column(getattr(raw, "class_id", None), xyxy.shape[0], dtype=np.int32)
    scores = _optional_column(
        getattr(raw, "confidence", None), xyxy.shape[0], dtype=np.float32, fill=1.0
    )
    names = names or {}
    detections: list[Detection] = []
    for index in range(xyxy.shape[0]):
        class_id = int(class_ids[index]) if index < class_ids.shape[0] else None
        if allowed_class_ids is not None and class_id not in allowed_class_ids:
            continue
        detections.append(
            Detection(
                class_name=_class_name(class_id, names),
                bbox=Box.from_xyxy(xyxy[index].tolist()).clip(frame_width, frame_height),
                score=float(scores[index]) if index < scores.shape[0] else 1.0,
                class_id=class_id,
            )
        )
    return detections


def first_mask(results: Any) -> np.ndarray | None:
    """The first instance mask as ``uint8`` HW in {0, 255}, or None."""
    result = _first_result(results)
    if result is None:
        return None
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None:
        return None
    array = as_numpy(data)
    if array.size == 0:
        return None
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        return None
    return (array > 0.5).astype(np.uint8) * 255


def first_keypoints(results: Any) -> np.ndarray | None:
    """The first pose as ``(N, 3)`` float32 (x, y, conf), or None."""
    result = _first_result(results)
    if result is None:
        return None
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None
    data = getattr(keypoints, "data", None)
    if data is None:
        xy = getattr(keypoints, "xy", None)
        conf = getattr(keypoints, "conf", None)
        if xy is None:
            return None
        xy_np = as_numpy(xy)
        if xy_np.ndim == 3:
            xy_np = xy_np[0]
        if conf is None:
            conf_np = np.ones((xy_np.shape[0], 1), dtype=np.float32)
        else:
            conf_np = as_numpy(conf)
            if conf_np.ndim == 2:
                conf_np = conf_np[0]
            conf_np = conf_np.reshape(-1, 1)
        return np.concatenate([xy_np.astype(np.float32), conf_np.astype(np.float32)], axis=1)
    array = as_numpy(data)
    if array.size == 0:
        return None
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    if array.shape[1] == 2:
        conf = np.ones((array.shape[0], 1), dtype=np.float32)
        array = np.concatenate([array.astype(np.float32), conf], axis=1)
    return array[:, :3].astype(np.float32)


def _first_result(results: Any) -> Any | None:
    if results is None:
        return None
    if isinstance(results, (list, tuple)):
        if not results:
            return None
        return results[0]
    return results


def _optional_column(
    value: Any, length: int, *, dtype: Any, fill: float = 0.0
) -> np.ndarray:
    if value is None:
        return np.full((length,), fill, dtype=dtype)
    array = as_numpy(value).reshape(-1)
    if array.shape[0] == 0:
        return np.full((length,), fill, dtype=dtype)
    return array.astype(dtype, copy=False)


def _class_name(class_id: int | None, names: Mapping[Any, str] | Sequence[str]) -> str:
    if class_id is None:
        return "object"
    if isinstance(names, Mapping) and class_id in names:
        return str(names[class_id])
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return COCO_ID_TO_NAME.get(class_id, f"class_{class_id}")
