"""Bind perception and representation backends into C2 stage callables.

The runner is the only layer allowed to look a registry up. These helpers
turn a config name into detections, masks, appearance bytes, motion bytes
and a temporal schedule. Construction is lazy: a stage that is injected over
or never invoked never loads weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.transport.payload import PlannedSchedule, dump_schedule
from src.contracts.keypoints import schema as lookup_schema
from src.contracts.lattice import ART_KEYPOINTS, ART_SCHEDULE, ART_SUBJECTS
from src.pipeline.reconstruction.compositor import heuristic_mask
from src.pipeline.reconstruction.reconstruct import ObjectRequest

#: Same bag key the runner uses for caller-injected subjects (`stages.OBJECTS`).
_ORACLE = "objects"


class _Ctx(Protocol):
    """The fields `StageContext` must expose for registry lookup."""

    builders: Mapping[str, Callable[..., Any]] | None
    config: Any

_AXIS_MODULES = {
    "detector": "src.components.detection",
    "selection": "src.components.selection",
    "tracking": "src.components.tracking",
    "pose": "src.components.pose",
    "segmenter": "src.components.segmentation",
    "appearance": "src.components.appearance",
    "motion": "src.components.motion",
    "temporal": "src.components.temporal",
}


def build_backend(ctx: _Ctx, axis: str, name: str, **kwargs: Any) -> Any:
    """Construct the named backend, honouring a test-injected factory first."""
    if name in ("", "none"):
        return None
    if ctx.builders is not None and axis in ctx.builders:
        return ctx.builders[axis](name, **kwargs)
    import importlib

    module = importlib.import_module(_AXIS_MODULES[axis])
    return module.REGISTRY.build(name, **kwargs)


def object_tuple(value: object) -> tuple[ObjectRequest, ...]:
    if value is None:
        return ()
    if isinstance(value, ObjectRequest):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, ObjectRequest))
    return ()


def oracle_objects(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    """Subjects the caller injected. Non-empty means perception backends stay idle."""
    return object_tuple(bag.get(_ORACLE))


def subjects_from_bag(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    """Prefer the latest rewritten subject list, then detections, then oracle."""
    for key in ("masks", ART_SUBJECTS, "salient-subjects", "detection", _ORACLE):
        found = object_tuple(bag.get(key))
        if found:
            return found
    return ()


def object_from_detection(
    source: np.ndarray, frame_index: int, detection: Detection
) -> ObjectRequest:
    frames, height, width, _ = source.shape
    box = detection.bbox.clip(width, height)
    x1, y1, x2, y2 = (
        int(np.floor(box.x1)),
        int(np.floor(box.y1)),
        int(np.ceil(box.x2)),
        int(np.ceil(box.y2)),
    )
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    crop = np.asarray(source[frame_index, y1:y2, x1:x2])
    if crop.size == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    mask = np.zeros((frames, height, width), dtype=bool)
    mask[frame_index] = heuristic_mask((x1, y1, x2, y2), height, width)
    return ObjectRequest(
        object_id=str(detection.track_id or f"{detection.class_name}-{frame_index}"),
        appearance=crop,
        bbox=(x1, y1, x2, y2),
        mask=mask,
        frame_index=frame_index,
    )


def as_detection(item: ObjectRequest) -> Detection:
    x1, y1, x2, y2 = item.bbox
    class_name = "sports ball" if "ball" in item.object_id else "person"
    return Detection(
        class_name=class_name,
        bbox=Box(float(x1), float(y1), float(x2), float(y2)),
        track_id=item.object_id,
    )


def detect_clip(source: np.ndarray, detector: Any) -> tuple[ObjectRequest, ...]:
    records: list[ObjectRequest] = []
    for index, frame in enumerate(source):
        found = detector.detect(frame)
        for detection in found:
            records.append(object_from_detection(source, index, detection))
    return tuple(records)


def filter_selected(
    subjects: Sequence[ObjectRequest],
    selector: Any,
    frame_shape: tuple[int, int],
) -> tuple[ObjectRequest, ...]:
    if not subjects:
        return ()
    by_frame: dict[int, list[ObjectRequest]] = {}
    for item in subjects:
        by_frame.setdefault(item.frame_index, []).append(item)
    kept: list[ObjectRequest] = []
    for group in by_frame.values():
        detections = [as_detection(item) for item in group]
        selected = list(selector.select(detections, frame_shape))
        selected_ids = {det.track_id for det in selected if det.track_id}
        selected_boxes = {
            (round(det.bbox.x1), round(det.bbox.y1), round(det.bbox.x2), round(det.bbox.y2))
            for det in selected
        }
        for item in group:
            box = (item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])
            if item.object_id in selected_ids or box in selected_boxes:
                kept.append(item)
    return tuple(kept)


def perception_frames(bag: Mapping[str, Any], object_id: str, frame_count: int) -> frozenset[int]:
    schedule = bag.get(ART_SCHEDULE)
    if not isinstance(schedule, PlannedSchedule):
        return frozenset(range(frame_count))
    marked = schedule.perception_frames(object_id)
    if not marked:
        return frozenset(range(frame_count))
    return frozenset(int(index) for index in marked)


def encode_appearance(ctx: _Ctx, subjects: Sequence[ObjectRequest]) -> dict[str, Any]:
    name = ctx.config.appearance.representation
    kwargs: dict[str, Any] = {}
    if name == "compressed-image":
        kwargs = {
            "quality": ctx.config.appearance.jpeg_quality,
            "downscale": ctx.config.appearance.downscale,
        }
    encoder = build_backend(ctx, "appearance", name, **kwargs)
    if encoder is None:
        return {"byte_count": 0, "representation": name}
    total = 0
    for item in subjects:
        crop = np.asarray(item.appearance)
        if crop.size == 0:
            continue
        _descriptor, payload = encoder.encode(crop)
        total += len(payload) if isinstance(payload, (bytes, bytearray)) else 0
    return {"byte_count": int(total), "representation": name}


def encode_motion(
    ctx: _Ctx,
    subjects: Sequence[ObjectRequest],
    keypoints: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    name = ctx.config.motion.representation
    kwargs: dict[str, Any] = {}
    if name == "keypoints":
        kwargs["schema"] = lookup_schema(ctx.config.pose.schema)
    encoder = build_backend(ctx, "motion", name, **kwargs)
    if encoder is None:
        return {"byte_count": 0, "representation": name}
    total = 0
    for item in subjects:
        if name == "keypoints":
            array = keypoints.get(item.object_id)
        else:
            x1, y1, x2, y2 = item.bbox
            array = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
        if array is None:
            continue
        _descriptor, payload = encoder.encode(array)
        total += len(payload) if isinstance(payload, (bytes, bytearray)) else 0
    return {"byte_count": int(total), "representation": name}


def plan_temporal(
    ctx: _Ctx, source: np.ndarray, subjects: Sequence[ObjectRequest]
) -> PlannedSchedule:
    from src.components.temporal.policy import ConfigurableTemporalPolicy

    policy = ConfigurableTemporalPolicy(temporal=ctx.config.temporal)
    ids = tuple(dict.fromkeys(item.object_id for item in subjects)) or ("object",)
    return policy.plan(
        frame_count=int(source.shape[0]),
        object_ids=ids,
        motion=_frame_motion(subjects, int(source.shape[0])),
    )


def schedule_bytes(planned: PlannedSchedule) -> int:
    return len(json.dumps(dump_schedule(planned), sort_keys=True).encode("utf-8"))


def keypoints_from_bag(bag: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = bag.get(ART_KEYPOINTS)
    if isinstance(raw, Mapping) and "by_object" in raw:
        values = raw["by_object"]
        if isinstance(values, Mapping):
            return {str(key): np.asarray(value) for key, value in values.items()}
    return {}


def metadata_bytes(bag: Mapping[str, Any]) -> int:
    total = 0
    motion = bag.get("motion-payload")
    if isinstance(motion, Mapping) and "byte_count" in motion:
        total += int(motion["byte_count"])
    keypoints = bag.get(ART_KEYPOINTS)
    if isinstance(keypoints, Mapping) and "byte_count" in keypoints:
        total += int(keypoints["byte_count"])
    elif isinstance(keypoints, Sequence) and not isinstance(keypoints, (str, bytes)):
        for pose in keypoints:
            if pose is None:
                continue
            values = getattr(pose, "values", None)
            if values is not None:
                total += int(np.asarray(values).nbytes)
            elif hasattr(pose, "nbytes"):
                total += int(pose.nbytes)
    schedule = bag.get(ART_SCHEDULE)
    if isinstance(schedule, PlannedSchedule):
        total += schedule_bytes(schedule)
    elif isinstance(schedule, Mapping) and "byte_count" in schedule:
        total += int(schedule["byte_count"])
    return total


def _frame_motion(subjects: Sequence[ObjectRequest], frame_count: int) -> list[float]:
    magnitudes = [0.0] * frame_count
    by_id: dict[str, list[ObjectRequest]] = {}
    for item in subjects:
        by_id.setdefault(item.object_id, []).append(item)
    for group in by_id.values():
        ordered = sorted(group, key=lambda item: item.frame_index)
        previous = None
        for item in ordered:
            if previous is not None and 0 <= item.frame_index < frame_count:
                dx = float(item.bbox[0] - previous.bbox[0])
                dy = float(item.bbox[1] - previous.bbox[1])
                magnitudes[item.frame_index] += float(np.hypot(dx, dy))
            previous = item
    return magnitudes


Builders = Mapping[str, Callable[..., Any]]

__all__ = [
    "Builders",
    "build_backend",
    "detect_clip",
    "encode_appearance",
    "encode_motion",
    "filter_selected",
    "keypoints_from_bag",
    "metadata_bytes",
    "object_tuple",
    "oracle_objects",
    "perception_frames",
    "plan_temporal",
    "schedule_bytes",
    "subjects_from_bag",
]
