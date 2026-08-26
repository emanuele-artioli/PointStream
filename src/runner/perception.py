"""Registry lookup and bag helpers the runner stages call.

Construction is lazy: a stage that is injected over or never invoked never
loads weights. Helpers that stages do not call do not live here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.transport.payload import PlannedSchedule, dump_schedule
from src.contracts.lattice import ART_KEYPOINTS, ART_SCHEDULE, ART_SUBJECTS
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


def subjects_from_bag(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    """Prefer the latest rewritten subject list, then detections, then oracle."""
    for key in ("masks", ART_SUBJECTS, "salient-subjects", "detection", _ORACLE):
        found = object_tuple(bag.get(key))
        if found:
            return found
    return ()


def as_detection(item: ObjectRequest) -> Detection:
    x1, y1, x2, y2 = item.bbox
    class_name = "sports ball" if "ball" in item.object_id else "person"
    return Detection(
        class_name=class_name,
        bbox=Box(float(x1), float(y1), float(x2), float(y2)),
        track_id=item.object_id,
    )


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


def schedule_bytes(planned: PlannedSchedule) -> int:
    return len(json.dumps(dump_schedule(planned), sort_keys=True).encode("utf-8"))


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


Builders = Mapping[str, Callable[..., Any]]

__all__ = [
    "Builders",
    "as_detection",
    "build_backend",
    "filter_selected",
    "metadata_bytes",
    "object_tuple",
    "schedule_bytes",
    "subjects_from_bag",
]
