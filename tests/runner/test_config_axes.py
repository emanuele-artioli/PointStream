"""Required behaviour: a named ablation axis reaches the run.

One property per newly-wired axis — changing the name (or the temporal
interval that stands in for a name) changes a measured output. Not one test
per config field.

Deliberately not tested: third-party YOLO/SAM inference, constructing every
registered backend, residual.codec (BP24), selection/tracking/rigid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.pose.wire import Pose
from src.contracts.codecs import RateControl
from src.contracts.config import (
    AppearanceConfig,
    BackendConfig,
    LatticeConfig,
    MotionConfig,
    PointstreamConfig,
    PoseConfig,
    ResidualConfig,
    TemporalConfig,
)
from src.contracts.errors import UnknownBackendError
from src.contracts.keypoints import COCO_17
from src.contracts.lattice import (
    ART_KEYPOINTS,
    ART_MASKS,
    ART_MOTION_PAYLOAD,
    ART_SCHEDULE,
    ART_SUBJECTS,
    OPTIONAL_STAGES,
    SOURCE_PASSTHROUGH,
)
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner import lattice_config_from, run
from tests.pipeline.clocks import ClockedStage


def _clip(frames: int = 4, height: int = 48, width: int = 64) -> np.ndarray:
    clip = np.full((frames, height, width, 3), 90, dtype=np.uint8)
    for index in range(frames):
        top, left = 8 + index * 2, 10 + index * 3
        clip[index, top : top + 16, left : left + 16] = 210
    return clip


def _object(clip: np.ndarray) -> ObjectRequest:
    return ObjectRequest(
        object_id="player",
        appearance=clip[0, 8:24, 10:26].copy(),
        bbox=(10, 8, 26, 24),
        frame_index=0,
    )


def _residual() -> ResidualConfig:
    return ResidualConfig(
        rate_control=RateControl.CRF,
        block_size=8,
        block_threshold=4.0,
        background_downscale=4,
    )


def _lattice(**enabled: bool) -> LatticeConfig:
    flags = {
        "scene_classification": False,
        "detection": False,
        "selection": False,
        "tracking": False,
        "appearance": False,
        "motion": False,
        "temporal_policy": False,
        "pose": False,
        "segmentation": False,
        "rigid_objects": False,
        "background": True,
        "generation": False,
        "residual": True,
    }
    flags.update(enabled)
    return LatticeConfig(**flags)


def _config(*, lattice: LatticeConfig, **overrides: object) -> PointstreamConfig:
    return PointstreamConfig(lattice=lattice, residual=_residual(), **overrides)  # type: ignore[arg-type]


class _BoxDetector:
    def __init__(self, box: Box) -> None:
        self.box = box

    def detect(self, frame: np.ndarray) -> list[Detection]:
        _ = frame
        return [Detection(class_name="person", bbox=self.box, track_id="player")]


class _ShiftPose:
    def __init__(self, shift: float) -> None:
        self.shift = shift

    def estimate(self, frame, detection, **kwargs):  # noqa: ANN001
        _ = (frame, detection, kwargs)
        n = len(COCO_17)
        values = np.zeros((n, 3), dtype=np.float32)
        values[:, 0] = 12.0 + self.shift
        values[:, 1] = 20.0 + self.shift
        values[:, 2] = 1.0
        return Pose(schema=COCO_17, values=values, present=np.ones(n, dtype=bool))


class _PatternSeg:
    def __init__(self, every: int) -> None:
        self.every = every

    def segment(self, frame, detection):  # noqa: ANN001
        _ = frame
        box = detection.bbox
        height = max(1, int(np.ceil(box.y2) - np.floor(box.y1)))
        width = max(1, int(np.ceil(box.x2) - np.floor(box.x1)))
        mask = np.zeros((height, width), dtype=bool)
        mask[:: self.every, :: self.every] = True
        return mask


def _art(result: Any, key: str) -> Any:
    return result.chunks[0].bag[key]


def test_detector_backend_name_changes_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping detector.backend must change who is named a subject."""

    def build(name: str, **kwargs: object) -> _BoxDetector:
        _ = kwargs
        boxes = {
            "yolo": Box(0, 0, 12, 12),
            "sam3": Box(24, 20, 48, 44),
        }
        return _BoxDetector(boxes[name])

    monkeypatch.setattr("src.components.detection.REGISTRY.build", build)
    clip = _clip()
    left = run(
        _config(
            lattice=_lattice(detection=True),
            detector=BackendConfig(backend="yolo", model="yolo26n.pt"),
        ),
        [clip],
    )
    right = run(
        _config(
            lattice=_lattice(detection=True),
            detector=BackendConfig(backend="sam3", model="sam3.pt"),
        ),
        [clip],
    )
    left_box = _art(left, ART_SUBJECTS)[0].bbox
    right_box = _art(right, ART_SUBJECTS)[0].bbox
    assert left_box != right_box
    assert left.delivered_quality.whole_frame() != right.delivered_quality.whole_frame()


def test_pose_backend_name_changes_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping pose.backend must change the keypoints on the bag.

    The registry's only class is YOLO; ``yolo`` and ``yolo-pose`` are aliases
    for it. The hook keys on the name the config spelled, which is what an
    ablation writes down.
    """

    def build(name: str, **kwargs: object) -> _ShiftPose:
        _ = kwargs
        return _ShiftPose(0.0 if name == "yolo" else 40.0)

    monkeypatch.setattr("src.components.pose.REGISTRY.build", build)
    monkeypatch.setattr(
        "src.components.detection.REGISTRY.build",
        lambda name, **kwargs: _BoxDetector(Box(0, 0, 16, 16)),
    )
    clip = _clip()
    left = run(
        _config(
            lattice=_lattice(detection=True, pose=True),
            pose=PoseConfig(backend="yolo", model="yolo26n-pose.pt"),
        ),
        [clip],
    )
    right = run(
        _config(
            lattice=_lattice(detection=True, pose=True),
            pose=PoseConfig(backend="yolo-pose", model="yolo26n-pose.pt"),
        ),
        [clip],
    )
    left_pose = _art(left, ART_KEYPOINTS)[0]
    right_pose = _art(right, ART_KEYPOINTS)[0]
    assert float(left_pose.values[0, 0]) != float(right_pose.values[0, 0])


def test_segmenter_backend_name_changes_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping segmenter.backend must change the mask pixels."""

    def build(name: str, **kwargs: object) -> _PatternSeg:
        _ = kwargs
        return _PatternSeg(1 if name == "yolo" else 2)

    monkeypatch.setattr("src.components.segmentation.REGISTRY.build", build)
    monkeypatch.setattr(
        "src.components.detection.REGISTRY.build",
        lambda name, **kwargs: _BoxDetector(Box(0, 0, 32, 32)),
    )
    clip = _clip()
    left = run(
        _config(
            lattice=_lattice(detection=True, segmentation=True),
            segmenter=BackendConfig(backend="yolo", model="yolo26n-seg.pt"),
        ),
        [clip],
    )
    right = run(
        _config(
            lattice=_lattice(detection=True, segmentation=True),
            segmenter=BackendConfig(backend="sam3", model="sam3.pt"),
        ),
        [clip],
    )
    left_mask = _art(left, ART_MASKS)["player"]
    right_mask = _art(right, ART_MASKS)["player"]
    assert int(left_mask.sum()) != int(right_mask.sum())
    assert left.sizes.residual != right.sizes.residual or not np.array_equal(
        left.frames, right.frames
    )


def test_appearance_representation_name_changes_the_result() -> None:
    """Swapping appearance.representation must change the payload byte count."""
    clip = _clip()
    objects = ((_object(clip),),)
    left = run(
        _config(
            lattice=_lattice(detection=True, appearance=True),
            appearance=AppearanceConfig(representation="compressed-image", jpeg_quality=90),
        ),
        [clip],
        objects=objects,
    )
    right = run(
        _config(
            lattice=_lattice(detection=True, appearance=True),
            appearance=AppearanceConfig(representation="image-embedding"),
        ),
        [clip],
        objects=objects,
    )
    left_bytes = _art(left, "appearance-payload")["byte_count"]
    right_bytes = _art(right, "appearance-payload")["byte_count"]
    assert left_bytes != right_bytes
    assert left.sizes.actor_reference != right.sizes.actor_reference


def test_motion_representation_name_changes_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping motion.representation must change the motion payload."""

    def pose_build(name: str, **kwargs: object) -> _ShiftPose:
        _ = (name, kwargs)
        return _ShiftPose(3.0)

    monkeypatch.setattr("src.components.pose.REGISTRY.build", pose_build)
    monkeypatch.setattr(
        "src.components.detection.REGISTRY.build",
        lambda name, **kwargs: _BoxDetector(Box(0, 0, 16, 16)),
    )
    clip = _clip()
    left = run(
        _config(
            lattice=_lattice(detection=True, pose=True, motion=True),
            motion=MotionConfig(representation="keypoints"),
            pose=PoseConfig(backend="yolo", schema="coco-17"),
        ),
        [clip],
    )
    right = run(
        _config(
            lattice=_lattice(detection=True, pose=True, motion=True),
            motion=MotionConfig(representation="sparse-trajectories"),
            pose=PoseConfig(backend="yolo", schema="coco-17"),
        ),
        [clip],
    )
    left_motion = _art(left, ART_MOTION_PAYLOAD)
    right_motion = _art(right, ART_MOTION_PAYLOAD)
    assert left_motion["representation"] != right_motion["representation"]
    assert left_motion["byte_count"] != right_motion["byte_count"]


def test_temporal_keyframe_interval_changes_the_result() -> None:
    """The temporal axis has no backend name; the interval is how it is named."""
    clip = _clip(frames=8)
    objects = ((_object(clip),),)
    left = run(
        _config(
            lattice=_lattice(detection=True, temporal_policy=True),
            temporal=TemporalConfig(
                metadata_sparsity=False,
                pipeline_sparsity=True,
                generation_sparsity=False,
                keyframe_interval=2,
                delta_threshold=1e6,
            ),
        ),
        [clip],
        objects=objects,
    )
    right = run(
        _config(
            lattice=_lattice(detection=True, temporal_policy=True),
            temporal=TemporalConfig(
                metadata_sparsity=False,
                pipeline_sparsity=True,
                generation_sparsity=False,
                keyframe_interval=8,
                delta_threshold=1e6,
            ),
        ),
        [clip],
        objects=objects,
    )
    left_plan = _art(left, ART_SCHEDULE)
    right_plan = _art(right, ART_SCHEDULE)
    left_n = len(left_plan.perception["player"])
    right_n = len(right_plan.perception["player"])
    assert left_n != right_n


def test_unknown_detector_name_fails_at_the_stage_not_silently() -> None:
    """A misspelled backend must not fall through to the pass-through."""
    clip = _clip()
    with pytest.raises(UnknownBackendError):
        run(
            _config(
                lattice=_lattice(detection=True),
                detector=BackendConfig(backend="not-a-detector"),
            ),
            [clip],
        )


def test_all_off_still_ignores_named_perception_backends() -> None:
    """Wiring must not start a disabled stage. Call counts stay 0."""
    clocks = {name: ClockedStage() for name in OPTIONAL_STAGES}
    clip = _clip(frames=1)
    result = run(
        PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH)),
        [clip],
        backends=clocks,
    )
    assert np.array_equal(clip, result.frames)
    assert result.sizes.residual == 0
    for name, clock in clocks.items():
        assert clock.calls == 0, name
