"""Rigid objects are optional lattice rows with no skeleton."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background import REGISTRY as BACKGROUND
from src.components.rigid import REGISTRY as RIGID
from src.components.rigid.strategy import bind
from src.components.rigid.types import ObservedObject, PlayerPose
from src.contracts import config
from src.contracts.config import BackendConfig, LatticeConfig, PointstreamConfig, validate_backends
from src.contracts.errors import ConfigError, ConfigValueError, UnknownBackendError

_REGISTRIES = {"background": BACKGROUND, "rigid": RIGID}


def _racket_mask(height: int = 40, width: int = 40) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:32, 16:24] = 255
    mask[8:16, 10:30] = 255
    return mask


def _player_pose(frame_index: int, wrist: tuple[float, float]) -> PlayerPose:
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[10] = (wrist[0], wrist[1], 0.9)  # coco-17 right_wrist
    joints[9] = (wrist[0] + 20.0, wrist[1], 0.2)  # left_wrist, farther and weaker
    return PlayerPose(
        object_id="player_0",
        frame_index=frame_index,
        keypoints=joints,
        schema_name="coco-17",
    )


def _racket_object(*, with_keypoints: bool = False) -> ObservedObject:
    keypoints = np.zeros((17, 3), dtype=np.float32) if with_keypoints else None
    return ObservedObject(
        object_id="racket_0",
        object_class="racket",
        frame_index=0,
        bbox=(10.0, 8.0, 30.0, 32.0),
        mask=_racket_mask(),
        keypoints=keypoints,
    )


def _ball_object() -> ObservedObject:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[20:26, 18:24] = 255
    return ObservedObject(
        object_id="ball_0",
        object_class="ball",
        frame_index=0,
        bbox=(18.0, 20.0, 24.0, 26.0),
        mask=mask,
    )


def _difference_frame() -> tuple[np.ndarray, np.ndarray]:
    plate = np.full((40, 40, 3), 40, dtype=np.uint8)
    frame = plate.copy()
    frame[20:26, 18:24] = 220
    return frame, plate


class TestTennisBackendIsRegistered:
    def test_default_rigid_backend_validates(self) -> None:
        loaded = config.default()
        assert loaded.rigid.backend == "tennis"
        validate_backends(loaded, registries=_REGISTRIES)
        assert "tennis" in RIGID
        built = RIGID.build("tennis")
        assert built.name == "tennis"  # type: ignore[attr-defined]

    def test_class_strategies_are_switchable(self) -> None:
        for name in ("racket-hull", "ball-difference", "ball-segmentation", "none"):
            assert name in RIGID
            RIGID.build(name)


class TestRigidOffChangesThePayload:
    def test_lattice_off_zeros_the_measured_payload(self) -> None:
        frame, plate = _difference_frame()
        objects = [_racket_object(), _ball_object()]
        poses = [_player_pose(0, (20.0, 30.0))]

        on_cfg = config.default()
        off_cfg = PointstreamConfig(lattice=LatticeConfig(rigid_objects=False))
        validate_backends(on_cfg, registries=_REGISTRIES)
        validate_backends(off_cfg, registries=_REGISTRIES)

        on = bind(on_cfg).extract(
            objects, player_poses=poses, frames=frame[np.newaxis, ...], background_plate=plate
        )
        off = bind(off_cfg).extract(
            objects, player_poses=poses, frames=frame[np.newaxis, ...], background_plate=plate
        )

        assert on.cost().byte_count > 0
        assert off.cost().byte_count == 0
        assert off.payload != on.payload
        assert on.artifact_counts.get("racket", 0) >= 1
        assert off.artifact_counts == {}
        assert off.deferred_to_residual == frozenset({"racket", "ball"})
        assert on.deferred_to_residual == frozenset()

    def test_turning_one_class_off_changes_artifacts(self) -> None:
        objects = [_racket_object(), _ball_object()]
        poses = [_player_pose(0, (20.0, 30.0))]
        both = RIGID.build("tennis", racket="hull", ball="segmentation")
        racket_only = RIGID.build("tennis", racket="hull", ball="none")
        both_payload = both.extract(objects, player_poses=poses)  # type: ignore[union-attr]
        racket_payload = racket_only.extract(objects, player_poses=poses)  # type: ignore[union-attr]
        assert both_payload.artifact_counts.get("ball", 0) == 1
        assert racket_payload.artifact_counts.get("ball", 0) == 0
        assert "ball" in racket_payload.deferred_to_residual
        assert both_payload.cost().byte_count != racket_payload.cost().byte_count


class TestRacketIsAHullNotAPose:
    def test_hull_is_anchored_to_the_player_wrist(self) -> None:
        wrist = (20.0, 34.0)
        payload = RIGID.build("racket-hull").extract(  # type: ignore[union-attr]
            [_racket_object()],
            player_poses=[_player_pose(0, wrist)],
        )
        assert len(payload.shapes) == 1
        shape = payload.shapes[0]
        assert shape.kind == "hull"
        assert shape.object_class == "racket"
        assert not hasattr(shape, "keypoints")
        assert shape.wrist_anchor is not None
        assert abs(shape.wrist_anchor[0] - wrist[0]) < 1e-6
        assert abs(shape.wrist_anchor[1] - wrist[1]) < 1e-6
        handle = min(
            shape.points, key=lambda point: (point[0] - wrist[0]) ** 2 + (point[1] - wrist[1]) ** 2
        )
        assert abs(handle[0] - wrist[0]) < 1e-5
        assert abs(handle[1] - wrist[1]) < 1e-5

    def test_keypoints_on_a_racket_are_rejected(self) -> None:
        with pytest.raises(ConfigValueError, match="no skeleton"):
            RIGID.build("tennis").extract([_racket_object(with_keypoints=True)])  # type: ignore[union-attr]

    def test_keypoints_on_a_ball_are_rejected(self) -> None:
        ball = ObservedObject(
            object_id="ball_0",
            object_class="ball",
            frame_index=0,
            bbox=(18.0, 20.0, 24.0, 26.0),
            keypoints=np.zeros((17, 3), dtype=np.float32),
        )
        with pytest.raises(ConfigValueError, match="no skeleton"):
            RIGID.build("ball-segmentation").extract([ball])  # type: ignore[union-attr]


class TestBallStrategies:
    def test_difference_finds_a_blob_the_plate_does_not_have(self) -> None:
        frame, plate = _difference_frame()
        payload = RIGID.build("ball-difference").extract(  # type: ignore[union-attr]
            [],
            frames=frame[np.newaxis, ...],
            background_plate=plate,
        )
        assert len(payload.shapes) == 1
        shape = payload.shapes[0]
        assert shape.kind == "difference"
        assert shape.object_class == "ball"
        cx, cy = shape.points[0]
        assert 17 <= cx <= 25
        assert 19 <= cy <= 27

    def test_segmentation_uses_the_mask_not_a_pose(self) -> None:
        payload = RIGID.build("ball-segmentation").extract([_ball_object()])  # type: ignore[union-attr]
        assert len(payload.shapes) == 1
        assert payload.shapes[0].kind == "segmentation"
        assert payload.shapes[0].wrist_anchor is None


class TestUnknownRigidBackend:
    def test_validate_backends_rejects_an_unregistered_name(self) -> None:
        loaded = config.default()
        broken = loaded.with_(rigid=BackendConfig(backend="heuristic-skeleton"))
        with pytest.raises(ConfigError):
            validate_backends(broken, registries=_REGISTRIES)
        with pytest.raises(UnknownBackendError, match="rigid"):
            RIGID.spec("heuristic-skeleton")
