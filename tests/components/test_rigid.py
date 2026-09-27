"""Rigid objects are optional lattice rows with no skeleton."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background import REGISTRY as BACKGROUND
from src.components.rigid import REGISTRY as RIGID
from src.components.rigid.strategy import TennisRigid, bind
from src.components.rigid.types import ObservedObject, PlayerPose
from src.contracts import config
from src.contracts.config import BackendConfig, LatticeConfig, PointstreamConfig, validate_backends
from src.contracts.errors import ConfigError, ConfigValueError, UnknownBackendError

_REGISTRIES = {"background": BACKGROUND, "rigid": RIGID}


def _backend(name: str, **kwargs: object) -> TennisRigid:
    built = RIGID.build(name, **kwargs)
    assert isinstance(built, TennisRigid)
    return built


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
        built = _backend("tennis")
        assert built.name == "tennis"

    def test_class_strategies_are_switchable(self) -> None:
        for name in ("racket-hull", "racket-cross", "ball-difference", "ball-segmentation", "none"):
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

        on_bytes = on.cost().byte_count
        off_bytes = off.cost().byte_count
        assert on_bytes is not None and on_bytes > 0
        assert off_bytes == 0
        assert off.payload != on.payload
        assert on.artifact_counts.get("racket", 0) >= 1
        assert off.artifact_counts == {}
        assert off.deferred_to_residual == frozenset({"racket", "ball"})
        assert on.deferred_to_residual == frozenset()

    def test_turning_one_class_off_changes_artifacts(self) -> None:
        objects = [_racket_object(), _ball_object()]
        poses = [_player_pose(0, (20.0, 30.0))]
        both = _backend("tennis", racket="hull", ball="segmentation")
        racket_only = _backend("tennis", racket="hull", ball="none")
        both_payload = both.extract(objects, player_poses=poses)
        racket_payload = racket_only.extract(objects, player_poses=poses)
        assert both_payload.artifact_counts.get("ball", 0) == 1
        assert racket_payload.artifact_counts.get("ball", 0) == 0
        assert "ball" in racket_payload.deferred_to_residual
        assert both_payload.cost().byte_count != racket_payload.cost().byte_count


class TestRacketIsAHullNotAPose:
    def test_hull_preserves_observed_mask_and_records_wrist_separately(self) -> None:
        wrist = (20.0, 34.0)
        payload = _backend("racket-hull").extract(
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
        assert min(x for x, _ in shape.points) == 10
        assert max(x for x, _ in shape.points) == 29
        assert min(y for _, y in shape.points) == 8
        assert max(y for _, y in shape.points) == 31
        assert shape.associated_player_id == "player_0"
        assert shape.associated_wrist == "right_wrist"

    def test_cross_intersects_continuous_polygon_and_orders_width_endpoints(self) -> None:
        from src.components.rigid.racket import extract_racket_cross

        result = extract_racket_cross(_racket_object(), [_player_pose(0, (20.0, 30.0))])
        assert result is not None
        assert result.kind == "racket_cross_v1"
        assert len(result.points) == 4
        assert result.points[0] == (20.0, 30.0)
        assert result.endpoint_order == "directed_axis_perpendicular_negative_then_positive"
        assert result.associated_player_id == "player_0"

    def test_polygon_width_is_independent_of_contour_vertex_sampling(self) -> None:
        from src.components.rigid.racket import _line_polygon_intersections

        polygon = np.asarray([[0.0, 0.0], [8.0, 0.0], [8.0, 20.0], [0.0, 20.0]])
        dense = np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0], [8.0, 20.0], [0.0, 20.0]]
        )
        center = np.array([4.0, 10.0])
        direction = np.array([1.0, 0.0])
        ordinary = _line_polygon_intersections(polygon, center, direction)
        expanded = _line_polygon_intersections(dense, center, direction)
        assert sorted(tuple(point) for point in ordinary) == sorted(tuple(point) for point in expanded)

    def test_cross_endpoints_are_stable_across_mask_boundary_sampling(self) -> None:
        from src.components.rigid.racket import extract_racket_cross

        plain = _racket_object()
        dense_mask = np.zeros_like(_racket_mask())
        dense_mask[8:32, 16:24] = 255
        dense_mask[8:16, 10:30] = 255
        dense_mask[9:15, 11:29] = 255
        dense = ObservedObject(
            object_id=plain.object_id,
            object_class=plain.object_class,
            frame_index=plain.frame_index,
            bbox=plain.bbox,
            mask=dense_mask,
        )
        pose = [_player_pose(0, (20.0, 34.0))]
        first = extract_racket_cross(plain, pose)
        second = extract_racket_cross(dense, pose)
        assert first is not None and second is not None
        assert first.kind == second.kind == "racket_cross_v1"
        np.testing.assert_allclose(first.points, second.points, atol=1.0)

    def test_missing_wrist_produces_a_separately_typed_hull_fallback(self) -> None:
        from src.components.rigid.racket import extract_racket_cross

        result = extract_racket_cross(_racket_object(), [])
        assert result is not None
        assert result.kind == "hull_fallback_v1"
        assert result.fallback_reason == "missing_visible_wrist"

    def test_keypoints_on_a_racket_are_rejected(self) -> None:
        with pytest.raises(ConfigValueError, match="no skeleton"):
            _backend("tennis").extract([_racket_object(with_keypoints=True)])

    def test_keypoints_on_a_ball_are_rejected(self) -> None:
        ball = ObservedObject(
            object_id="ball_0",
            object_class="ball",
            frame_index=0,
            bbox=(18.0, 20.0, 24.0, 26.0),
            keypoints=np.zeros((17, 3), dtype=np.float32),
        )
        with pytest.raises(ConfigValueError, match="no skeleton"):
            _backend("ball-segmentation").extract([ball])


class TestBallStrategies:
    def test_difference_finds_a_blob_the_plate_does_not_have(self) -> None:
        frame, plate = _difference_frame()
        payload = _backend("ball-difference").extract(
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
        payload = _backend("ball-segmentation").extract([_ball_object()])
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
