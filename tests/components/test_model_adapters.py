from types import SimpleNamespace

import numpy as np

from src.components.perception.coordinates import make_crop_transform
from src.components.perception.model_adapters import render_model_adapter_views
from src.components.pose.wire import from_coco17


def _pose():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, :2] = np.array([16.0, 16.0])
    points[:, 2] = 0.9
    # Distinct shoulders make the derived OpenPose neck observable.
    points[5, :2] = (10.0, 8.0)
    points[6, :2] = (22.0, 8.0)
    return from_coco17(points)


def _transform():
    return make_crop_transform(
        (0, 0, 32, 32), source_size=(32, 32), target_size=(32, 32)
    )


def test_model_adapters_project_openpose_and_keep_geometry_channels_separate():
    player = np.zeros((32, 32), dtype=np.uint8)
    player[3:20, 8:24] = 1
    racket = np.zeros_like(player)
    racket[22:29, 13:19] = 1
    cross = SimpleNamespace(
        kind="racket_cross_v1",
        points=((14.0, 26.0), (18.0, 12.0), (12.0, 18.0), (20.0, 20.0)),
    )

    views = render_model_adapter_views(
        view="joint",
        transform=_transform(),
        player_mask=player,
        racket_mask=racket,
        pose=_pose(),
        racket_geometry=cross,
    )

    assert views.animate_anyone_pose_rgb is None
    assert views.eligible_for_cross_training
    assert views.schema == "pointstream.model-adapter-views.v1"
    assert views.controlnet_condition_rgb.shape == (32, 32, 3)
    assert views.controlnet_channels["player_mask"][5, 10] == 255
    assert views.controlnet_channels["racket_mask"][24, 15] == 255
    assert np.count_nonzero(views.controlnet_channels["racket_axis"])
    assert np.count_nonzero(views.controlnet_channels["racket_width"])
    assert np.array_equal(
        views.controlnet_condition_rgb[..., 0], views.controlnet_channels["player_mask"]
    )


def test_fallback_racket_geometry_is_explicitly_ineligible():
    empty = np.zeros((32, 32), dtype=np.uint8)
    fallback = SimpleNamespace(kind="hull_fallback_v1", points=((1, 1), (2, 2)))
    views = render_model_adapter_views(
        view="racket",
        transform=_transform(),
        player_mask=empty,
        racket_mask=empty,
        racket_geometry=fallback,
    )

    assert not views.eligible_for_cross_training
    assert views.exclusion_reason == "missing_or_fallback_racket_cross"
    assert not np.count_nonzero(views.controlnet_channels["racket_axis"])
    assert not np.count_nonzero(views.controlnet_channels["racket_width"])
