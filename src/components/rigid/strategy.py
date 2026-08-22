"""Per-class rigid strategies, plus the combined tennis backend.

Config default is ``rigid.backend="tennis"``. The two class strategies are
constructor kwargs (``racket=``, ``ball=``) because ``BackendConfig`` has no
per-class fields — a contract change we are not making in this stream.

Lattice off (``lattice.rigid_objects=False``) binds ``none``: empty payload,
both classes deferred to the residual. That is the measurable ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.components.rigid.ball import extract_ball_difference, extract_ball_segmentation
from src.components.rigid.racket import extract_racket, reject_keypoints
from src.components.rigid.types import (
    ObservedObject,
    PlayerPose,
    RigidPayload,
    RigidShape,
    payload_from_shapes,
)
from src.contracts.config import PointstreamConfig
from src.contracts.errors import ConfigValueError

RACKET_HULL = "hull"
RACKET_NONE = "none"
BALL_DIFFERENCE = "difference"
BALL_SEGMENTATION = "segmentation"
BALL_NONE = "none"

ALL_RACKET = frozenset({RACKET_HULL, RACKET_NONE})
ALL_BALL = frozenset({BALL_DIFFERENCE, BALL_SEGMENTATION, BALL_NONE})
TENNIS_RIGID_CLASSES = frozenset({"racket", "ball"})


class TennisRigid:
    """Racket hull plus a ball strategy. Either class can be switched off."""

    name = "tennis"

    def __init__(
        self,
        racket: str = RACKET_HULL,
        ball: str = BALL_DIFFERENCE,
        difference_threshold: float = 18.0,
        min_blob_area: int = 6,
    ) -> None:
        if racket not in ALL_RACKET:
            raise ConfigValueError(
                "rigid.racket",
                f"{racket!r} is not a racket strategy. Known: {', '.join(sorted(ALL_RACKET))}.",
            )
        if ball not in ALL_BALL:
            raise ConfigValueError(
                "rigid.ball",
                f"{ball!r} is not a ball strategy. Known: {', '.join(sorted(ALL_BALL))}.",
            )
        self.racket = racket
        self.ball = ball
        self._difference_threshold = float(difference_threshold)
        self._min_blob_area = int(min_blob_area)

    @property
    def enabled_classes(self) -> frozenset[str]:
        enabled: set[str] = set()
        if self.racket != RACKET_NONE:
            enabled.add("racket")
        if self.ball != BALL_NONE:
            enabled.add("ball")
        return frozenset(enabled)

    @property
    def deferred_to_residual(self) -> frozenset[str]:
        return TENNIS_RIGID_CLASSES - self.enabled_classes

    def extract(
        self,
        objects: Sequence[ObservedObject],
        *,
        player_poses: Sequence[PlayerPose] = (),
        frames: np.ndarray | None = None,
        background_plate: np.ndarray | None = None,
    ) -> RigidPayload:
        for obj in objects:
            if obj.object_class in TENNIS_RIGID_CLASSES:
                reject_keypoints(obj)

        shapes: list[RigidShape] = []
        if self.racket == RACKET_HULL:
            for obj in objects:
                if obj.object_class != "racket":
                    continue
                shape = extract_racket(obj, player_poses)
                if shape is not None:
                    shapes.append(shape)

        if self.ball == BALL_SEGMENTATION:
            for obj in objects:
                if obj.object_class != "ball":
                    continue
                shape = extract_ball_segmentation(obj)
                if shape is not None:
                    shapes.append(shape)
        elif self.ball == BALL_DIFFERENCE and frames is not None:
            stack = np.asarray(frames)
            n_frames = int(stack.shape[0]) if stack.ndim == 4 else 1
            if stack.ndim == 3:
                stack = stack[np.newaxis, ...]
                n_frames = 1
            for index in range(n_frames):
                shape = extract_ball_difference(
                    stack[index],
                    background_plate,
                    objects,
                    frame_index=index,
                    threshold=self._difference_threshold,
                    min_area=self._min_blob_area,
                )
                if shape is not None:
                    shapes.append(shape)

        return payload_from_shapes(
            shapes,
            enabled_classes=self.enabled_classes,
            deferred_to_residual=self.deferred_to_residual,
            backend=self.name,
        )


class RacketHull(TennisRigid):
    name = "racket-hull"

    def __init__(self) -> None:
        super().__init__(racket=RACKET_HULL, ball=BALL_NONE)


class BallDifference(TennisRigid):
    name = "ball-difference"

    def __init__(self) -> None:
        super().__init__(racket=RACKET_NONE, ball=BALL_DIFFERENCE)


class BallSegmentation(TennisRigid):
    name = "ball-segmentation"

    def __init__(self) -> None:
        super().__init__(racket=RACKET_NONE, ball=BALL_SEGMENTATION)


class RigidNone:
    name = "none"

    def extract(
        self,
        objects: Sequence[ObservedObject],
        **_kwargs: Any,
    ) -> RigidPayload:
        for obj in objects:
            if obj.object_class in TENNIS_RIGID_CLASSES:
                reject_keypoints(obj)
        return payload_from_shapes(
            (),
            enabled_classes=frozenset(),
            deferred_to_residual=TENNIS_RIGID_CLASSES,
            backend=self.name,
        )


def bind(config: PointstreamConfig, **overrides: Any) -> TennisRigid | RigidNone:
    """Construct the configured rigid backend.

    Lattice off binds ``none`` so the payload change is the ablation, not a
    still-running extractor whose output is discarded.
    """
    from src.components.rigid import REGISTRY

    name = config.rigid.backend if config.lattice.rigid_objects else "none"
    built = REGISTRY.build(name, **overrides)
    if not isinstance(built, (TennisRigid, RigidNone)):
        raise TypeError(f"rigid backend {name!r} did not construct a rigid strategy")
    return built
