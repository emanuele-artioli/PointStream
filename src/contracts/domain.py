"""What is being modelled, as distinct from which component models it.

A `DomainProfile` declares **semantics**: which object classes are salient, what
skeleton applies to each where one applies at all, what the camera is assumed to
be doing, and what scene classification means here. Components are
interchangeable implementations that satisfy those needs — in any human domain we
care about people, but whether YOLO26, SAM3 or RF-DETR finds them is a separate
axis. The domain says *what*; components say *how*; config picks both separately.

**The camera-motion assumption is the load-bearing field**, and it is not about
camera hardware. It is about whether the background can be modelled as a single
warpable plane. A tennis broadcast camera is near-static with pan, tilt and zoom,
so successive frames relate by a homography and a panorama background is valid.
Freely-moving handheld footage has parallax, so no single homography exists and a
panorama background is *invalid* — it will produce garbage, and it will do so
quietly, which is the dangerous part. This matters immediately rather than in
principle: the DAVIS clips the general profile is evaluated on are largely
handheld. So the profile declares its regime and `assert_background_valid`
rejects a panorama request under parallax at validation time, instead of letting
the run finish and the numbers look merely disappointing.

The second thing a profile prevents is a motion representation an object class
cannot carry. Keypoints require a stable skeleton; a racket and a ball do not
have one, and asking for `motion: keypoints` on them is a config error rather
than something to discover when the pose estimator returns nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.contracts.capabilities import (
    ALL_MOTION,
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_SPARSE_TRAJECTORIES,
)
from src.contracts.errors import ConfigValueError, UnknownBackendError
from src.contracts.keypoints import CANONICAL_HUMAN, KeypointSchema

# --------------------------------------------------------------------------
# Camera motion
# --------------------------------------------------------------------------


class CameraMotion(str, Enum):
    """Whether the background can be modelled as one warpable plane.

    The three regimes differ in exactly one way that matters downstream: whether
    a single homography relates successive frames. That is the precondition for
    every panorama-based background method, and nothing else in the system
    checks it.
    """

    STATIC = "static"
    """The camera does not move. Frames relate by the identity transform, so a
    single background plate covers the whole clip."""

    PAN_TILT_ZOOM = "pan-tilt-zoom"
    """The camera rotates about its optical centre and zooms, but does not
    translate. Frames relate by a homography, so a panorama is valid and a
    frame's background is a warp of it. This is broadcast tennis."""

    FREE_MOVING = "free-moving"
    """The camera translates, so the scene shows parallax and no single
    homography relates the frames. A panorama built anyway will be internally
    inconsistent — near and far content cannot both align — and the result is
    plausible-looking garbage rather than an obvious failure. Handheld footage,
    which is most of DAVIS."""

    @property
    def is_planar(self) -> bool:
        """Whether one homography per frame pair is a sound model."""
        return self in (CameraMotion.STATIC, CameraMotion.PAN_TILT_ZOOM)

    @property
    def supports_panorama(self) -> bool:
        """Whether a panorama background model is valid under this assumption."""
        return self.is_planar


# --------------------------------------------------------------------------
# Background methods, and which of them need a plane
# --------------------------------------------------------------------------

#: A full panorama transmitted once and warped per frame.
BACKGROUND_PANORAMA_FULL: Final = "panorama-full"

#: A panorama plus per-frame deltas where the plate has gone stale.
BACKGROUND_PANORAMA_DELTA: Final = "panorama-delta"

#: No background model at all — the background lands in the residual.
BACKGROUND_NONE: Final = "none"

ALL_BACKGROUND_METHODS: Final[frozenset[str]] = frozenset(
    {BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_DELTA, BACKGROUND_NONE}
)

#: The methods that assume one warpable plane, and are therefore invalid under a
#: parallax camera assumption.
PANORAMA_METHODS: Final[frozenset[str]] = frozenset(
    {BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_DELTA}
)


# --------------------------------------------------------------------------
# Salient classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SalientClass:
    """One kind of object the domain considers worth modelling separately.

    Args:
        name: The domain's name for it, as used in config and in payloads.
        keypoint_schema: The schema that applies, or `None` for classes with no
            skeleton. `None` is a real answer, not a missing one — a racket has
            no joints, and pretending otherwise is how a zero-filled pose vector
            ends up rendering a limb at the origin.
        rigid: Whether the object is better served by a shape strategy —
            convex hull, anchoring — than by a deformable model.
        prompt: Open-vocabulary phrase for detectors that accept one, where it
            differs from `name`.
        summary: One line, for listings.
    """

    name: str
    keypoint_schema: KeypointSchema | None = None
    rigid: bool = False
    prompt: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SalientClass needs a name.")
        if self.rigid and self.keypoint_schema is not None:
            raise ValueError(
                f"SalientClass {self.name!r} is declared rigid but carries keypoint "
                f"schema {self.keypoint_schema.name!r}. A rigid object has no joints to "
                f"track; if it genuinely has a skeleton it is not rigid."
            )

    @property
    def has_skeleton(self) -> bool:
        """Whether keypoints are a usable motion representation for this class."""
        return self.keypoint_schema is not None

    @property
    def detection_prompt(self) -> str:
        """What to ask an open-vocabulary detector for."""
        return self.prompt or self.name

    def supported_motion(self) -> frozenset[str]:
        """Motion representations this class can actually carry.

        Every class can carry trajectories or encoded video — those are
        class-agnostic by construction, which is the point of having them.
        Keypoints need a skeleton.
        """
        base = {MOTION_SPARSE_TRAJECTORIES, MOTION_ENCODED_VIDEO}
        if self.has_skeleton:
            base.add(MOTION_KEYPOINTS)
        return frozenset(base)


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainProfile:
    """The semantics of one task domain.

    Args:
        name: Config key — ``domain: tennis`` resolves to this.
        salient_classes: What is worth modelling separately here, in priority
            order.
        camera_motion: The assumption every background method is checked
            against.
        scene_classes: What scene classification means in this domain. Empty
            means classification does not apply and the whole input is one span,
            which is also the row-1-off behaviour.
        summary: One line, for listings.
    """

    name: str
    salient_classes: tuple[SalientClass, ...]
    camera_motion: CameraMotion
    scene_classes: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.salient_classes:
            raise ValueError(
                f"DomainProfile {self.name!r} declares no salient classes, so nothing "
                f"would ever be modelled semantically. Use the all-off lattice corner "
                f"for that instead — it is a configuration, not a domain."
            )
        seen = [item.name for item in self.salient_classes]
        if len(set(seen)) != len(seen):
            duplicates = sorted({name for name in seen if seen.count(name) > 1})
            raise ValueError(
                f"DomainProfile {self.name!r} repeats salient class(es): {duplicates}."
            )

    @property
    def class_names(self) -> tuple[str, ...]:
        """Salient class names, in declaration order."""
        return tuple(item.name for item in self.salient_classes)

    @property
    def uses_scene_classification(self) -> bool:
        """Whether routing by scene class means anything in this domain."""
        return bool(self.scene_classes)

    @property
    def supports_panorama(self) -> bool:
        """Whether a panorama background is sound under this camera assumption."""
        return self.camera_motion.supports_panorama

    def class_of(self, name: str) -> SalientClass:
        """Look up a salient class.

        Raises:
            UnknownBackendError: With the classes this domain does declare.
        """
        for item in self.salient_classes:
            if item.name == name:
                return item
        raise UnknownBackendError(f"{self.name} salient class", name, self.class_names)

    def schema_for(self, name: str) -> KeypointSchema | None:
        """The keypoint schema for a salient class, or `None` if it has no skeleton."""
        return self.class_of(name).keypoint_schema

    def detection_prompts(self) -> tuple[str, ...]:
        """What to ask an open-vocabulary detector for, one phrase per class."""
        return tuple(item.detection_prompt for item in self.salient_classes)

    def assert_motion_supported(
        self, object_class: str, motion_kind: str, *, path: str = "object-stream.motion"
    ) -> None:
        """Raise unless `object_class` can carry `motion_kind`.

        The case that motivates this: keypoints on a racket or a ball. Those
        classes have no skeleton, so a pose estimator returns nothing for them
        and the stream carries an empty motion representation the decoder cannot
        use — a silent quality loss rather than a failure.

        Raises:
            ConfigValueError: Naming what the class can carry instead.
        """
        if motion_kind not in ALL_MOTION:
            raise ConfigValueError(
                path,
                f"{motion_kind!r} is not a motion representation. "
                f"Known: {', '.join(sorted(ALL_MOTION))}.",
            )
        salient = self.class_of(object_class)
        supported = salient.supported_motion()
        if motion_kind not in supported:
            raise ConfigValueError(
                path,
                f"class {object_class!r} in domain {self.name!r} cannot carry motion "
                f"representation {motion_kind!r}: it has no keypoint schema, so there is "
                f"no skeleton to send per frame. It can carry: "
                f"{', '.join(sorted(supported))}.",
            )

    def assert_background_valid(
        self, method: str, *, path: str = "background.method"
    ) -> None:
        """Raise unless `method` is sound under this domain's camera assumption.

        Raises:
            ConfigValueError: For an unknown method, or for a panorama method
                under a parallax assumption.
        """
        if method not in ALL_BACKGROUND_METHODS:
            raise ConfigValueError(
                path,
                f"{method!r} is not a background method. "
                f"Known: {', '.join(sorted(ALL_BACKGROUND_METHODS))}.",
            )
        if method in PANORAMA_METHODS and not self.supports_panorama:
            raise ConfigValueError(
                path,
                f"background method {method!r} models the background as one warpable "
                f"plane, but domain {self.name!r} assumes a "
                f"{self.camera_motion.value} camera, which shows parallax. No single "
                f"homography relates the frames, so the panorama cannot be internally "
                f"consistent and the reconstruction will be quietly wrong rather than "
                f"obviously broken. Use {BACKGROUND_NONE!r} and let the residual carry "
                f"the background.",
            )

    def describe(self) -> str:
        """A readable summary, for `list-domains`-style output."""
        width = max(len(item.name) for item in self.salient_classes)
        lines = [
            f"domain {self.name}: {self.summary}",
            f"  camera: {self.camera_motion.value} "
            f"(panorama background {'valid' if self.supports_panorama else 'INVALID'})",
            f"  scenes: {', '.join(self.scene_classes) or 'not classified; one span'}",
            "  salient classes:",
        ]
        for item in self.salient_classes:
            schema_name = item.keypoint_schema.name if item.keypoint_schema else "no skeleton"
            rigid = " rigid" if item.rigid else ""
            lines.append(
                f"    {item.name.ljust(width)}  {schema_name:<20}{rigid}  {item.summary}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The registered profiles
# --------------------------------------------------------------------------

TENNIS = DomainProfile(
    name="tennis",
    salient_classes=(
        SalientClass(
            name="player",
            keypoint_schema=CANONICAL_HUMAN,
            prompt="tennis player",
            summary="The two competitors. Ball kids and crowd are not salient.",
        ),
        SalientClass(
            name="racket",
            rigid=True,
            prompt="tennis racket",
            summary="Rigid, anchored to a wrist; convex hull rather than a skeleton.",
        ),
        SalientClass(
            name="ball",
            rigid=True,
            prompt="tennis ball",
            summary="Small, fast, and mostly recovered by difference or segmentation.",
        ),
    ),
    camera_motion=CameraMotion.PAN_TILT_ZOOM,
    scene_classes=("point", "interlude"),
    summary=(
        "Broadcast tennis. Deliberately constrained: near-static camera, known "
        "background, few actors."
    ),
)

GENERAL = DomainProfile(
    name="general",
    salient_classes=(
        SalientClass(
            name="person",
            keypoint_schema=CANONICAL_HUMAN,
            prompt="person",
            summary="Whichever humans are present, with no domain-specific selection.",
        ),
    ),
    camera_motion=CameraMotion.FREE_MOVING,
    scene_classes=(),
    summary=(
        "General human video, evaluated on the DAVIS clips containing people. "
        "Largely handheld, so parallax rules out panorama backgrounds; clips are "
        "short, so scene routing has nothing to route."
    ),
)

#: Every profile, by config name. Football is deliberately absent — it is a
#: later decision, and a half-built third profile would be read as a supported
#: one.
PROFILES: Final[Mapping[str, DomainProfile]] = {
    profile.name: profile for profile in (TENNIS, GENERAL)
}


def profile(name: str) -> DomainProfile:
    """Look up a domain profile by config name.

    Raises:
        UnknownBackendError: With the registered profiles and a close-match
            suggestion.
    """
    try:
        return PROFILES[name]
    except KeyError:
        raise UnknownBackendError("domain", name, sorted(PROFILES)) from None


def describe_profiles(names: Iterable[str] | None = None) -> str:
    """A readable block for every registered profile."""
    chosen = [profile(name) for name in names] if names else list(PROFILES.values())
    return "\n".join(item.describe() for item in chosen)
