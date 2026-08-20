"""Behaviour and plausible misuse of the domain profile contract."""

import pytest

from src.contracts.capabilities import (
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_VECTORS,
)
from src.contracts.domain import (
    ALL_BACKGROUND_METHODS,
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_DELTA,
    BACKGROUND_PANORAMA_FULL,
    GENERAL,
    PROFILES,
    TENNIS,
    CameraMotion,
    DomainProfile,
    SalientClass,
    describe_profiles,
    profile,
)
from src.contracts.errors import ConfigValueError, UnknownBackendError
from src.contracts.keypoints import CANONICAL_HUMAN


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_domain_tennis_resolves():
    assert profile("tennis") is TENNIS
    assert profile("general") is GENERAL


def test_an_unregistered_domain_names_the_ones_that_exist():
    with pytest.raises(UnknownBackendError) as excinfo:
        profile("football")
    message = str(excinfo.value)
    assert "tennis" in message and "general" in message


def test_football_is_not_half_registered():
    """A deferred third profile present in the table would read as supported."""
    assert set(PROFILES) == {"tennis", "general"}


# --------------------------------------------------------------------------
# Camera motion — the load-bearing field
# --------------------------------------------------------------------------


def test_broadcast_tennis_is_planar_so_a_panorama_holds():
    assert TENNIS.camera_motion is CameraMotion.PAN_TILT_ZOOM
    assert TENNIS.supports_panorama
    TENNIS.assert_background_valid(BACKGROUND_PANORAMA_FULL)
    TENNIS.assert_background_valid(BACKGROUND_PANORAMA_DELTA)


def test_davis_handheld_footage_has_parallax_so_it_does_not():
    assert GENERAL.camera_motion is CameraMotion.FREE_MOVING
    assert not GENERAL.supports_panorama


@pytest.mark.parametrize(
    "method", [BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_DELTA]
)
def test_a_panorama_under_a_parallax_assumption_is_a_validation_error(method):
    """It would not fail loudly at runtime — it would produce a plausible-looking
    background that is quietly wrong, which is the worst available outcome."""
    with pytest.raises(ConfigValueError) as excinfo:
        GENERAL.assert_background_valid(method)
    message = str(excinfo.value)
    assert "parallax" in message
    assert BACKGROUND_NONE in message


def test_disabling_the_background_is_always_valid():
    for item in PROFILES.values():
        item.assert_background_valid(BACKGROUND_NONE)


def test_an_unknown_background_method_names_the_known_ones():
    with pytest.raises(ConfigValueError) as excinfo:
        TENNIS.assert_background_valid("panorama-fullish")
    assert all(name in str(excinfo.value) for name in ALL_BACKGROUND_METHODS)


def test_a_static_camera_is_planar_too():
    assert CameraMotion.STATIC.is_planar
    assert CameraMotion.STATIC.supports_panorama
    assert not CameraMotion.FREE_MOVING.is_planar


# --------------------------------------------------------------------------
# Salient classes and their schemas
# --------------------------------------------------------------------------


def test_tennis_declares_players_racket_and_ball():
    assert TENNIS.class_names == ("player", "racket", "ball")


def test_players_carry_the_canonical_human_schema():
    assert TENNIS.schema_for("player") is CANONICAL_HUMAN
    assert GENERAL.schema_for("person") is CANONICAL_HUMAN


def test_rigid_classes_have_no_schema_and_that_is_an_answer_not_a_gap():
    assert TENNIS.schema_for("racket") is None
    assert TENNIS.schema_for("ball") is None
    assert TENNIS.class_of("racket").rigid


def test_an_unknown_class_names_the_ones_the_domain_declares():
    with pytest.raises(UnknownBackendError) as excinfo:
        TENNIS.class_of("umpire")
    assert "player" in str(excinfo.value)


def test_a_rigid_class_carrying_a_skeleton_is_a_contradiction():
    with pytest.raises(ValueError, match="rigid"):
        SalientClass(name="racket", keypoint_schema=CANONICAL_HUMAN, rigid=True)


# --------------------------------------------------------------------------
# Motion representations against object classes
# --------------------------------------------------------------------------


def test_keypoints_on_a_class_with_no_skeleton_are_rejected():
    """The pose estimator would return nothing for a racket, so the stream would
    carry an empty motion representation — a silent quality loss."""
    with pytest.raises(ConfigValueError) as excinfo:
        TENNIS.assert_motion_supported("racket", MOTION_KEYPOINTS)
    message = str(excinfo.value)
    assert "no skeleton" in message
    assert MOTION_VECTORS in message and MOTION_ENCODED_VIDEO in message


def test_class_agnostic_representations_apply_to_everything():
    """This is what makes skeleton-less objects a configuration rather than a
    separate design."""
    for name in TENNIS.class_names:
        TENNIS.assert_motion_supported(name, MOTION_VECTORS)
        TENNIS.assert_motion_supported(name, MOTION_ENCODED_VIDEO)


def test_all_three_representations_apply_to_humans():
    """Which is what gives the paper its controlled comparison, on identical
    objects with identical appearance."""
    supported = TENNIS.class_of("player").supported_motion()
    assert supported == {MOTION_KEYPOINTS, MOTION_VECTORS, MOTION_ENCODED_VIDEO}


def test_a_motion_kind_outside_the_vocabulary_is_named_as_such():
    with pytest.raises(ConfigValueError, match="not a motion representation"):
        TENNIS.assert_motion_supported("player", "optical-flow")


# --------------------------------------------------------------------------
# Scene classification
# --------------------------------------------------------------------------


def test_tennis_routes_points_against_interludes():
    assert TENNIS.uses_scene_classification
    assert TENNIS.scene_classes == ("point", "interlude")


def test_short_davis_clips_have_nothing_to_route():
    assert not GENERAL.uses_scene_classification


# --------------------------------------------------------------------------
# Profile construction
# --------------------------------------------------------------------------


def test_a_profile_with_no_salient_classes_is_rejected():
    """That is the all-off lattice corner, which is a configuration rather than
    a domain."""
    with pytest.raises(ValueError, match="no salient classes"):
        DomainProfile(
            name="empty", salient_classes=(), camera_motion=CameraMotion.STATIC
        )


def test_a_profile_repeating_a_class_is_rejected():
    with pytest.raises(ValueError, match="repeats"):
        DomainProfile(
            name="dup",
            salient_classes=(SalientClass("person"), SalientClass("person")),
            camera_motion=CameraMotion.STATIC,
        )


def test_open_vocabulary_prompts_are_the_domain_words_not_the_class_keys():
    assert TENNIS.detection_prompts() == ("tennis player", "tennis racket", "tennis ball")


def test_describe_states_whether_a_panorama_is_valid():
    assert "panorama background valid" in TENNIS.describe()
    assert "panorama background INVALID" in GENERAL.describe()
    assert "tennis" in describe_profiles()
