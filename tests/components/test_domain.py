"""Component-layer domain profiles: resolution, wiring, and the traps they exist to catch."""

from __future__ import annotations

import pytest

from src.components.domain import REGISTRY as DOMAINS
from src.components.domain.profiles import DomainBackend
from src.contracts.config import default, load, parse, validate_backends
from src.contracts.domain import BACKGROUND_NONE, BACKGROUND_PANORAMA_FULL, GENERAL, TENNIS, profile
from src.contracts.errors import ConfigError, ConfigValueError, UnknownBackendError

_DETECTOR_BACKENDS = (
    "yolo",
    "yolo26",
    "sam3",
    "sam2",
    "rf-detr",
    "rfdetr",
    "grounding-dino",
)


@pytest.mark.parametrize("name", ["tennis", "general"])
def test_profiles_resolve_by_name_and_round_trip(name: str) -> None:
    spec = DOMAINS.spec(name)
    built = DOMAINS.build(name)
    assert spec.name == name
    assert built.name == name
    assert built.profile is profile(name)
    assert profile(built.name) is built.profile
    again = DOMAINS.build(built.name)
    assert again.profile is built.profile
    assert again.selector == spec.defaults["selector"]


def test_football_is_not_registered() -> None:
    """A half-built third profile would be read as a supported one."""
    assert "football" not in DOMAINS
    assert set(DOMAINS.names()) == {"general", "tennis"}
    with pytest.raises(UnknownBackendError) as excinfo:
        DOMAINS.spec("football")
    message = str(excinfo.value)
    assert "tennis" in message and "general" in message


def test_a_panorama_under_general_is_rejected_with_a_usable_message() -> None:
    """The component must surface the contract check, not swallow it.

    A panorama under a free-moving camera is quietly incoherent, not slightly
    worse — the message has to say that, and name the residual fallback.
    """
    general = DOMAINS.build("general")
    with pytest.raises(ConfigValueError) as excinfo:
        general.assert_background_valid(BACKGROUND_PANORAMA_FULL)
    message = str(excinfo.value)
    assert "parallax" in message
    assert BACKGROUND_NONE in message
    assert general.profile is GENERAL


def test_tennis_still_accepts_a_panorama() -> None:
    tennis = DOMAINS.build("tennis")
    tennis.assert_background_valid(BACKGROUND_PANORAMA_FULL)
    assert tennis.profile is TENNIS


def test_a_profile_does_not_name_a_detector_backend() -> None:
    """The domain says what is salient; detection backends say how it is found."""
    for spec in DOMAINS:
        blob = " ".join(
            [
                spec.name,
                spec.target,
                spec.summary,
                *spec.aliases,
                *spec.capabilities,
                *(str(value) for value in spec.defaults.values()),
            ]
        ).lower()
        for detector in _DETECTOR_BACKENDS:
            assert detector not in blob, f"{spec.name} names detector {detector!r}"


def test_the_profile_names_a_selector_it_does_not_encode_the_rule() -> None:
    tennis = DOMAINS.build("tennis")
    general = DOMAINS.build("general")
    assert tennis.selector == "heuristic"
    assert general.selector == "identity"
    assert not hasattr(tennis, "select_players")


def test_a_backend_without_a_selector_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="selector"):
        DomainBackend(profile=TENNIS, selector="")


def test_validate_backends_accepts_the_registered_domains() -> None:
    validate_backends(default(), registries={"domain": DOMAINS})
    general = load({"domain": "general", "background": {"method": BACKGROUND_NONE}})
    validate_backends(general, registries={"domain": DOMAINS})


def test_validate_backends_rejects_an_unregistered_domain() -> None:
    with pytest.raises(ConfigError, match="football"):
        validate_backends(parse({"domain": "football"}), registries={"domain": DOMAINS})
