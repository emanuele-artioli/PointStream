"""Cross-stream wiring: named backends exist on every axis and join cleanly."""

from __future__ import annotations

import pytest

from src.components import all_registries, describe_all, validate_config
from src.components.domain import REGISTRY as DOMAINS
from src.components.domain.profiles import DomainBackend
from src.components.generation import REGISTRY as GENERATORS
from src.components.selection import REGISTRY as SELECTION
from src.contracts.config import (
    BackendConfig,
    BackgroundConfig,
    GeneratorConfig,
    LatticeConfig,
    default,
    validate,
)
from src.contracts.errors import ConfigError, ConfigValueError


def test_every_axis_lists_at_least_one_backend() -> None:
    tables = all_registries()
    empty = [axis for axis, registry in tables.items() if len(registry) == 0]
    assert empty == [], f"unpopulated axes after the join: {empty}"
    listing = describe_all()
    for axis in tables:
        assert f"{axis}:" in listing
        assert "(nothing registered)" not in listing.split(f"{axis}:", 1)[1].split("\n\n", 1)[0]


def test_default_config_validates_against_every_registry() -> None:
    cfg = default()
    validate(cfg)
    validate_config(cfg)


def test_general_domain_uses_identity_selection_and_no_panorama() -> None:
    backend = DOMAINS.build("general")
    assert isinstance(backend, DomainBackend)
    assert backend.selector == "identity"
    SELECTION.spec(backend.selector)
    cfg = default().with_(
        domain="general",
        background=BackgroundConfig(method="none"),
        selection=BackendConfig(backend="identity"),
    )
    validate(cfg)
    validate_config(cfg)


def test_tennis_heuristic_is_rejected_under_the_general_domain() -> None:
    cfg = default().with_(
        domain="general",
        background=BackgroundConfig(method="none"),
        selection=BackendConfig(backend="heuristic"),
    )
    validate(cfg)
    with pytest.raises(ConfigError, match="not domain 'general'"):
        validate_config(cfg)


def test_a_named_generator_that_cannot_decode_the_pair_is_rejected() -> None:
    """Some other generator accepting the pair is not enough once one is named."""
    cfg = default().with_(
        generator=GeneratorConfig(backend="pix2pix"),
        lattice=LatticeConfig(generation=True),
    )
    # pix2pix declares keypoints, not the racket/ball sparse-trajectory fallback.
    if GENERATORS.spec("pix2pix").accepts("motion", "sparse-trajectories"):
        pytest.skip("pix2pix now declares sparse-trajectories")
    validate(cfg)
    with pytest.raises((ConfigError, ConfigValueError), match="pix2pix"):
        validate_config(cfg)
