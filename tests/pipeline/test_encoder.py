"""The encoder is a lattice corner with injected backends.

It does not loop over chunks, look up a registry, or special-case the
baseline. Those are C3. What it must do: every enumerated corner encodes;
a disabled stage still costs nothing when driven through ``Encoder.encode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from src.contracts.config import LatticeConfig, PointstreamConfig
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_BITSTREAM,
    ART_DELIVERED,
    ART_QUALITY,
    FULL,
    REQUIRED_STAGES,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_METRICS,
    STAGE_TRANSPORT,
    StageLattice,
)
from src.pipeline.dag import iter_lattice_corners
from src.pipeline.encoder import SOURCE, Encoder
from tests.pipeline.clocks import full_roster


def _corner_id(lattice: StageLattice) -> str:
    return lattice.label()


@pytest.mark.parametrize("lattice", iter_lattice_corners(), ids=_corner_id)
def test_every_enumerated_corner_encodes(lattice: StageLattice) -> None:
    encoder = Encoder.build(lattice, full_roster())
    assert encoder.stages == lattice.dag()
    bag = encoder.encode({SOURCE: "chunk"})
    assert bag[SOURCE] == "chunk"
    assert ART_BITSTREAM in bag
    assert ART_DELIVERED in bag
    assert ART_QUALITY in bag
    for name in REQUIRED_STAGES:
        assert name in bag


def test_encoder_builds_from_a_lattice_config() -> None:
    encoder = Encoder.build(LatticeConfig(), full_roster())
    assert encoder.lattice == LatticeConfig().to_lattice()
    bag = encoder.encode({SOURCE: "chunk"})
    assert ART_QUALITY in bag


def test_encoder_builds_from_a_pointstream_config() -> None:
    config = PointstreamConfig()
    encoder = Encoder.build(config, full_roster())
    assert encoder.lattice == config.stages
    encoder.encode({SOURCE: "chunk"})


def test_encode_forwards_the_source_bag_to_the_first_stage() -> None:
    seen: dict[str, object] = {}

    def codec(bag: Mapping[str, Any]) -> str:
        seen.update(bag)
        return "bits"

    roster: dict[str, Any] = dict(full_roster())
    roster[STAGE_CODEC] = codec
    encoder = Encoder.build(StageLattice.all_off(), roster)
    encoder.encode({SOURCE: "frames", "hint": 7})
    assert seen[SOURCE] == "frames"
    assert seen["hint"] == 7


def test_incoherent_corner_fails_at_encoder_build() -> None:
    with pytest.raises(ConfigValueError, match="subjects"):
        Encoder.build(FULL.disable(STAGE_DETECTION), full_roster())


def test_encoder_stages_match_the_required_spine_when_everything_is_off() -> None:
    encoder = Encoder.build(StageLattice.all_off(), full_roster())
    assert encoder.stages == (STAGE_CODEC, STAGE_TRANSPORT, STAGE_METRICS)
