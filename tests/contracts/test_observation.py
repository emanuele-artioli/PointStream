from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import (
    EstimatorProvenance,
    Observation,
    ObservationStatus,
    PIXEL_COORDINATES,
    Pts,
)


def _provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        name="sam3.1_multiplex",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        policy="offline_bidirectional",
    )


def test_missing_observation_is_a_full_identity_record() -> None:
    item = Observation.missing(
        source_id="source-01",
        frame_index=4,
        pts=Pts(120, 1, 30),
        object_id="racket-2",
        object_class="racket",
        frame_width=64,
        frame_height=48,
        provenance=_provenance(),
        reason="tracker_lost",
        associated_player_id="player-1",
        associated_wrist="right_wrist",
    )
    record = item.to_record()
    assert item.status is ObservationStatus.MISSING
    assert item.mask is None
    assert item.pts.seconds == 4.0
    assert record["coordinate_system"] == PIXEL_COORDINATES
    assert record["associated_player_id"] == "player-1"
    assert record["reason"] == "tracker_lost"


def test_observed_mask_must_be_binary_and_frame_sized() -> None:
    base = dict(
        source_id="source-01",
        frame_index=0,
        pts=Pts(0, 1, 30),
        object_id="player-1",
        object_class="player",
        coordinate_system=PIXEL_COORDINATES,
        frame_width=3,
        frame_height=2,
        provenance=_provenance(),
        status=ObservationStatus.OBSERVED,
    )
    with pytest.raises(ValueError, match="shape"):
        Observation(**base, mask=np.zeros((1, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="binary"):
        Observation(**base, mask=np.full((2, 3), 2, dtype=np.uint8))

