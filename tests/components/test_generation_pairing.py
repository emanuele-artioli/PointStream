"""Appearance/motion pairing: an undecodable pair names what would work."""

from __future__ import annotations

import pytest

from src.components.appearance import REGISTRY as APPEARANCE
from src.components.generation import REGISTRY as GENERATORS
from src.components.generation import validate
from src.components.motion import REGISTRY as MOTION
from src.contracts.capabilities import (
    APPEARANCE_COMPRESSED_IMAGE,
    APPEARANCE_IMAGE_EMBEDDING,
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_SPARSE_TRAJECTORIES,
)
from src.contracts.config import AppearanceConfig, MotionConfig, default
from src.contracts.errors import ConfigError, UndecodableStreamError
from src.contracts.objectstream import assert_decodable, workable_pairings


def test_compressed_image_plus_keypoints_is_decodable():
    spec = assert_decodable(APPEARANCE_COMPRESSED_IMAGE, MOTION_KEYPOINTS, GENERATORS)
    assert spec.name in GENERATORS


def test_an_undecodable_pair_names_workable_pairings():
    with pytest.raises(UndecodableStreamError) as excinfo:
        assert_decodable(APPEARANCE_IMAGE_EMBEDDING, MOTION_ENCODED_VIDEO, GENERATORS)
    message = str(excinfo.value)
    assert APPEARANCE_IMAGE_EMBEDDING in message
    assert MOTION_ENCODED_VIDEO in message
    assert f"{APPEARANCE_COMPRESSED_IMAGE}+{MOTION_KEYPOINTS}" in message
    assert f"{APPEARANCE_COMPRESSED_IMAGE}+{MOTION_SPARSE_TRAJECTORIES}" in message
    assert f"{APPEARANCE_IMAGE_EMBEDDING}+{MOTION_ENCODED_VIDEO}" not in workable_pairings(
        GENERATORS
    )


def test_validate_backends_rejects_an_undecodable_config_and_keeps_none_unset():
    cfg = default().with_(
        appearance=AppearanceConfig(representation=APPEARANCE_IMAGE_EMBEDDING),
        motion=MotionConfig(representation=MOTION_ENCODED_VIDEO),
    )
    with pytest.raises(ConfigError, match="No registered generator can decode"):
        validate(cfg)


def test_default_tennis_pairing_validates_because_fallbacks_are_covered():
    """Racket/ball fall back to sparse-trajectories; MOFA declares that half."""
    validate(default())
    assert (APPEARANCE_COMPRESSED_IMAGE, MOTION_SPARSE_TRAJECTORIES) in workable_pairings(
        GENERATORS
    )


def test_appearance_and_motion_axes_are_registered_under_their_vocabulary_names():
    APPEARANCE.spec(APPEARANCE_COMPRESSED_IMAGE)
    MOTION.spec(MOTION_KEYPOINTS)
    MOTION.spec(MOTION_SPARSE_TRAJECTORIES)
    MOTION.spec(MOTION_ENCODED_VIDEO)
