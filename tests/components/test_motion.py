"""Motion representations: sparse trajectories, never dense flow."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.motion import REGISTRY as MOTION
from src.components.motion.encoded_video import EncodedVideoMotionEncoder
from src.components.motion.keypoints import KeypointMotionEncoder
from src.components.motion.trajectories import SparseTrajectoryEncoder
from src.contracts.capabilities import (
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_SPARSE_TRAJECTORIES,
)
from src.contracts.objectstream import MAX_SPARSE_POINTS


def test_sparse_trajectories_accept_a_handful_of_points():
    encoder = SparseTrajectoryEncoder()
    points = np.linspace(0, 1, 16 * 2, dtype=np.float32).reshape(16, 2)
    desc, payload = encoder.encode(points)
    assert desc.kind == MOTION_SPARSE_TRAJECTORIES
    assert desc.point_count == 16
    decoded = encoder.decode(payload, desc)
    assert decoded.shape == (16, 2)


def test_sparse_trajectories_reject_a_dense_flow_field():
    encoder = SparseTrajectoryEncoder()
    dense = np.zeros((64, 64, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dense flow"):
        encoder.encode(dense)


def test_sparse_trajectories_reject_a_point_count_above_the_ceiling():
    encoder = SparseTrajectoryEncoder()
    too_many = np.zeros((MAX_SPARSE_POINTS + 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=str(MAX_SPARSE_POINTS)):
        encoder.encode(too_many)


def test_keypoints_refuse_a_joint_count_that_does_not_match_the_schema():
    encoder = KeypointMotionEncoder()
    with pytest.raises(ValueError, match="joints"):
        encoder.encode(np.zeros((5, 3), dtype=np.float32))
    points = np.zeros((18, 3), dtype=np.float32)
    points[0] = (12.0, 24.0, 0.9)
    desc, payload = encoder.encode(points)
    assert desc.kind == MOTION_KEYPOINTS
    decoded = encoder.decode(payload)
    np.testing.assert_allclose(decoded[0].astype(np.float32), points[0], atol=1e-2)


def test_encoded_video_carries_a_real_encode_request_not_just_a_codec_name():
    encoder = EncodedVideoMotionEncoder(width=48, height=64)
    desc = encoder.encode(measured_bytes_per_frame=1200)
    assert desc.kind == MOTION_ENCODED_VIDEO
    assert desc.request.codec_name == "av1"
    assert desc.width == 48 and desc.height == 64
    assert desc.cost().byte_count == 1200


def test_registry_names_match_the_capability_vocabulary():
    for name in (MOTION_KEYPOINTS, MOTION_SPARSE_TRAJECTORIES, MOTION_ENCODED_VIDEO):
        MOTION.spec(name)
        MOTION.build(name)
