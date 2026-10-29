"""Receiver pose prompts must not depend on a segmentation or target box."""
import numpy as np
import pytest

from experiments.tennis.k6b_receiver_sam import pose_prompt


def test_pose_prompt_bounds_labels_and_visibility() -> None:
    pose = np.zeros((17, 3))
    pose[5] = [100, 200, 1]
    pose[6] = [200, 200, 1]
    pose[11] = [120, 400, 1]
    pose[12] = [180, 400, 1]
    pose[0] = [10000, 10000, 1]  # outside the actually delivered crop
    points, labels = pose_prompt(pose)
    assert labels.tolist() == [2, 3, 1, 1, 1, 1]
    np.testing.assert_allclose(points[:2] * [512, 784], [[90, 180], [210, 420]])
    np.testing.assert_allclose(points[2:] * [512, 784], pose[[5, 6, 11, 12], :2])
    assert ((points >= 0) & (points <= 1)).all()


def test_pose_prompt_rejects_missing_receiver_support() -> None:
    with pytest.raises(ValueError, match="Insufficient paid pose"):
        pose_prompt(np.zeros((17, 3)))
