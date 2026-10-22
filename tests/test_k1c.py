"""K1c: K6's skeleton warp with the 21-joint hand skeleton (docs/experiments.md, 2026-10-21 K1c)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from experiments.visor import k1c  # noqa: E402
from experiments.visor.h3 import HANDS  # noqa: E402


def hand_joints(x: float, y: float) -> np.ndarray:
    """A flat open hand: the wrist at (x, y), five fingers of four joints spread upwards; confidence 1."""
    kp = [(x, y)]
    for f in range(5):
        dx = (f - 2) * 8.0
        kp += [(x + dx * (1 + j / 3), y - 10.0 - 8.0 * j) for j in range(4)]
    return np.concatenate([np.asarray(kp), np.ones((21, 1))], 1)


def scene() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    right = np.zeros((120, 160), bool)
    right[30:100, 50:110] = True
    return image, {HANDS[0]: np.zeros_like(right), HANDS[1]: right}, np.zeros_like(right)


def test_skeleton() -> None:
    assert len(k1c.HAND_SKELETON) == 20
    assert {j for pair in k1c.HAND_SKELETON for j in pair} == set(range(21))
    assert sum(1 for a, _ in k1c.HAND_SKELETON if a == 0) == 5


def test_identity_reproduces_the_source() -> None:
    image, sides, objects = scene()
    kp = hand_joints(80, 90)
    frame, drawn = k1c.compose(image, sides, objects, [(True, kp, kp, 1.0)])
    assert drawn[sides[HANDS[1]]].all()
    assert np.array_equal(frame[sides[HANDS[1]]], image[sides[HANDS[1]]])


def test_translation_moves_the_hand_and_spares_objects() -> None:
    image, sides, objects = scene()
    objects[0:120, 0:45] = True  # outside R, as the job's objects (foreground minus R)
    kp = hand_joints(80, 90)
    moved = kp.copy()
    moved[:, 0] -= 6
    frame, drawn = k1c.compose(image, sides, objects, [(True, kp, moved, 1.0)])
    assert np.array_equal(frame[50:90, 54:88], image[50:90, 60:94])
    assert not (drawn & objects).any()
    assert np.array_equal(frame[objects], image[objects])


def test_unfitted_hand_is_filled() -> None:
    image, sides, objects = scene()
    frame, drawn = k1c.compose(image, sides, objects, [])
    assert not drawn.any()
    assert not np.array_equal(frame[sides[HANDS[1]]], image[sides[HANDS[1]]])
