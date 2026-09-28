"""Geometry helpers for the clip-1 hand-model comparison."""

import numpy as np

from demo.experiments.archive.compare_hand_models_clip1 import (
    fit_joints_to_bbox,
    network_to_frame,
    openpose_from_mano,
    payload_kbps,
    quantize_axis_angle,
)


def test_quantize_axis_angle_roundtrip_stays_close():
    pose = np.linspace(-1.5, 1.5, 45, dtype=np.float32)
    packet, recovered = quantize_axis_angle(pose)
    assert len(packet) == 4 + 45
    assert recovered.shape == (45,)
    assert np.max(np.abs(recovered - pose)) < 1.5 / 127.0 + 1e-5


def test_payload_kbps_matches_bytes_per_frame():
    packets = [b"abc", b"abcd"]
    assert payload_kbps(packets, fps=30.0) == (7 * 8) / (2 / 30.0) / 1000.0


def test_network_pixels_scale_to_the_frame():
    mapped = network_to_frame(np.array([[112.0, 112.0], [0.0, 224.0]]))
    assert mapped[0, 0] == 1920 / 2
    assert mapped[0, 1] == 1080 / 2
    assert mapped[1, 1] == 1080


def test_openpose_from_mano_places_wrist_and_tips():
    joints = np.arange(16 * 3, dtype=np.float64).reshape(16, 3)
    vertices = np.arange(778 * 3, dtype=np.float64).reshape(778, 3)
    posed = openpose_from_mano(joints, vertices)
    assert posed.shape == (21, 3)
    assert np.allclose(posed[0], joints[0])
    assert np.allclose(posed[4], vertices[743])
    assert np.allclose(posed[20], vertices[671])


def test_bbox_fit_lands_inside_the_box():
    joints = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [-0.1, 0.05, 0.0]])
    # 21 points are not required by the fitter.
    pixels = fit_joints_to_bbox(joints, [100, 200, 300, 400])
    assert pixels.shape == (3, 2)
    assert 100 <= pixels[0, 0] <= 300
    assert 200 <= pixels[0, 1] <= 400
