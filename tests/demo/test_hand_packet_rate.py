import numpy as np

from demo.experiments.hand_packet_rate import (
    _decode_per_frame,
    decode_segment_delta,
    landmark_error,
    per_frame_bytes,
    score_tracks,
    segment_delta_bytes,
)


def _hand(x: float, y: float, side: str = "Left") -> dict:
    joints = [[x + 2 * i, y + i] for i in range(21)]
    return {
        "side": side,
        "confidence": 0.8,
        "box": [x, y, x + 80, y + 100],
        "landmarks_pixel": joints,
    }


def test_still_hand_packs_tighter_as_one_segment():
    frames = [[_hand(100, 200)] for _ in range(30)]
    one = len(segment_delta_bytes(frames, 8))
    per = len(per_frame_bytes(frames, 8))
    assert one < per / 5


def test_four_bit_joints_move_more_than_eight_bit():
    frames = [[_hand(400.4, 500.2)]]
    assert landmark_error(frames, 4) > landmark_error(frames, 8)


def test_delta_packet_restores_the_quantized_joints():
    frames = [[_hand(100 + i * 3, 200 + (i % 5))] for i in range(20)]
    decoded = decode_segment_delta(segment_delta_bytes(frames, 8))
    for hands, pred in zip(frames, decoded):
        alone = _decode_per_frame(per_frame_bytes([hands], 8), 1920, 1080)[0]
        assert np.allclose(pred[0], alone, atol=1e-6)


def test_segment_length_does_not_change_joint_error():
    frames = [[_hand(100 + i, 200)] for i in range(10)]
    rows = score_tracks(frames, segment_lengths=(1, 10), fps=30.0)
    by_name = {row["technique"]: row for row in rows if row["segment_frames"] == 10}
    short = {row["technique"]: row for row in rows if row["segment_frames"] == 1}
    assert by_name["segment_delta_u8_zlib"]["bytes"] < short["segment_delta_u8_zlib"]["bytes"]
    assert by_name["segment_delta_u8_zlib"]["mean_joint_px"] == short["segment_delta_u8_zlib"]["mean_joint_px"]
    assert np.isfinite(by_name["per_frame_u4"]["mean_joint_px"])
