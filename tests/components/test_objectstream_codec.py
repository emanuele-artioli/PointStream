"""objectstream.v1: lossless codes, parity with DWB2 and PSFG, and corruption."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from src.components.transport import objectstream_codec as oc

HAND = oc.ObjectClass("hand", 21, attribute_bits=2, confidence_bits=7)
PLAYER = oc.ObjectClass("player", 17)
BALL = oc.ObjectClass("ball")


def _random_frames(config: oc.StreamConfig, seed: int = 7, frames: int = 40):
    """Tracks that appear, vanish, drift and lose joints, as pixel observations."""
    rng = random.Random(seed)
    classes = (HAND, PLAYER, BALL)
    tracks = {
        0: (0, [300.0, 200.0]),
        1: (0, [900.0, 400.0]),
        5: (1, [1200.0, 500.0]),
        9: (2, [50.0, 60.0]),
    }
    out = []
    for index in range(frames):
        frame = []
        for track_id, (cls, centre) in tracks.items():
            if track_id == 1 and 10 <= index < 15:
                continue  # leaves and returns inside the segment
            if track_id == 9 and index < 5:
                continue  # appears late
            centre[0] = min(max(centre[0] + rng.uniform(-6, 6), 80), config.width - 80)
            centre[1] = min(max(centre[1] + rng.uniform(-4, 4), 80), config.height - 80)
            half = 60.0 if cls != 2 else 4.0
            box = (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)
            declared = classes[cls]
            joints = tuple(
                (rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3]))
                for _ in range(declared.joints)
            )
            present = tuple(rng.random() > 0.1 for _ in range(declared.joints))
            frame.append(
                oc.ObjectObservation(
                    track_id=track_id,
                    class_index=cls,
                    box=box,
                    joints=joints,
                    present=present,
                    attribute=rng.randrange(1 << declared.attribute_bits),
                    confidence=rng.randrange(1 << declared.confidence_bits),
                )
            )
        out.append(frame)
    return classes, out


CONFIGS = [
    oc.StreamConfig(1920, 1080),
    oc.StreamConfig(1920, 1080, box_motion_prediction=True),
    oc.StreamConfig(1920, 1080, box_grid="frame", box_bits=8, joint_grid="box", joint_bits=8),
    oc.StreamConfig(
        3840, 2160, box_grid="frame", box_bits=10, joint_grid="box", joint_bits=6,
        box_motion_prediction=True,
    ),
    oc.StreamConfig(3840, 2160, box_bits=0, joint_bits=1),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_codes_round_trip_exactly(config: oc.StreamConfig) -> None:
    classes, frames = _random_frames(config)
    report = oc.QuantizationReport()
    coded = [[oc.quantize(config, classes, item, report) for item in frame] for frame in frames]
    packed = oc.encode_codes(config, classes, coded, segment_start=120)
    decoded = oc.decode(packed.data)
    assert decoded.config == config
    assert decoded.classes == classes
    assert decoded.segment_start == 120
    assert [list(frame) for frame in decoded.frames] == [
        sorted(frame, key=lambda item: item.track_id) for frame in coded
    ]
    assert packed.header_bytes + packed.payload_bytes == packed.total_bytes


@pytest.mark.parametrize("config", CONFIGS)
def test_quantization_error_is_bounded_by_the_grid(config: oc.StreamConfig) -> None:
    classes, frames = _random_frames(config)
    encoded = oc.encode(config, classes, frames)
    report = encoded.quantization
    assert report.clamped_codes == 0
    if config.box_grid == "pixel":
        box_step = 1.0 / (1 << config.box_bits)
    else:
        box_step = max(config.width, config.height) / ((1 << config.box_bits) - 1)
    # Box-relative joints round the box outward (one step); otherwise half a step.
    box_bound = box_step if config.joint_grid == "box" else box_step / 2
    assert report.max_box_error_px <= box_bound + 1e-9
    if config.joint_grid == "pixel":
        joint_bound = 0.5 / (1 << config.joint_bits)
    else:
        widest = 120.0 + 2 * box_step
        joint_bound = 0.5 * widest / ((1 << config.joint_bits) - 1)
    assert report.max_joint_error_px <= joint_bound + 1e-9
    decoded = oc.decode(encoded.data).observations()
    for source, rebuilt in zip(frames, decoded):
        by_id = {item.track_id: item for item in rebuilt}
        for item in source:
            got = by_id[item.track_id]
            for flag, (x, y) in zip(got.present, got.joints):
                assert flag == (not math.isnan(x))
            assert got.attribute == item.attribute and got.confidence == item.confidence


def test_prediction_beats_absolute_codes_on_smooth_motion() -> None:
    config = oc.StreamConfig(1920, 1080)
    classes, frames = _random_frames(config)
    smooth = [
        [replace(item, joints=tuple((x + 0.0, y) for x, y in frames[0][i].joints))
         for i, item in enumerate(frame) if i < len(frames[0])
         and frame[i].track_id == frames[0][i].track_id]
        for frame in frames
    ]
    single = oc.encode(config, classes, smooth[:1])
    many = oc.encode(config, classes, smooth)
    per_frame_absolute = single.payload_bytes
    assert many.payload_bytes < per_frame_absolute * len(smooth) / 2


def test_corruption_and_truncation_are_rejected() -> None:
    config = oc.StreamConfig(1920, 1080)
    classes, frames = _random_frames(config, frames=8)
    data = oc.encode(config, classes, frames).data
    flipped = bytearray(data)
    flipped[-3] ^= 0x10
    with pytest.raises(ValueError, match="CRC"):
        oc.decode(bytes(flipped))
    with pytest.raises(ValueError):
        oc.decode(data[:-1])
    with pytest.raises(ValueError):
        oc.decode(data + b"\x00")
    with pytest.raises(ValueError):
        oc.decode(b"PSFG" + data[4:])


def test_invalid_records_are_rejected_at_encode() -> None:
    config = oc.StreamConfig(640, 480)
    good = oc.CodedObject(0, 0, (0, 0, 16, 16))
    with pytest.raises(ValueError, match="undeclared|not declared"):
        oc.encode_codes(config, (BALL,), [[replace(good, class_index=3)]])
    with pytest.raises(ValueError, match="unique"):
        oc.encode_codes(config, (BALL,), [[good, good]])
    with pytest.raises(ValueError, match="changes class"):
        oc.encode_codes(config, (BALL, PLAYER), [[good], [replace(good, class_index=1,
                        joints=(0,) * 34, present=(True,) * 17)]])
    with pytest.raises(ValueError, match="outside"):
        oc.encode_codes(config, (BALL,), [[replace(good, box=(0, 0, 640 * 16 + 1, 16))]])


def test_out_of_frame_coordinates_are_clamped_and_counted() -> None:
    config = oc.StreamConfig(100, 100)
    observation = oc.ObjectObservation(0, 0, (-5.0, 10.0, 120.0, 50.0))
    encoded = oc.encode(config, (BALL,), [[observation]])
    assert encoded.quantization.clamped_codes == 2


def test_dwb2_codes_carry_over_losslessly_and_no_larger() -> None:
    """Code-level parity: DWB2's decoded PK codes re-encode exactly here."""
    pose_delta = pytest.importorskip("demo.pipeline.maps.pose_delta")
    rng = random.Random(3)
    packets = []
    for index in range(60):
        hands = []
        for slot in range(1 if index % 7 == 0 else 2):
            base = 60 + slot * 90 + index // 4
            box = [min(base, 250), 80, min(base + 30, 255), 140]
            lm = [min(255, max(0, 128 + rng.randint(-3, 3) + j)) for j in range(42)]
            hands.append((slot, rng.randrange(128), box, lm))
        packets.append(
            (pose_delta._emit_part(b"PK", hands, True), b"PF\x00", b"PB\x00")
        )
    blob = pose_delta.encode_pose_stream(packets)
    frames = pose_delta.decode_pose_stream(blob)
    config = oc.StreamConfig(
        1920, 1080, box_grid="frame", box_bits=8, joint_grid="box", joint_bits=8
    )
    classes = (oc.ObjectClass("hand", 21, attribute_bits=1, confidence_bits=7),)
    coded = []
    for hand_packet, _, _ in frames:
        instances = pose_delta._parse_part(hand_packet, b"PK", 21, True)
        coded.append(
            [
                oc.CodedObject(slot, 0, tuple(box), tuple(lm), (True,) * 21, side, conf)
                for slot, (side, conf, box, lm) in enumerate(instances)
            ]
        )
    packed = oc.encode_codes(config, classes, coded)
    assert [list(frame) for frame in oc.decode(packed.data).frames] == coded
    assert packed.total_bytes <= len(blob) + packed.header_bytes


def test_psfg_codes_carry_over_losslessly_and_smaller() -> None:
    """Code-level parity with PSFG v2 at 1/16 px, the lossless operating point."""
    psfg = pytest.importorskip("demo.pipeline.foreground_codec_v2")
    rng = random.Random(5)
    frames = []
    for index in range(45):
        hands = []
        for track_id, x0 in ((3, 400.0), (8, 1100.0)):
            x = x0 + index * 1.5 + rng.uniform(-0.5, 0.5)
            box = (x, 300.0, x + 120.0, 420.0)
            joints = tuple(
                (x + 10 + j * 4.8 + rng.uniform(-0.3, 0.3), 320 + j * 3.1) for j in range(21)
            )
            hands.append(psfg.TrackedHand(track_id, ("left", "right")[track_id % 2], box, joints))
        frames.append(hands)
    packet = psfg.encode_segment(frames, width=1920, height=1080, start_frame=0,
                                 method="delta_zlib")
    decoded = psfg.decode_segment(packet)
    config = oc.StreamConfig(1920, 1080, box_bits=4, joint_bits=4, box_motion_prediction=True)
    classes = (oc.ObjectClass("hand", 21, attribute_bits=2),)
    side = {None: 0, "left": 1, "right": 2}
    coded = [
        [
            oc.CodedObject(
                hand.track_id,
                0,
                tuple(int(round(v * 16)) for v in hand.bbox),
                tuple(int(round(v * 16)) for point in hand.joints for v in point),
                (True,) * 21,
                side[hand.handedness],
            )
            for hand in frame
        ]
        for frame in decoded["frames"]
    ]
    packed = oc.encode_codes(config, classes, coded)
    assert [list(frame) for frame in oc.decode(packed.data).frames] == coded
    assert packed.total_bytes < len(packet)
