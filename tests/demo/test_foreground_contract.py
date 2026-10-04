from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from demo.pipeline.foreground_codec_v2 import (
    CausalTrackAssociator,
    HandDetection,
    TrackedHand,
    _HEADER,
    accept_scored_prediction,
    av1_slots,
    decode_segment,
    decoded_bbox_int,
    encode_segment,
    generator_slots,
    map_points_to_letterbox,
)
from demo.pipeline.foreground_segmenter import letterbox_crop


def _detection(x: float, side: str | None = "left", tag: str = "") -> HandDetection:
    return HandDetection(
        bbox=(x, 10, x + 20, 30),
        joints=tuple((x + 1 + (i % 4), 12 + (i // 4)) for i in range(21)),
        handedness=side,
        score=0.9,
        candidate_id=tag,
    )


def _tracked(track_id: int, x: float, side: str | None = "left") -> TrackedHand:
    detection = _detection(x, side)
    return TrackedHand(track_id, side, detection.bbox, detection.joints)


def test_track_ids_do_not_depend_on_left_right_labels_and_survive_short_gap():
    associator = CausalTrackAssociator(200, 100)
    first = associator.assign([_detection(10, "left", "a"), _detection(100, "left", "b")])
    assert [hand.track_id for hand in first.hands] == [0, 1]
    assert [hand.handedness for hand in first.hands] == ["left", "left"]

    swapped = associator.assign([_detection(101, "right", "b"), _detection(11, "right", "a")])
    assert [(hand.track_id, hand.handedness) for hand in swapped.hands] == [(0, "right"), (1, "right")]
    associator.assign([])
    reappeared = associator.assign([_detection(12, None, "a")])
    assert len(reappeared.hands) == 1
    assert reappeared.hands[0].track_id == 0
    assert reappeared.hands[0].handedness is None
    assert any(event["kind"] == "association_cost" for event in associator.events)


def test_duplicate_candidates_are_resolved_but_third_distinct_hand_is_unsupported():
    associator = CausalTrackAssociator(200, 100)
    result = associator.assign([
        _detection(10, "left", "a"),
        _detection(10.5, "left", "duplicate"),
        _detection(90, "left", "b"),
    ])
    assert result.supported and result.duplicate_count == 1
    assert len(result.hands) == 2
    assert result.hands[0].track_id != result.hands[1].track_id

    unsupported = associator.assign([_detection(10), _detection(90), _detection(150)])
    assert not unsupported.supported
    assert unsupported.hands == ()
    assert unsupported.reason == "more_than_two_unresolved_hands"


def test_long_gap_resets_track_and_the_next_record_is_complete_state():
    associator = CausalTrackAssociator(200, 100, max_gap=0)
    associator.assign([_detection(10, "left", "a")])
    associator.assign([])
    reset = associator.assign([_detection(12, "right", "a")])
    assert reset.hands[0].track_id == 1
    assert any(event["kind"] == "reset" for event in associator.events)
    delta = decode_segment(encode_segment([reset.hands], width=200, height=100, start_frame=2, method="delta_zlib"))
    raw = decode_segment(encode_segment([reset.hands], width=200, height=100, start_frame=2, method="raw"))
    assert delta["frames"] == raw["frames"]
    assert delta["frames"][0][0].presence is True


@pytest.mark.parametrize("method", ["raw", "zlib", "delta_zlib"])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_v2_packet_roundtrips_presence_and_1_16_pixel_coordinates(method, count):
    frame = [_tracked(index, 4.0 + index * 50, "right" if index else None) for index in range(count)]
    packet = encode_segment([frame], width=160, height=80, start_frame=17, method=method)
    decoded = decode_segment(packet)
    assert decoded["start_frame"] == 17
    assert decoded["method"] == method
    assert len(decoded["frames"][0]) == count
    for expected, actual in zip(frame, decoded["frames"][0]):
        assert actual.track_id == expected.track_id
        assert actual.handedness == expected.handedness
        assert actual.presence is True
        for pair_a, pair_b in zip((*expected.bbox, *(v for p in expected.joints for v in p)),
                                  (*actual.bbox, *(v for p in actual.joints for v in p))):
            assert abs(pair_a - pair_b) <= 1 / 32 + 1e-9


def test_odd_nonsquare_box_stays_within_a_thirty_second_of_a_pixel():
    joints = tuple((11.2 + (i % 3), 8.4 + (i % 5)) for i in range(21))
    hand = TrackedHand(4, "left", (10.2, 3.4, 31.7, 48.9), joints)
    decoded = decode_segment(encode_segment([[hand]], width=1920, height=1080, start_frame=0))["frames"][0][0]
    for actual, expected in zip(decoded.bbox, hand.bbox):
        assert abs(actual - expected) <= 1 / 32 + 1e-9
    assert decoded.bbox[2] - decoded.bbox[0] != decoded.bbox[3] - decoded.bbox[1]


@pytest.mark.parametrize("bbox,joints", [
    ((-1, 0, 20, 20), None),
    ((0, 0, 20, 20), "nan"),
    ((0, 0, 20, 20), "outside"),
])
def test_packet_rejects_invalid_coordinates(bbox, joints):
    hand = _tracked(0, 0)
    points = list(hand.joints)
    if joints == "nan":
        points[0] = (float("nan"), 1)
    elif joints == "outside":
        points[0] = (161, 1)
    bad = TrackedHand(0, "left", bbox, tuple(points))
    with pytest.raises(ValueError):
        encode_segment([[bad]], width=160, height=80, start_frame=0)


def test_decoder_rejects_truncation_version_crc_trailing_bytes_and_bad_counts():
    packet = encode_segment([[_tracked(0, 4)]], width=160, height=80, start_frame=0, method="zlib")
    with pytest.raises(ValueError, match="truncated"):
        decode_segment(packet[:10])
    wrong_version = bytearray(packet)
    wrong_version[4] = 99
    with pytest.raises(ValueError, match="version"):
        decode_segment(bytes(wrong_version))
    bad_crc = bytearray(packet)
    bad_crc[32] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_segment(bytes(bad_crc))
    with pytest.raises(ValueError, match="length"):
        decode_segment(packet + b"x")

    raw = encode_segment([[]], width=160, height=80, start_frame=0)
    over_count = bytearray(raw)
    over_count[36] = 3
    struct.pack_into("<I", over_count, 32, zlib_crc32(over_count[36:]))
    with pytest.raises(ValueError, match="more than two"):
        decode_segment(bytes(over_count))


def zlib_crc32(value: bytes) -> int:
    import zlib
    return zlib.crc32(value) & 0xFFFFFFFF


def test_schema_header_width_matches_the_encoder():
    schema = json.loads(Path("demo/pipeline/foreground_packet_schema.json").read_text())
    assert _HEADER.size == schema["header_bytes"] == 36
    assert schema["magic"] == "PSFG"
    assert schema["max_quantization_error_px"] == 1 / 32


def test_segments_decode_independently_and_future_frames_do_not_change_prior_output():
    first = _tracked(3, 10)
    later = _tracked(3, 15)
    a = encode_segment([[first], [later]], width=160, height=80, start_frame=0, method="delta_zlib")
    b = encode_segment([[first], [_tracked(3, 90)]], width=160, height=80, start_frame=0, method="delta_zlib")
    decoded_a = decode_segment(a)["frames"]
    decoded_b = decode_segment(b)["frames"]
    assert decoded_a[0] == decoded_b[0]
    independent = encode_segment([[later]], width=160, height=80, start_frame=1, method="delta_zlib")
    assert decode_segment(independent)["frames"][0][0].bbox == decode_segment(
        encode_segment([[later]], width=160, height=80, start_frame=1, method="raw")
    )["frames"][0][0].bbox


def test_poisoned_source_records_cannot_change_packet_decode_or_crop_geometry():
    source = {"bbox": [10, 10, 30, 30], "joints": [[11 + (i % 4), 12] for i in range(21)]}
    hand = TrackedHand(5, "left", tuple(source["bbox"]), tuple(tuple(point) for point in source["joints"]))
    packet = encode_segment([[hand]], width=160, height=80, start_frame=0)
    expected = decode_segment(packet)["frames"][0][0]
    source["bbox"] = [0, 0, 1, 1]
    source["joints"] = [[0, 0] for _ in range(21)]
    actual = decode_segment(packet)["frames"][0][0]
    assert actual == expected
    assert decoded_bbox_int(actual, 160, 80) == decoded_bbox_int(expected, 160, 80)
    assert av1_slots(decode_segment(packet)) == generator_slots(decode_segment(packet))
    with pytest.raises(KeyError, match="not present"):
        accept_scored_prediction(decode_segment(packet), frame_index=0, track_id=99)


def test_decoded_joint_markers_align_with_real_letterbox_crop_geometry():
    import cv2

    image = np.zeros((70, 100, 3), dtype=np.uint8)
    image[30, 50] = (0, 0, 255)
    hand = TrackedHand(0, "left", (10, 10, 90, 50), tuple([(50, 30)] * 21))
    decoded = decode_segment(encode_segment([[hand]], width=100, height=70, start_frame=0))["frames"][0][0]
    crop, meta = letterbox_crop(image, list(decoded_bbox_int(decoded, 100, 70)), target_size=80, interpolation=cv2.INTER_NEAREST)
    mapped = map_points_to_letterbox(decoded.joints, meta)
    px, py = (int(round(v)) for v in mapped[0])
    assert crop[py, px, 2] == 255
    assert (px, py) == (40, 40)


def test_edge_crop_uses_the_same_integer_window_as_letterbox():
    import cv2

    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[0, 0] = (0, 255, 0)
    hand = TrackedHand(0, None, (0, 0, 15.2, 21.4), tuple([(0, 0)] * 21))
    decoded = decode_segment(encode_segment([[hand]], width=50, height=40, start_frame=0))["frames"][0][0]
    box = list(decoded_bbox_int(decoded, 50, 40))
    crop, meta = letterbox_crop(image, box, target_size=32, interpolation=cv2.INTER_NEAREST)
    mapped = map_points_to_letterbox([(0, 0)], meta)[0]
    px, py = int(round(mapped[0])), int(round(mapped[1]))
    assert crop[py, px, 1] == 255
    assert meta["orig_x1"] == 0 and meta["orig_y1"] == 0
