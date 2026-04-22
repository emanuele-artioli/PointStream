from __future__ import annotations

import numpy as np

from src.encoder.actor_components import PayloadEncoder
from src.shared.mask_codec import decode_binary_mask
from src.shared.schemas import FrameState, SceneActor, VideoChunk


def _make_chunk() -> VideoChunk:
    return VideoChunk(
        chunk_id="maskmeta001",
        source_uri="memory://source",
        start_frame_id=100,
        fps=25.0,
        num_frames=2,
        width=64,
        height=48,
    )


def _make_pose(offset: float) -> list[list[float]]:
    pose = np.zeros((18, 3), dtype=np.float32)
    pose[:, 0] = np.linspace(10.0 + offset, 25.0 + offset, 18)
    pose[:, 1] = np.linspace(8.0, 34.0, 18)
    pose[:, 2] = 0.9
    return pose.tolist()


def _make_mask(height: int = 12, width: int = 8) -> list[list[int]]:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2 : height - 2, 2 : width - 2] = 1
    return mask.tolist()


def _make_polygons() -> list[list[list[float]]]:
    return [
        [[1.0, 1.0], [6.0, 1.0], [6.0, 10.0], [1.0, 10.0]],
    ]


def test_payload_encoder_emits_mask_metadata_when_enabled() -> None:
    chunk = _make_chunk()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="player_1",
                    class_name="player",
                    bbox=[5.0, 6.0, 21.0, 36.0],
                    mask=_make_mask(),
                    pose_dw=_make_pose(offset=0.0),
                )
            ],
        ),
        FrameState(
            frame_id=1,
            actors=[
                SceneActor(
                    track_id="player_1",
                    class_name="player",
                    bbox=[6.0, 6.0, 22.0, 36.0],
                    mask=_make_mask(),
                    pose_dw=_make_pose(offset=1.0),
                )
            ],
        ),
    ]

    packets = PayloadEncoder(
        pose_delta_threshold=0.0,
        include_mask_metadata=True,
        metadata_mask_codec="rle-v1",
    ).encode(
        chunk=chunk,
        frame_states=frame_states,
    )

    assert len(packets) == 1
    packet = packets[0]
    assert packet.object_id == "player_1"
    assert [frame.frame_id for frame in packet.mask_frames] == [100, 101]
    assert all(frame.source == "source" for frame in packet.mask_frames)
    assert all(frame.mask_codec == "rle-v1" for frame in packet.mask_frames)

    for frame in packet.mask_frames:
        assert frame.mask_payload is not None
        assert frame.mask_height is not None
        assert frame.mask_width is not None
        decoded = decode_binary_mask(
            codec=frame.mask_codec,
            payload=frame.mask_payload,
            height=int(frame.mask_height),
            width=int(frame.mask_width),
        )
        assert decoded.ndim == 2
        assert int(np.count_nonzero(decoded)) > 0


def test_payload_encoder_skips_mask_metadata_when_disabled() -> None:
    chunk = _make_chunk()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="player_1",
                    class_name="player",
                    bbox=[5.0, 6.0, 21.0, 36.0],
                    mask=_make_mask(),
                    pose_dw=_make_pose(offset=0.0),
                )
            ],
        )
    ]

    packets = PayloadEncoder(pose_delta_threshold=0.0, include_mask_metadata=False).encode(
        chunk=chunk,
        frame_states=frame_states,
    )

    assert len(packets) == 1
    assert packets[0].mask_frames == []


def test_payload_encoder_segmenter_native_prefers_polygon_codec() -> None:
    chunk = _make_chunk()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="player_1",
                    class_name="player",
                    bbox=[5.0, 6.0, 21.0, 36.0],
                    mask=_make_mask(),
                    mask_polygons=_make_polygons(),
                    pose_dw=_make_pose(offset=0.0),
                )
            ],
        )
    ]

    packets = PayloadEncoder(
        pose_delta_threshold=0.0,
        include_mask_metadata=True,
        metadata_mask_codec="segmenter-native",
    ).encode(
        chunk=chunk,
        frame_states=frame_states,
    )

    assert len(packets) == 1
    assert len(packets[0].mask_frames) == 1
    frame = packets[0].mask_frames[0]
    assert frame.mask_codec == "poly-v1"
    assert frame.mask_payload is not None
    assert frame.mask_height is not None
    assert frame.mask_width is not None

    decoded = decode_binary_mask(
        codec=frame.mask_codec,
        payload=frame.mask_payload,
        height=int(frame.mask_height),
        width=int(frame.mask_width),
    )
    assert int(np.count_nonzero(decoded)) > 0


def test_payload_encoder_segmenter_native_falls_back_to_binary_codec() -> None:
    chunk = _make_chunk()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="player_1",
                    class_name="player",
                    bbox=[5.0, 6.0, 21.0, 36.0],
                    mask=_make_mask(),
                    pose_dw=_make_pose(offset=0.0),
                )
            ],
        )
    ]

    packets = PayloadEncoder(
        pose_delta_threshold=0.0,
        include_mask_metadata=True,
        metadata_mask_codec="segmenter-native",
    ).encode(
        chunk=chunk,
        frame_states=frame_states,
    )

    assert len(packets) == 1
    assert len(packets[0].mask_frames) == 1
    frame = packets[0].mask_frames[0]
    assert frame.mask_codec in {"rle-v1", "bitpack-z1"}


def test_payload_encoder_accepts_yolo_native_alias() -> None:
    encoder = PayloadEncoder(
        pose_delta_threshold=0.0,
        include_mask_metadata=True,
        metadata_mask_codec="yolo-native",
    )
    assert encoder.metadata_mask_codec == "segmenter-native"
