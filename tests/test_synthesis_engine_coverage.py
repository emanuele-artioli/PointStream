from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from src.shared import synthesis_engine as se
from src.shared.schemas import (
    ActorPacket,
    BallPacket,
    BallState,
    CameraPose,
    EncodedChunkPayload,
    KeyframeEvent,
    ObjectClass,
    PanoramaPacket,
    ResidualPacket,
    SemanticEvent,
    TensorSpec,
    VideoChunk,
)


def _make_payload(
    *,
    num_frames: int = 5,
    start_frame_id: int = 10,
    homographies: list[list[list[float]]] | None = None,
    panorama_image: np.ndarray | None = None,
    panorama_uri: str = "memory://panorama",
    ball_states: list[BallState] | None = None,
    ball_events: Sequence[SemanticEvent] | None = None,
    actor_events: Sequence[SemanticEvent] | None = None,
) -> EncodedChunkPayload:
    chunk = VideoChunk(
        chunk_id="syn001",
        source_uri="memory://source",
        start_frame_id=start_frame_id,
        fps=25.0,
        num_frames=num_frames,
        width=32,
        height=24,
    )

    identity = np.eye(3, dtype=np.float32).tolist()
    homographies = homographies if homographies is not None else [identity for _ in range(num_frames)]

    camera_poses = [
        CameraPose(frame_id=start_frame_id + i, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        for i in range(num_frames)
    ]
    panorama = PanoramaPacket(
        chunk_id=chunk.chunk_id,
        panorama_uri=panorama_uri,
        frame_width=chunk.width,
        frame_height=chunk.height,
        camera_poses=camera_poses,
        panorama_image=(np.zeros((24, 32, 3), dtype=np.uint8) if panorama_image is None else panorama_image).tolist()
        if panorama_image is not None or panorama_uri.startswith("memory://")
        else None,
        homography_matrices=homographies,
        selected_frame_indices=list(range(num_frames)),
    )

    default_actor_events: list[SemanticEvent] = [
        KeyframeEvent(
            frame_id=start_frame_id,
            object_id="person_0",
            object_class=ObjectClass.PERSON,
            coordinates=np.zeros((18, 3), dtype=np.float32).reshape(-1).tolist(),
        )
    ]
    actor = ActorPacket(
        chunk_id=chunk.chunk_id,
        object_id="person_0",
        appearance_embedding_spec=TensorSpec(name="appearance", shape=[1, 2, 256], dtype="torch.float32"),
        pose_tensor_spec=TensorSpec(name="pose", shape=[1, num_frames, 18, 3], dtype="torch.float32"),
        events=default_actor_events if actor_events is None else list(actor_events),
    )

    ball = BallPacket(
        chunk_id=chunk.chunk_id,
        object_id="ball_0",
        trajectory_spec=TensorSpec(name="ball", shape=[1, num_frames, 4], dtype="torch.float32"),
        events=[] if ball_events is None else list(ball_events),
        states=[] if ball_states is None else ball_states,
    )

    residual = ResidualPacket(chunk_id=chunk.chunk_id, codec="libx265", residual_video_uri="memory://residual")

    return EncodedChunkPayload(
        chunk=chunk,
        panorama=panorama,
        actors=[actor],
        actor_references=[],
        rigid_objects=[],
        ball=ball,
        residual=residual,
    )


def test_synthesis_engine_builds_diffusers_when_genai_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyDiffusersCompositor:
        def __init__(self, seed: int, device: torch.device) -> None:
            self.seed = seed
            self.device = device

    monkeypatch.setenv("POINTSTREAM_ENABLE_GENAI", "1")
    monkeypatch.setattr(se, "DiffusersCompositor", _DummyDiffusersCompositor)

    engine = se.SynthesisEngine(seed=2026, device="cpu")
    compositor = engine.get_genai_compositor()

    assert isinstance(compositor, _DummyDiffusersCompositor)
    assert compositor.seed == 2026


def test_resolve_panorama_image_errors_for_invalid_shapes_and_missing_files(tmp_path: Path) -> None:
    engine = se.SynthesisEngine(device="cpu")

    payload_bad_shape = _make_payload()
    payload_bad_shape.panorama.panorama_image = np.zeros((24, 32), dtype=np.uint8).tolist()
    with pytest.raises(ValueError, match="Invalid panorama image shape"):
        engine._resolve_panorama_image(payload_bad_shape)

    payload_missing = _make_payload(panorama_uri=str(tmp_path / "missing_panorama.jpg"))
    payload_missing.panorama.panorama_image = None
    with pytest.raises(FileNotFoundError, match="Panorama image not found"):
        engine._resolve_panorama_image(payload_missing)


def test_collect_keyframes_and_interpolation_mask_branches() -> None:
    engine = se.SynthesisEngine(device="cpu")

    invalid_actor = ActorPacket(
        chunk_id="syn001",
        object_id="person_0",
        appearance_embedding_spec=TensorSpec(name="a", shape=[1], dtype="torch.float32"),
        pose_tensor_spec=TensorSpec(name="p", shape=[1], dtype="torch.float32"),
        events=[
            KeyframeEvent(
                frame_id=10,
                object_id="person_0",
                object_class=ObjectClass.PERSON,
                coordinates=[0.0] * 10,
            )
        ],
    )
    with pytest.raises(ValueError, match="expected 54 values"):
        engine._collect_keyframes(invalid_actor, start_frame=10, frame_count=4)

    no_interp_actor = ActorPacket(
        chunk_id="syn001",
        object_id="person_0",
        appearance_embedding_spec=TensorSpec(name="a", shape=[1], dtype="torch.float32"),
        pose_tensor_spec=TensorSpec(name="p", shape=[1], dtype="torch.float32"),
        events=[],
    )
    mask = engine._collect_interpolate_mask(no_interp_actor, start_frame=10, frame_count=4)
    assert bool(torch.all(mask))


def test_unroll_ball_states_from_events_interpolates_and_holds_tail() -> None:
    engine = se.SynthesisEngine(device="cpu")

    events = [
        KeyframeEvent(
            frame_id=10,
            object_id="ball_0",
            object_class=ObjectClass.BALL,
            coordinates=[0.2, 0.3, 0.1, 0.0],
        ),
        KeyframeEvent(
            frame_id=12,
            object_id="ball_0",
            object_class=ObjectClass.BALL,
            coordinates=[0.6, 0.7, 0.0, 0.2],
        ),
    ]
    payload = _make_payload(num_frames=5, start_frame_id=10, ball_events=events, ball_states=[])

    dense = engine._unroll_ball_states(payload)

    assert len(dense) == 5
    assert dense[0].is_visible
    assert dense[1].is_visible
    assert dense[3].x == pytest.approx(dense[2].x)
    assert dense[4].y == pytest.approx(dense[2].y)


def test_unroll_ball_states_from_states_propagates_last_visible_position() -> None:
    engine = se.SynthesisEngine(device="cpu")

    states = [
        BallState(frame_id=10, ball_x=14.0, ball_y=9.0, velocity_x=1.0, velocity_y=0.5, is_visible=True),
        BallState(frame_id=11, ball_x=0.0, ball_y=0.0, velocity_x=0.0, velocity_y=0.0, is_visible=False),
    ]
    payload = _make_payload(num_frames=3, start_frame_id=10, ball_states=states)

    dense = engine._unroll_ball_states(payload)

    assert dense[1].x == pytest.approx(14.0)
    assert dense[1].y == pytest.approx(9.0)
    assert dense[1].is_visible is False


def test_draw_motion_blurred_ball_handles_visibility_and_coordinate_modes() -> None:
    engine = se.SynthesisEngine(device="cpu")
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    engine._draw_motion_blurred_ball(
        frame,
        se._BallRenderState(x=0.5, y=0.5, vx=0.1, vy=0.08, is_visible=True),
    )
    assert int(np.max(frame)) > 0

    before = frame.copy()
    engine._draw_motion_blurred_ball(
        frame,
        se._BallRenderState(x=22.0, y=16.0, vx=4.0, vy=2.0, is_visible=True),
    )
    assert not np.array_equal(before, frame)

    unchanged = frame.copy()
    engine._draw_motion_blurred_ball(
        frame,
        se._BallRenderState(x=10.0, y=8.0, vx=1.0, vy=1.0, is_visible=False),
    )
    assert np.array_equal(unchanged, frame)


def test_reconstruct_background_frames_handles_homography_padding_and_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = se.SynthesisEngine(device="cpu")

    def _warp_perspective(src, M, dsize, mode, padding_mode, align_corners):
        _ = (M, mode, padding_mode, align_corners)
        return torch.full((src.shape[0], 3, dsize[0], dsize[1]), 0.5, dtype=src.dtype, device=src.device)

    fake_kornia = types.SimpleNamespace(
        geometry=types.SimpleNamespace(transform=types.SimpleNamespace(warp_perspective=_warp_perspective))
    )
    monkeypatch.setitem(sys.modules, "kornia", fake_kornia)

    identity = np.eye(3, dtype=np.float32).tolist()
    payload_padded = _make_payload(num_frames=4, homographies=[identity, identity])
    out_padded = engine._reconstruct_background_frames(payload_padded)
    assert tuple(out_padded.shape) == (4, 3, 24, 32)

    payload_trimmed = _make_payload(num_frames=2, homographies=[identity, identity, identity])
    out_trimmed = engine._reconstruct_background_frames(payload_trimmed)
    assert tuple(out_trimmed.shape) == (2, 3, 24, 32)


def test_reconstruct_background_frames_requires_kornia(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = se.SynthesisEngine(device="cpu")
    monkeypatch.setitem(sys.modules, "kornia", None)

    with pytest.raises(RuntimeError, match="kornia is required"):
        engine._reconstruct_background_frames(_make_payload())


def test_reconstruct_background_frames_rejects_empty_homography_list(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = se.SynthesisEngine(device="cpu")

    def _warp_perspective(src, M, dsize, mode, padding_mode, align_corners):
        _ = (M, mode, padding_mode, align_corners)
        return torch.zeros((src.shape[0], 3, dsize[0], dsize[1]), dtype=src.dtype, device=src.device)

    fake_kornia = types.SimpleNamespace(
        geometry=types.SimpleNamespace(transform=types.SimpleNamespace(warp_perspective=_warp_perspective))
    )
    monkeypatch.setitem(sys.modules, "kornia", fake_kornia)

    with pytest.raises(ValueError, match="no homography matrices"):
        engine._reconstruct_background_frames(_make_payload(homographies=[]))
