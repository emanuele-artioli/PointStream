from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.ball_extractor import BallExtractor
from src.encoder.mock_extractors import MockActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg
from src.shared.schemas import CameraPose, FrameState, PanoramaPacket, VideoChunk
from src.transport.disk import DiskTransport


def test_ball_extractor_tracks_visible_motion(test_run_artifacts_dir: Path) -> None:
    pytest.importorskip("kornia")

    num_frames = 12
    width = 96
    height = 64
    fps = 30.0
    video_path = test_run_artifacts_dir / "test_chunks" / "ball_extract.mp4"
    expected_positions = _create_moving_ball_video(
        path=video_path,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=fps,
    )

    chunk = VideoChunk(
        chunk_id="ball_extract_0001",
        source_uri=str(video_path),
        start_frame_id=0,
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
    )
    panorama = _build_static_panorama(chunk=chunk)
    frame_states = [FrameState(frame_id=idx, actors=[]) for idx in range(num_frames)]

    extractor = BallExtractor(difference_threshold=10.0, min_blob_area=4, device="cpu")
    packet = extractor.process(chunk=chunk, panorama=panorama, frame_states=frame_states)

    assert len(packet.states) == num_frames
    visible_states = [state for state in packet.states if state.is_visible]
    assert len(visible_states) >= 10
    assert any(state.velocity_x > 0.0 for state in packet.states[1:] if state.is_visible)
    assert all(frame_state.ball_state is not None for frame_state in frame_states)

    for frame_idx, state in enumerate(packet.states):
        if not state.is_visible:
            continue
        expected_x, expected_y = expected_positions[frame_idx]
        assert abs(state.ball_x - expected_x) <= 8.0
        assert abs(state.ball_y - expected_y) <= 8.0


def test_end_to_end_reconstruction_contains_tracked_ball(test_run_artifacts_dir: Path) -> None:
    pytest.importorskip("kornia")

    num_frames = 12
    width = 96
    height = 64
    fps = 30.0
    chunk_id = "ball_e2e_0001"
    video_path = test_run_artifacts_dir / "test_chunks" / "ball_e2e.mp4"
    expected_positions = _create_moving_ball_video(
        path=video_path,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=fps,
    )

    pipeline = EncoderPipeline(actor_extractor=MockActorExtractor())
    try:
        payload, _decoded = pipeline.encode_video_file(
            video_path=video_path,
            chunk_id=chunk_id,
            start_frame_id=0,
        )
    finally:
        pipeline.shutdown()

    with tempfile.TemporaryDirectory() as tmp_dir:
        transport = DiskTransport(root_dir=tmp_dir)
        transport.send(payload)
        recovered_payload = transport.receive(chunk_id)

        output_path = test_run_artifacts_dir / "debug_final_reconstruction_ball.mp4"
        output_path.unlink(missing_ok=True)
        decoded = DecoderRenderer(output_root=test_run_artifacts_dir).process(
            recovered_payload,
            output_path=output_path,
        )

    assert decoded.output_uri == str(output_path)
    assert output_path.exists()
    assert len(recovered_payload.ball.states) == num_frames

    visible_states = [state for state in recovered_payload.ball.states if state.is_visible]
    assert len(visible_states) >= 6

    reconstructed_frames = list(
        iter_video_frames_ffmpeg(
            output_path,
            width=width,
            height=height,
        )
    )
    assert len(reconstructed_frames) == num_frames

    highlight_hits = 0
    for state in visible_states:
        local_idx = int(state.frame_id) - int(recovered_payload.chunk.start_frame_id)
        if local_idx < 0 or local_idx >= len(reconstructed_frames):
            continue

        expected_x, expected_y = expected_positions[local_idx]
        if abs(state.ball_x - expected_x) > 10.0 or abs(state.ball_y - expected_y) > 10.0:
            continue

        frame = reconstructed_frames[local_idx]
        px, py = _state_to_pixel(float(state.ball_x), float(state.ball_y), width=width, height=height)
        y0 = max(0, py - 3)
        y1 = min(height, py + 4)
        x0 = max(0, px - 3)
        x1 = min(width, px + 4)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if int(np.max(patch[:, :, 1])) > 120 and int(np.max(patch[:, :, 2])) > 120:
            highlight_hits += 1

    assert highlight_hits >= max(3, len(visible_states) // 4)


def _create_moving_ball_video(
    path: Path,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
) -> list[tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center_x = min(width - 5, 12 + frame_idx * 5)
        center_y = min(height - 5, 18 + frame_idx * 2)
        cv2.circle(frame, (center_x, center_y), 3, (90, 235, 250), thickness=-1, lineType=cv2.LINE_AA)
        frames.append(frame)
        positions.append((center_x, center_y))

    encode_video_frames_ffmpeg(
        output_path=path,
        frames_bgr=frames,
        fps=fps,
        width=width,
        height=height,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )
    return positions


def _build_static_panorama(chunk: VideoChunk) -> PanoramaPacket:
    frame_count = int(chunk.num_frames)
    identity = np.eye(3, dtype=np.float32).tolist()

    camera_poses = [
        CameraPose(
            frame_id=chunk.start_frame_id + frame_idx,
            tx=0.0,
            ty=0.0,
            tz=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        )
        for frame_idx in range(frame_count)
    ]

    return PanoramaPacket(
        chunk_id=chunk.chunk_id,
        panorama_uri="memory://panorama/static",
        frame_width=chunk.width,
        frame_height=chunk.height,
        camera_poses=camera_poses,
        panorama_image=np.zeros((chunk.height, chunk.width, 3), dtype=np.uint8).tolist(),
        homography_matrices=[identity for _ in range(frame_count)],
        selected_frame_indices=list(range(frame_count)),
    )


def _state_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if 0.0 <= x <= 1.2 and 0.0 <= y <= 1.2:
        px = int(round(x * float(width - 1)))
        py = int(round(y * float(height - 1)))
    else:
        px = int(round(x))
        py = int(round(y))

    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    return px, py
