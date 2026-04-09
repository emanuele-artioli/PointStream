from __future__ import annotations

from pathlib import Path

from tests.video_utils import create_dummy_video


def test_encode_chunk_contract_and_events(mock_encoder_pipeline) -> None:
    project_root = Path(__file__).resolve().parents[1]
    video_path = create_dummy_video(
        path=project_root / "assets" / "test_chunks" / "enc001.mp4",
        num_frames=12,
        width=960,
        height=540,
        fps=25.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="enc001",
        start_frame_id=10,
    )

    assert payload.chunk.chunk_id == "enc001"
    assert payload.panorama.frame_width >= 960
    assert payload.panorama.frame_height >= 540
    assert len(payload.panorama.camera_poses) == 12
    assert len(payload.actors) == 2
    assert len(payload.rigid_objects) == 1
    assert payload.ball.object_id == "ball_0"
    assert payload.residual.codec == "libx265"

    for actor in payload.actors:
        assert actor.events
        for event in actor.events:
            assert event.object_id is not None
            assert event.frame_id >= 0


def test_execution_tags_are_propagated_to_dag_nodes(mock_encoder_pipeline) -> None:
    project_root = Path(__file__).resolve().parents[1]
    video_path = create_dummy_video(
        path=project_root / "assets" / "test_chunks" / "enc002.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )

    payload, _decoded = mock_encoder_pipeline.encode_video_file(
        video_path=video_path,
        chunk_id="enc002",
        start_frame_id=0,
    )
    context = mock_encoder_pipeline._dag.run(initial_context={"chunk": payload.chunk})

    assert context["chunk__tag"] == "cpu"
    assert context["panorama__tag"] == "cpu"
    assert context["actors__tag"] == "gpu"
    assert context["rigid_objects__tag"] == "gpu"
    assert context["ball__tag"] == "gpu"
    assert context["residual__tag"] == "gpu"
