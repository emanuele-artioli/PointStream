from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.encoder.video_io import probe_video_metadata
from src.shared.schemas import VideoChunk
from tests.video_utils import create_dummy_video


def test_encode_chunk_contract_and_events(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "enc001.mp4",
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
    assert len(payload.actor_references) == 2
    assert len(payload.rigid_objects) == 1
    assert payload.ball.object_id == "ball_0"
    assert len(payload.ball.states) == 12
    assert payload.residual.codec == "libsvtav1"

    for actor in payload.actors:
        assert actor.events
        for event in actor.events:
            assert event.object_id is not None
            assert event.frame_id >= 0


def test_execution_tags_are_propagated_to_dag_nodes(mock_encoder_pipeline, test_run_artifacts_dir: Path) -> None:
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "enc002.mp4",
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
    assert context["actor_references__tag"] == "cpu"
    assert context["rigid_objects__tag"] == "gpu"
    assert context["ball__tag"] == "gpu"
    assert context["residual__tag"] == "gpu"


def test_panorama_cache_reused_across_subchunks_in_same_scene(
    mock_encoder_pipeline, test_run_artifacts_dir: Path
) -> None:
    """Report 10 Phase 5.1(e): within one scene, BackgroundModeler.process()
    must run once and every subsequent sub-chunk reuses the cached,
    codec-round-tripped panorama packet -- not recompute it from scratch."""
    spy = MagicMock(wraps=mock_encoder_pipeline._background_modeler.process)
    mock_encoder_pipeline._background_modeler.process = spy

    video_a = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "panocache_a.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )
    video_b = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "panocache_b.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )

    mock_encoder_pipeline.set_scene_context("scene-0")
    payload_1, _ = mock_encoder_pipeline.encode_video_file(
        video_path=video_a, chunk_id="panocache_0001", start_frame_id=0
    )
    assert spy.call_count == 1

    # Second sub-chunk of the SAME scene: BackgroundModeler must not be
    # called again -- the cached packet (re-keyed to this chunk_id) is reused.
    payload_2, _ = mock_encoder_pipeline.encode_video_file(
        video_path=video_b, chunk_id="panocache_0002", start_frame_id=0
    )
    assert spy.call_count == 1
    assert payload_2.chunk.chunk_id == "panocache_0002"
    assert payload_2.panorama.chunk_id == "panocache_0002"
    assert payload_2.panorama.panorama_image == payload_1.panorama.panorama_image
    assert payload_2.panorama.panorama_codec_bytes == payload_1.panorama.panorama_codec_bytes

    # A new scene must invalidate the cache: the next encode recomputes.
    mock_encoder_pipeline.set_scene_context("scene-1")
    payload_3, _ = mock_encoder_pipeline.encode_video_file(
        video_path=video_b, chunk_id="panocache_0003", start_frame_id=0
    )
    assert spy.call_count == 2
    assert payload_3.panorama.chunk_id == "panocache_0003"


def test_panorama_cache_disabled_by_default_recomputes_every_chunk(
    mock_encoder_pipeline, test_run_artifacts_dir: Path
) -> None:
    """Callers that never call set_scene_context() (e.g. the single-chunk
    run_pipeline path) must keep today's per-chunk recompute behavior."""
    spy = MagicMock(wraps=mock_encoder_pipeline._background_modeler.process)
    mock_encoder_pipeline._background_modeler.process = spy

    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "panocache_nocache.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )

    mock_encoder_pipeline.encode_video_file(video_path=video_path, chunk_id="nc0001", start_frame_id=0)
    mock_encoder_pipeline.encode_video_file(video_path=video_path, chunk_id="nc0002", start_frame_id=0)

    assert spy.call_count == 2


def test_panorama_delta_layer_bypasses_scene_cache(
    mock_encoder_pipeline, test_run_artifacts_dir: Path
) -> None:
    """Reconciliation (report 10 Phase 5.1(e) vs 5.3, 2026-07-12): the
    full-packet scene cache added for panorama-static/roi-video must NOT
    short-circuit background_layer == "panorama-delta" -- that rung needs
    BackgroundModeler.process() to run on every sub-chunk so it can diff the
    true current panorama against the scene's previous one. If the cache
    shortcut ever regresses to also cover panorama-delta, this test catches
    it: BackgroundModeler would only run once per scene instead of once per
    sub-chunk, and the second sub-chunk's packet would come back byte-identical
    to the first (a permanent no-op delta) instead of marked "delta"."""
    mock_encoder_pipeline.config.background_layer = "panorama-delta"
    spy = MagicMock(wraps=mock_encoder_pipeline._background_modeler.process)
    mock_encoder_pipeline._background_modeler.process = spy

    video_a = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "panodelta_cache_a.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )
    video_b = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "panodelta_cache_b.mp4",
        num_frames=4,
        width=320,
        height=180,
        fps=30.0,
    )

    # encode_video_file() never sets VideoChunk.scene_id (only
    # match_orchestrator's point-scene loop does), so build chunks directly
    # and drive the DAG, matching test_execution_tags_are_propagated_to_dag_nodes.
    metadata_a = probe_video_metadata(video_a)
    metadata_b = probe_video_metadata(video_b)
    chunk_1 = VideoChunk(
        chunk_id="panodelta_0001",
        source_uri=str(video_a),
        start_frame_id=0,
        fps=metadata_a.fps,
        num_frames=metadata_a.num_frames,
        width=metadata_a.width,
        height=metadata_a.height,
        scene_id="scene-delta-0",
    )
    chunk_2 = VideoChunk(
        chunk_id="panodelta_0002",
        source_uri=str(video_b),
        start_frame_id=0,
        fps=metadata_b.fps,
        num_frames=metadata_b.num_frames,
        width=metadata_b.width,
        height=metadata_b.height,
        scene_id="scene-delta-0",
    )

    mock_encoder_pipeline.set_scene_context("scene-delta-0")
    context_1 = mock_encoder_pipeline._dag.run(initial_context={"chunk": chunk_1})
    assert spy.call_count == 1
    assert context_1["panorama"].panorama_mode == "full"

    # Second sub-chunk of the same scene: under panorama-delta this must
    # recompute (not hit the 5.1(e) cache) and come back marked "delta".
    context_2 = mock_encoder_pipeline._dag.run(initial_context={"chunk": chunk_2})
    assert spy.call_count == 2
    assert context_2["panorama"].panorama_mode == "delta"
