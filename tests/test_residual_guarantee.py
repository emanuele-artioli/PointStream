from __future__ import annotations

from pathlib import Path

from src.encoder.orchestrator import EncoderPipeline
from src.shared.config import PointstreamConfig
from tests.mock_components import MockActorExtractor, ObjectTracker
from tests.video_utils import create_dummy_video


def _encode_residual(video_path: Path, chunk_id: str, panorama_jpeg_quality: int) -> Path:
    config = PointstreamConfig(panorama_jpeg_quality=panorama_jpeg_quality)
    pipeline = EncoderPipeline(
        config=config,
        actor_extractor=MockActorExtractor(),
        object_tracker=ObjectTracker(),
    )
    try:
        payload, _decoded = pipeline.encode_video_file(video_path=video_path, chunk_id=chunk_id)
    finally:
        pipeline.shutdown()
    return Path(payload.residual.residual_video_uri)


def test_panorama_jpeg_quality_changes_the_residual(test_run_artifacts_dir: Path) -> None:
    """Residual Guarantee regression: the encoder must synthesize its residual against the
    same codec-decoded panorama the client will reconstruct from, not raw pre-codec pixels.

    Before the fix, ResidualCalculator synthesized against payload.panorama.panorama_image's
    raw in-memory pixels regardless of panorama_jpeg_quality, so the residual (and thus what
    it corrects the client's JPEG-decoded panorama toward) never changed with the quality
    setting -- see reports/8_residual_guarantee_benchmarks_report.md, 2026-07-10 entry.
    """
    video_path = create_dummy_video(
        path=test_run_artifacts_dir / "test_chunks" / "residual_guarantee.mp4",
        num_frames=6,
        width=320,
        height=180,
        fps=30.0,
    )

    low_quality_residual = _encode_residual(video_path, chunk_id="rg_q50", panorama_jpeg_quality=50)
    high_quality_residual = _encode_residual(video_path, chunk_id="rg_q90", panorama_jpeg_quality=90)

    assert low_quality_residual.exists()
    assert high_quality_residual.exists()
    assert low_quality_residual.read_bytes() != high_quality_residual.read_bytes()
