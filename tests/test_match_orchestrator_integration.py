from __future__ import annotations

from pathlib import Path

import pytest

from src.encoder import match_orchestrator as mo
from src.encoder.video_io import probe_video_metadata
from src.shared.config import PointstreamConfig
from src.shared.schemas import SceneClass

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _required_actor_weights_present() -> bool:
    weights_dir = REPO_ROOT / "assets" / "weights"
    return all(
        (weights_dir / name).exists()
        for name in ("yolo26n.pt", "yolo26n-seg.pt", "yolo26n-pose.pt")
    )


@pytest.fixture(scope="module")
def real_tennis_point_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """~3.4s clip: long enough to clear classify_scene_stats' small-N
    fallback threshold (avg_score < 0.005 and duration > 3.0 ->
    cluster_point), so this smoke test actually exercises the semantic
    routing path, not just the fallback-only one."""
    source_path = REPO_ROOT / "assets" / "real_tennis.mp4"
    if not source_path.exists():
        pytest.skip("Expected test asset is missing: assets/real_tennis.mp4")

    import subprocess

    out_dir = tmp_path_factory.mktemp("match_orchestrator_integration")
    out_path = out_dir / "real_tennis_point_clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(source_path),
            "-t", "3.4", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-y", str(out_path),
        ],
        check=True,
    )
    return out_path


def test_encode_full_match_real_smoke(real_tennis_point_clip: Path, tmp_path: Path) -> None:
    """Real, non-mocked run of the report-10-Phase-2 full-match orchestrator:
    real ffmpeg scene-score extraction/classification, real yolo26n
    detector/pose/segmenter, real residual computation, real fallback-codec
    encode, and the outcome-safe routing comparison — end to end on a real
    (if short) clip, per CLAUDE.md's "verify with a real run" requirement.
    """
    if not _required_actor_weights_present():
        pytest.skip("Missing yolo26n* weights in assets/weights/")

    config = PointstreamConfig(
        source_uri=str(real_tennis_point_clip),
        num_frames=None,
        execution_pool="inline",
        detector="yolo26n.pt",
        pose_estimator="yolo26n-pose.pt",
        segmenter="yolo26n-seg.pt",
        genai_backend=None,
        log_level="warning",
        run_mode="full_match",
        scene_chunk_duration_sec=2.0,
    )

    transport_root = tmp_path / "transport"
    cache_root = tmp_path / "scene_cache"
    anchor_cache_root = tmp_path / "anchor_cache"

    summary = mo.encode_full_match(
        config=config,
        video_path=real_tennis_point_clip,
        transport_root=transport_root,
        scene_cache_root=cache_root,
        anchor_cache_root=anchor_cache_root,
    )

    source_metadata = probe_video_metadata(real_tennis_point_clip)

    # Structural invariants Phase 2 must guarantee regardless of which
    # class the real classifier assigns this clip (report 10 "G1 plumbing").
    assert summary["totals"]["num_scenes"] >= 1
    assert summary["totals"]["source_bytes"] > 0
    assert summary["totals"]["total_bytes"] > 0
    assert summary["num_frames_total"] == source_metadata.num_frames

    covered_frames = sum(scene["num_frames"] for scene in summary["scenes"])
    assert covered_frames > 0

    for scene in summary["scenes"]:
        assert scene["routing_summary"] in {"semantic", "fallback", "mixed"}
        assert scene["bytes"] >= 0
        if scene["scene_class"] == SceneClass.POINT.value:
            assert scene["sub_chunks"], "point scene must have attempted sub-chunks"
            for sub_chunk in scene["sub_chunks"]:
                assert sub_chunk["fallback_bytes"] > 0
                assert sub_chunk["semantic_bytes"] is not None and sub_chunk["semantic_bytes"] > 0
                assert sub_chunk["routing"] in {"semantic", "fallback"}
