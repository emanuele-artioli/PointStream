from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.encoder.mock_extractors import ActorExtractor, MockActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.video_io import (
    encode_video_frames_ffmpeg,
    iter_video_frames_ffmpeg,
    probe_video_metadata,
)


def _clean_generated_assets() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assets_dir = project_root / "assets"

    debug_actors_dir = assets_dir / "debug_actors"
    if debug_actors_dir.exists():
        for png_file in debug_actors_dir.glob("*.png"):
            png_file.unlink(missing_ok=True)

    for pano_file in assets_dir.glob("debug_panorama*.jpg"):
        pano_file.unlink(missing_ok=True)

    test_chunks_dir = assets_dir / "test_chunks"
    if test_chunks_dir.exists():
        for mp4_file in test_chunks_dir.glob("*.mp4"):
            mp4_file.unlink(missing_ok=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_assets_before_tests() -> None:
    _clean_generated_assets()


def _required_weight_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    weights_dir = project_root / "assets" / "weights"
    paths = {
        "detector": weights_dir / "yolo26n.pt",
        "segmenter": weights_dir / "yolo26n-seg.pt",
        "pose": weights_dir / "yolo26n-pose.pt",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        pytest.skip(f"Missing required actor weights in {weights_dir}: {', '.join(missing)}")
    return paths


@pytest.fixture(scope="session")
def yolo_model_bundle() -> dict[str, Any]:
    ultralytics = pytest.importorskip("ultralytics")
    yolo_ctor = ultralytics.YOLO
    paths = _required_weight_paths()
    return {
        "detector": yolo_ctor(str(paths["detector"])),
        "segmenter": yolo_ctor(str(paths["segmenter"])),
        "pose": yolo_ctor(str(paths["pose"])),
    }


@pytest.fixture()
def real_actor_extractor(yolo_model_bundle: dict[str, Any]) -> ActorExtractor:
    return ActorExtractor(
        detector_model=yolo_model_bundle["detector"],
        segmenter_model=yolo_model_bundle["segmenter"],
        pose_model=yolo_model_bundle["pose"],
        render_debug_keyframes=False,
    )


@pytest.fixture()
def real_encoder_pipeline(real_actor_extractor: ActorExtractor):
    pipeline = EncoderPipeline(actor_extractor=real_actor_extractor)
    try:
        yield pipeline
    finally:
        pipeline.shutdown()


@pytest.fixture()
def mock_encoder_pipeline():
    pipeline = EncoderPipeline(actor_extractor=MockActorExtractor())
    try:
        yield pipeline
    finally:
        pipeline.shutdown()


@pytest.fixture(scope="session")
def real_tennis_10f_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "assets" / "real_tennis.mp4"
    if not source_path.exists():
        pytest.skip("Expected test asset is missing: assets/real_tennis.mp4")

    metadata = probe_video_metadata(source_path)
    out_dir = tmp_path_factory.mktemp("integration_videos")
    out_path = out_dir / "real_tennis_10f.mp4"

    frames: list[Any] = []
    frame_count = 0
    for frame in iter_video_frames_ffmpeg(
        source_path,
        width=metadata.width,
        height=metadata.height,
    ):
        frames.append(frame)
        frame_count += 1
        if frame_count >= 10:
            break

    encode_video_frames_ffmpeg(
        output_path=out_path,
        frames_bgr=frames,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )
    if frame_count == 0:
        raise RuntimeError("real_tennis.mp4 produced zero decodable frames")

    return out_path
