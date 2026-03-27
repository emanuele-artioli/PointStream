from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from ultralytics import YOLO

from src.encoder.mock_extractors import ActorExtractor, MockActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata


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
def yolo_model_bundle() -> dict[str, YOLO]:
    paths = _required_weight_paths()
    return {
        "detector": YOLO(str(paths["detector"])),
        "segmenter": YOLO(str(paths["segmenter"])),
        "pose": YOLO(str(paths["pose"])),
    }


@pytest.fixture()
def real_actor_extractor(yolo_model_bundle: dict[str, YOLO]) -> ActorExtractor:
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

    writer = cv2.VideoWriter(
        str(out_path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create clipped integration video at: {out_path}")

    frame_count = 0
    for frame in iter_video_frames_ffmpeg(
        source_path,
        width=metadata.width,
        height=metadata.height,
    ):
        writer.write(frame)
        frame_count += 1
        if frame_count >= 10:
            break

    writer.release()
    if frame_count == 0:
        raise RuntimeError("real_tennis.mp4 produced zero decodable frames")

    return out_path
