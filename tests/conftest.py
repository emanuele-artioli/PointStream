from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.shared.video_io import (
    encode_video_frames_ffmpeg,
    iter_video_frames_ffmpeg,
    probe_video_metadata,
)


def _create_test_run_artifacts_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "outputs" / "tests" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture(scope="session")
def test_run_artifacts_dir() -> Path:
    return _create_test_run_artifacts_dir()


@pytest.fixture(scope="session", autouse=True)
def configure_test_debug_artifact_env(test_run_artifacts_dir: Path):
    previous = os.environ.get("POINTSTREAM_DEBUG_ARTIFACT_DIR")
    os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = str(test_run_artifacts_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POINTSTREAM_DEBUG_ARTIFACT_DIR", None)
        else:
            os.environ["POINTSTREAM_DEBUG_ARTIFACT_DIR"] = previous


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


@pytest.fixture(scope="session")
def real_tennis_20f_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "assets" / "real_tennis.mp4"
    if not source_path.exists():
        pytest.skip("Expected test asset is missing: assets/real_tennis.mp4")

    metadata = probe_video_metadata(source_path)
    out_dir = tmp_path_factory.mktemp("integration_videos")
    out_path = out_dir / "real_tennis_20f.mp4"

    frames: list[Any] = []
    frame_count = 0
    for frame in iter_video_frames_ffmpeg(
        source_path,
        width=metadata.width,
        height=metadata.height,
    ):
        frames.append(frame)
        frame_count += 1
        if frame_count >= 20:
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
    if frame_count < 20:
        pytest.skip("assets/real_tennis.mp4 has fewer than 20 decodable frames")

    return out_path
