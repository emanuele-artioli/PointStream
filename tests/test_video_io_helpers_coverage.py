from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

import src.encoder.video_io as video_io


def test_video_io_helper_parsers_and_presets() -> None:
    assert video_io._parse_ffprobe_fps("30000/1001") > 29.9
    assert video_io._parse_ffprobe_fps("25") == 25.0
    assert video_io._parse_optional_float("1.5") == 1.5
    assert video_io._parse_optional_float(None) == 0.0
    assert video_io._parse_optional_int("4") == 4
    assert video_io._parse_optional_int("bad") == 0

    encoder_names = video_io._parse_ffmpeg_encoder_names(
        """
Encoders:
 V..... libsvtav1            SVT-AV1 encoder (codec av1)
 V....D libx264              libx264 H.264 / AVC
 V..... wrapped_avframe      AVFrame passthrough
"""
    )
    assert encoder_names == {"libsvtav1", "libx264", "wrapped_avframe"}

    assert video_io._normalize_preset_for_codec("libx265", "veryfast") == "veryfast"
    assert video_io._normalize_preset_for_codec("libsvtav1", "10") == "10"
    assert video_io._normalize_preset_for_codec("libsvtav1", "slower") == "5"
    assert video_io._normalize_preset_for_codec("libsvtav1", "unknown") == "8"
    assert video_io._normalize_preset_for_codec("libsvtav1", None) is None


def test_video_io_resolve_binary_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FFMPEG_BIN", "/opt/ffmpeg")
    assert video_io._resolve_binary_path("FFMPEG_BIN", "ffmpeg") == "/opt/ffmpeg"

    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr(video_io.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert video_io._resolve_binary_path("FFMPEG_BIN", "ffmpeg") == "/usr/bin/ffmpeg"

    monkeypatch.setattr(video_io.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        video_io._resolve_binary_path("FFMPEG_BIN", "ffmpeg")


def test_encode_video_frames_ffmpeg_rejects_bad_shapes_and_empty_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(video_io, "_resolve_binary_path", lambda *args, **kwargs: "ffmpeg")
    monkeypatch.setattr(video_io, "_assert_ffmpeg_encoder_available", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_io, "_normalize_preset_for_codec", lambda codec, preset: None)

    class _FakePipe:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()

        def wait(self):
            return 0

    monkeypatch.setattr(video_io.subprocess, "Popen", lambda *args, **kwargs: _FakePipe())

    with pytest.raises(ValueError, match="Unexpected frame shape"):
        video_io.encode_video_frames_ffmpeg(
            output_path=tmp_path / "bad.mp4",
            frames_bgr=[np.zeros((2, 2, 3), dtype=np.uint8)],
            fps=30.0,
            width=3,
            height=3,
            codec="libsvtav1",
        )

    with pytest.raises(ValueError, match="zero frames"):
        video_io.encode_video_frames_ffmpeg(
            output_path=tmp_path / "empty.mp4",
            frames_bgr=[],
            fps=30.0,
            width=2,
            height=2,
            codec="libsvtav1",
        )
