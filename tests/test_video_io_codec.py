from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.encoder.video_io as video_io


def test_ffmpeg_encoder_check_uses_exact_encoder_names(monkeypatch) -> None:
    ffmpeg_output = """
Encoders:
 V..... libsvtav1            SVT-AV1 encoder (codec av1)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V..... wrapped_avframe      AVFrame to AVPacket passthrough
 V..... foo_encoder          Description mentions definitely_not_a_codec but this is not its name
"""

    def fake_run(*args, **kwargs):
        _ = (args, kwargs)
        return SimpleNamespace(returncode=0, stdout=ffmpeg_output, stderr="")

    monkeypatch.setattr(video_io.subprocess, "run", fake_run)

    video_io._FFMPEG_ENCODER_CACHE.clear()
    video_io._assert_ffmpeg_encoder_available(ffmpeg_bin="ffmpeg-test-1", codec="libsvtav1")

    with pytest.raises(RuntimeError, match="definitely_not_a_codec"):
        video_io._assert_ffmpeg_encoder_available(
            ffmpeg_bin="ffmpeg-test-2",
            codec="definitely_not_a_codec",
        )
