"""E03B persistent bitstream wrapper, source-cache guard, and score-free eligibility."""

from __future__ import annotations

from unittest.mock import patch

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tier.campaign_result import ingest_for_claim, validate_campaign_record
from experiments.tier.e03b_confirmation import scene_bounds_from_mad
from experiments.tier.e03b_persist import persistent_timed_roundtrip, sha256_path
from experiments.tier.e03b_source import (
    BP46_EXTRACT,
    extraction_argv,
    materialize_display_low,
    nearest_native,
)
from src.components.codec.encode import EncodeRecord
from src.contracts.codecs import EncodeRequest, RateControl


def test_historical_extract_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="historical extraction cache"):
        materialize_display_low(video_path=tmp_path / "missing.mp4", run_dir=BP46_EXTRACT.parent)


def test_extraction_argv_is_ss_before_input() -> None:
    argv = extraction_argv(Path("/raw.mp4"), 80.7807, 4.0, Path("/new"), "/opt/local/bin/ffmpeg")
    assert argv[argv.index("-ss") + 1] == "80.780700"
    assert argv.index("-ss") < argv.index("-i")
    assert argv[argv.index("-r") + 1] == "24"


def test_probe_prefers_seconds_not_tick_timestamps() -> None:
    from experiments.tier.e03b_source import probe_native_frames

    payload = {
        "frames": [
            {"pkt_pts_time": None, "best_effort_timestamp": 6966960, "pkt_dts_time": "80.780767", "pict_type": "P", "key_frame": 0},
            {"pkt_pts_time": None, "best_effort_timestamp": 6968462, "pkt_dts_time": "80.797456", "pict_type": "P", "key_frame": 0},
        ]
    }
    with patch("experiments.tier.e03b_source.subprocess.check_output", return_value=json.dumps(payload)):
        frames = probe_native_frames(Path("/raw.mp4"), t_start=80.78, t_end=80.9, ffprobe="ffprobe")
    assert [row["native_pts_s"] for row in frames] == [80.780767, 80.797456]


def test_nearest_native_pts_is_not_the_24fps_grid() -> None:
    natives = [{"native_pts_s": 80.7801}, {"native_pts_s": 80.7968}]
    mapped = nearest_native(80.7807, natives)
    assert mapped["native_pts_s"] == 80.7801
    assert mapped["native_pts_s"] != 80.7807


def test_scene_bounds_from_mad_do_not_use_quality_scores() -> None:
    mads = np.array([0.4, 0.4, 8.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 9.0, 0.4], dtype=float)
    scenes = scene_bounds_from_mad(mads, fps=2.0, min_duration_s=2.0, factor=3.0)
    assert scenes
    assert all("psnr" not in row and "vmaf" not in row and "ssim" not in row for row in scenes)
    assert all(row["duration_s"] >= 2.0 for row in scenes)


def test_persistent_roundtrip_keeps_bitstream_and_command(tmp_path: Path) -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1] = 40
    encoder = tmp_path / "SvtAv1EncApp"
    ffmpeg = tmp_path / "ffmpeg"
    encoder.write_bytes(b"encoder-bin")
    ffmpeg.write_bytes(b"ffmpeg-bin")
    payload = b"charged-bitstream-bytes"

    def fake_encode(source, dest, request, **kwargs):
        dest.write_bytes(payload)
        return EncodeRecord(
            codec_name=request.codec_name,
            output=dest,
            size_bytes=len(payload),
            encode_seconds=0.2,
            tool_path=str(encoder),
            tool_version="SVT-AV1 v1.8.0 (release)",
            command=(str(encoder), "--qp", "63"),
            rate_control="qp",
            rate=request.rate,
            preset=request.preset,
            pix_fmt="yuv420p",
            roi_arm=None,
            ffmpeg_path=str(ffmpeg),
            ffmpeg_version="n7.1.1",
        )

    def fake_decode(bitstream, dest, request, **kwargs):
        dest.write_bytes(b"decoded-container")

    def fake_rgb(_ffmpeg_path: str, _video_path: Path, height: int, width: int, count: int) -> np.ndarray:
        return np.zeros((count, height, width, 3), dtype=np.uint8)

    request = EncodeRequest(
        codec_name="av1",
        rate_control=RateControl.QP,
        rate=63,
        preset="0",
        pix_fmt="yuv420p",
    )
    work = tmp_path / "persist"
    with (
        patch("experiments.tier.e03b_persist.encode", fake_encode),
        patch("experiments.tier.e03b_persist.decode", fake_decode),
        patch("experiments.tier.e03b_persist._rgb_dump", fake_rgb),
        patch("experiments.tier.e03b_persist._run_ffmpeg", lambda argv, stdin: b""),
        patch("experiments.tier.e03b_persist.tools.resolve_ffmpeg") as resolve,
    ):
        resolve.return_value.path = str(ffmpeg)
        result = persistent_timed_roundtrip(frames, request=request, fps=12.0, work_dir=work)

    assert result.bitstream_path.is_file()
    assert result.bitstream_path.read_bytes() == payload
    assert result.bitstream_sha256 == sha256_path(result.bitstream_path)
    assert result.ledger_matched is True
    assert result.encode_record["command"] == [str(encoder), "--qp", "63"]
    assert result.trip.size_bytes == len(payload)
    assert (work / "encode_record.json").is_file()
    assert (work / "decoded_rgb.npy").is_file()


def test_campaign_row_from_e03b_shape_validates() -> None:
    from experiments.tier.e03b_run import build_campaign_row

    class Trip:
        size_bytes = 48000
        frames = np.zeros((48, 360, 640, 3), dtype=np.uint8)
        encode_seconds = 12.0
        decode_seconds = 1.1
        tool_path = "/opt/local/bin/SvtAv1EncApp"
        tool_version = "SVT-AV1 v1.8.0 (release)"
        preset = "0"
        qp = 63

    class Persist:
        trip = Trip()
        bitstream_path = Path("/tmp/e03b-fake.ivf")
        bitstream_sha256 = "ab" * 32
        ledger_matched = True
        encode_record = {
            "command": ["enc"],
            "encoder_sha256": "cd" * 32,
            "ffmpeg_sha256": "ef" * 32,
        }
        standalone_shape = (48, 360, 640, 3)
        standalone_pixels_match = True

    row = build_campaign_row(
        setting={"codec": "av1", "qp": 63, "preset": "0"},
        recipe={"frame_ids": {"count": 48, "fps": 12.0, "native_timestamps": [80.78, 80.86]}},
        persistent=Persist(),
        scores={"psnr_y": 28.4, "ssim": 0.81, "vmaf": 55.0},
        decode_ok=True,
        calibration_id="/tmp/metric-calibration.json",
        encode_s=12.0,
        decode_s=1.1,
        score_s=2.0,
        host="gpu-test",
    )
    assert validate_campaign_record(row, purpose="validated") == []
    ingested = ingest_for_claim([row], "rd", purpose="validated")
    assert ingested["n_kept"] == 1


def test_short_rgb24_decode_is_rejected_not_padded() -> None:
    from experiments.tier.e03b_persist import DecodeCountError, frames_from_rgb24

    raw = bytes(2 * 8 * 8 * 3)
    with pytest.raises(DecodeCountError, match="decoded 2 frames, expected 48"):
        frames_from_rgb24(raw, width=8, height=8, expected_count=48, source="repro")


def test_empty_and_partial_rgb24_decodes_are_rejected() -> None:
    from experiments.tier.e03b_persist import DecodeCountError, frames_from_rgb24

    with pytest.raises(DecodeCountError, match="empty decode"):
        frames_from_rgb24(b"", width=8, height=8, expected_count=2, source="empty")
    with pytest.raises(DecodeCountError, match="partial frame"):
        frames_from_rgb24(b"\x00\x01", width=8, height=8, expected_count=1, source="partial")
    extra = bytes(3 * 8 * 8 * 3)
    with pytest.raises(DecodeCountError, match="decoded 3 frames, expected 2"):
        frames_from_rgb24(extra, width=8, height=8, expected_count=2, source="extra")


def test_exact_rgb24_count_reshapes() -> None:
    from experiments.tier.e03b_persist import frames_from_rgb24

    raw = bytes(i % 256 for i in range(2 * 8 * 8 * 3))
    frames = frames_from_rgb24(raw, width=8, height=8, expected_count=2, source="ok")
    assert frames.shape == (2, 8, 8, 3)


def test_refuse_overwrite_on_existing_probe_report(tmp_path: Path) -> None:
    from experiments.tier.e03b_run import refuse_overwrite

    (tmp_path / "probe_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refuse_overwrite(tmp_path)


def test_prepared_reuse_rejects_hash_mismatch(tmp_path: Path) -> None:
    from experiments.tier.e03b_run import load_prepared_reuse
    from experiments.tier.e03b_source import EXPECTED_SHAPE, stack_sha256

    frames = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
    np.save(tmp_path / "prepared_rgb.npy", frames)
    recipe = {
        "prepared_sha256": "0" * 64,
        "extraction": {"interpolation": False, "selected_positions": list(range(0, 96, 2))},
    }
    (tmp_path / "source_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared SHA-256"):
        load_prepared_reuse(tmp_path)
    assert stack_sha256(frames) != recipe["prepared_sha256"]


def test_materialize_refuses_existing_prepared(tmp_path: Path) -> None:
    (tmp_path / "source_recipe.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite prepared source"):
        materialize_display_low(video_path=tmp_path / "missing.mp4", run_dir=tmp_path)


def test_mismatched_standalone_pixels_fail_closed(tmp_path: Path) -> None:
    from experiments.tier.e03b_persist import DecodeCountError

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    encoder = tmp_path / "SvtAv1EncApp"
    ffmpeg = tmp_path / "ffmpeg"
    encoder.write_bytes(b"encoder-bin")
    ffmpeg.write_bytes(b"ffmpeg-bin")
    payload = b"charged-bitstream-bytes"
    calls = {"n": 0}

    def fake_encode(source, dest, request, **kwargs):
        dest.write_bytes(payload)
        return EncodeRecord(
            codec_name=request.codec_name,
            output=dest,
            size_bytes=len(payload),
            encode_seconds=0.2,
            tool_path=str(encoder),
            tool_version="SVT-AV1 v1.8.0 (release)",
            command=(str(encoder), "--qp", "63"),
            rate_control="qp",
            rate=request.rate,
            preset=request.preset,
            pix_fmt="yuv420p",
            roi_arm=None,
            ffmpeg_path=str(ffmpeg),
            ffmpeg_version="n7.1.1",
        )

    def fake_decode(bitstream, dest, request, **kwargs):
        dest.write_bytes(b"decoded-container")

    def fake_rgb(_ffmpeg_path: str, _video_path: Path, height: int, width: int, count: int) -> np.ndarray:
        calls["n"] += 1
        out = np.zeros((count, height, width, 3), dtype=np.uint8)
        if calls["n"] == 2:
            out[...] = 9
        return out

    request = EncodeRequest(
        codec_name="av1",
        rate_control=RateControl.QP,
        rate=63,
        preset="0",
        pix_fmt="yuv420p",
    )
    with (
        patch("experiments.tier.e03b_persist.encode", fake_encode),
        patch("experiments.tier.e03b_persist.decode", fake_decode),
        patch("experiments.tier.e03b_persist._rgb_dump", fake_rgb),
        patch("experiments.tier.e03b_persist._run_ffmpeg", lambda argv, stdin: b""),
        patch("experiments.tier.e03b_persist.tools.resolve_ffmpeg") as resolve,
    ):
        resolve.return_value.path = str(ffmpeg)
        with pytest.raises(DecodeCountError, match="ordinary and standalone"):
            persistent_timed_roundtrip(frames, request=request, fps=12.0, work_dir=tmp_path / "persist")
