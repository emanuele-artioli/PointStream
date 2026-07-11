"""Fast unit tests for scripts/hnerv_baseline.py's CLI parsing and report rendering.

No real video I/O or GPU training here — see tests/test_hnerv_baseline_integration.py
for the end-to-end (real ffmpeg, real training loop) proof of the harness.
"""

from __future__ import annotations

from pathlib import Path

from scripts import hnerv_baseline as hnerv
from src.shared.hnerv_arch import HNeRVConfig


def test_arg_parser_defaults() -> None:
    args = hnerv._build_arg_parser().parse_args([])
    assert args.input is None
    assert args.height == hnerv.DEFAULT_HEIGHT
    assert args.width == hnerv.DEFAULT_WIDTH
    assert args.embed_height == hnerv.DEFAULT_EMBED_HEIGHT
    assert args.embed_width == hnerv.DEFAULT_EMBED_WIDTH
    assert args.embed_channels == hnerv.DEFAULT_EMBED_CHANNELS
    assert args.strides == ",".join(str(s) for s in hnerv.DEFAULT_STRIDES)
    assert args.channels == ",".join(str(c) for c in hnerv.DEFAULT_CHANNELS)
    assert args.epochs == hnerv.DEFAULT_EPOCHS
    assert args.metrics == "psnr,ssim,vmaf"


def test_arg_parser_overrides() -> None:
    args = hnerv._build_arg_parser().parse_args(
        ["--height", "16", "--width", "32", "--strides", "2,2", "--channels", "8,4", "--epochs", "10"]
    )
    assert args.height == 16
    assert args.width == 32
    assert args.strides == "2,2"
    assert args.channels == "8,4"
    assert args.epochs == 10


def test_fmt_ratio_and_quality_helpers() -> None:
    assert hnerv._fmt_ratio(50, 100) == "0.5000"
    assert hnerv._fmt_ratio(50, 0) == "—"
    assert hnerv._fmt_quality(1.23456) == "1.235"
    assert hnerv._fmt_quality(None) == "null"


def test_build_report_renders_expected_table_and_payload() -> None:
    config = HNeRVConfig(
        height=8, width=8, embed_height=2, embed_width=2, embed_channels=4, strides=(2, 2), channels=(8, 4)
    )
    markdown, payload = hnerv.build_report(
        input_path=Path("assets/real_tennis.mp4"),
        config=config,
        epochs=10,
        decoder_params=12345,
        output_bytes=999,
        source_bytes=1998,
        evaluation={"psnr_mean": 30.111, "ssim_mean": 0.9, "vmaf_mean": 88.4},
        train_seconds=12.3,
        decode_seconds=0.5,
    )
    assert "| HNeRV | 999 | 0.5000 | 30.111 | 0.900 | 88.400 | 12 | 0.5 |" in markdown
    assert payload["output_bytes"] == 999
    assert payload["source_bytes"] == 1998
    assert payload["config"]["strides"] == [2, 2]
    assert payload["evaluation"]["psnr_mean"] == 30.111
