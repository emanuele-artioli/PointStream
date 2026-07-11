"""End-to-end proof of the HNeRV harness: train -> serialize -> measure bytes ->
decode -> score, on a tiny synthetic clip. Touches real ffmpeg I/O (encoding the
synthetic source video, and evaluate_run_summary's ffmpeg-based PSNR), so this
is integration+slow, not a fast unit test — see tests/test_hnerv_baseline.py and
tests/test_hnerv_arch.py for the fast unit coverage of the same pieces in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import hnerv_baseline as hnerv
from src.encoder.video_io import encode_video_frames_ffmpeg

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _write_tiny_synthetic_video(path, num_frames: int, height: int, width: int) -> None:
    rng = np.random.default_rng(seed=0)

    def _frames():
        for _ in range(num_frames):
            yield rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    encode_video_frames_ffmpeg(
        path,
        _frames(),
        fps=10.0,
        width=width,
        height=height,
        codec="libx264",
        crf=10,
        preset="ultrafast",
    )


def test_hnerv_baseline_end_to_end_on_tiny_synthetic_clip(tmp_path) -> None:
    source_video = tmp_path / "tiny_source.mp4"
    _write_tiny_synthetic_video(source_video, num_frames=4, height=32, width=32)

    output_root = tmp_path / "hnerv_out"
    exit_code = hnerv.main(
        [
            "--input",
            str(source_video),
            "--height",
            "16",
            "--width",
            "16",
            "--embed-height",
            "2",
            "--embed-width",
            "2",
            "--embed-channels",
            "8",
            "--strides",
            "2,2,2",
            "--channels",
            "16,8,4",
            "--epochs",
            "30",
            "--lr",
            "0.01",
            "--log-every",
            "10",
            "--checkpoint-every",
            "0",
            "--device",
            "cpu",
            "--metrics",
            "psnr",
            "--output-root",
            str(output_root),
        ]
    )
    assert exit_code == 0

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    checkpoint_path = run_dir / "hnerv_checkpoint.pt.gz"
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0

    decoded_dir = run_dir / "decoded_frames"
    decoded_frames = sorted(decoded_dir.glob("frame_*.png"))
    assert len(decoded_frames) == 4

    report_json = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["output_bytes"] == checkpoint_path.stat().st_size
    # A few training steps on a tiny overfit target should already beat pure noise.
    assert report_json["evaluation"]["psnr_mean"] is not None
    assert report_json["evaluation"]["psnr_mean"] > 5.0

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| HNeRV |" in report_md

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) > 0
    assert history[0]["epoch"] == 0
