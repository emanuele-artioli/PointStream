from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import src.experiment_evaluation as eval_module
from src.encoder.video_io import encode_video_frames_ffmpeg
from src.experiment_evaluation import _compute_psnr, _compute_ssim_ffmpeg, _compute_vmaf_ffmpeg


def _make_frames(num_frames: int, width: int, height: int, value: int) -> list[np.ndarray]:
    return [np.full((height, width, 3), value, dtype=np.uint8) for _ in range(num_frames)]


def _ffmpeg_has_libvmaf() -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    result = subprocess.run([ffmpeg_bin, "-hide_banner", "-filters"], capture_output=True, text=True, check=False)
    return "libvmaf" in result.stdout


# Ubuntu's apt ffmpeg package (e.g. CI runners) is not reliably built with
# libvmaf; skip rather than fail so VMAF-dependent tests don't block CI on an
# environment gap unrelated to the code under test.
_HAS_LIBVMAF = _ffmpeg_has_libvmaf()


def test_identical_videos_yield_high_or_infinite_psnr(tmp_path: Path) -> None:
    frames = _make_frames(4, width=32, height=24, value=50)
    ref = tmp_path / "ref.mp4"
    pred = tmp_path / "pred.mp4"
    encode_video_frames_ffmpeg(ref, frames, fps=10.0, width=32, height=24, codec="libx264")
    encode_video_frames_ffmpeg(pred, frames, fps=10.0, width=32, height=24, codec="libx264")

    res = _compute_psnr(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["note"] is None
    assert res["psnr_num_frames"] == 4
    assert (res["psnr_mean"] is not None) or (res.get("psnr_infinite_frames", 0) > 0)


def test_mismatched_dimensions_are_scaled_and_psnr_is_finite(tmp_path: Path) -> None:
    ref_frames = _make_frames(3, width=32, height=24, value=10)
    pred_frames = _make_frames(3, width=16, height=12, value=200)
    ref = tmp_path / "ref2.mp4"
    pred = tmp_path / "pred2.mp4"
    encode_video_frames_ffmpeg(ref, ref_frames, fps=15.0, width=32, height=24, codec="libx264")
    encode_video_frames_ffmpeg(pred, pred_frames, fps=15.0, width=16, height=12, codec="libx264")

    res = _compute_psnr(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["note"] is None
    assert res["psnr_num_frames"] == 3
    assert res["psnr_infinite_frames"] == 0
    assert res["psnr_mean"] is not None


def test_mismatched_dimensions_are_scaled_and_ssim_is_computed(tmp_path: Path) -> None:
    """SSIM is a framesync filter like PSNR: ffmpeg requires equal input
    resolutions and errors out on mismatch without an explicit scale step.
    """
    ref_frames = _make_frames(3, width=32, height=24, value=10)
    pred_frames = _make_frames(3, width=16, height=12, value=200)
    ref = tmp_path / "ref_ssim.mp4"
    pred = tmp_path / "pred_ssim.mp4"
    encode_video_frames_ffmpeg(ref, ref_frames, fps=15.0, width=32, height=24, codec="libx264")
    encode_video_frames_ffmpeg(pred, pred_frames, fps=15.0, width=16, height=12, codec="libx264")

    res = _compute_ssim_ffmpeg(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["note"] is None
    assert res["ssim_num_frames"] == 3
    assert res["ssim_mean"] is not None


@pytest.mark.skipif(not _HAS_LIBVMAF, reason="ffmpeg on PATH lacks the libvmaf filter")
def test_mismatched_dimensions_are_scaled_and_vmaf_is_computed(tmp_path: Path) -> None:
    """VMAF is a framesync filter like PSNR: ffmpeg requires equal input
    resolutions and errors out on mismatch without an explicit scale step.

    Reference dims must exceed libvmaf's ADM minimum (>32px each side) since
    both streams are compared at the (scaled-to-)reference resolution.
    """
    ref_frames = _make_frames(3, width=64, height=48, value=10)
    pred_frames = _make_frames(3, width=32, height=24, value=200)
    ref = tmp_path / "ref_vmaf.mp4"
    pred = tmp_path / "pred_vmaf.mp4"
    encode_video_frames_ffmpeg(ref, ref_frames, fps=15.0, width=64, height=48, codec="libx264")
    encode_video_frames_ffmpeg(pred, pred_frames, fps=15.0, width=32, height=24, codec="libx264")

    res = _compute_vmaf_ffmpeg(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["note"] is None
    assert res["vmaf_num_frames"] == 3
    assert res["vmaf_mean"] is not None


def test_av1_encoded_predicted_video_is_readable(tmp_path: Path) -> None:
    """Regression test for the real-run bug: opencv-python's bundled ffmpeg
    lacks an AV1 decoder, so cv2.VideoCapture opens a libsvtav1-encoded
    decoder output (isOpened() True) but read() immediately returns False —
    every frame-pairing strategy built on cv2 silently finds zero pairs.
    SSIM/VMAF already avoid this by shelling out to the system ffmpeg
    binary; PSNR must use the same approach so AV1 chunks are readable.
    """
    # SVT-AV1 refuses to encode below 64x64.
    frames = _make_frames(3, width=64, height=64, value=80)
    ref = tmp_path / "ref3.mp4"
    pred = tmp_path / "pred3.mp4"
    encode_video_frames_ffmpeg(ref, frames, fps=10.0, width=64, height=64, codec="libx264")
    encode_video_frames_ffmpeg(pred, frames, fps=10.0, width=64, height=64, codec="libsvtav1")

    res = _compute_psnr(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["note"] is None
    assert res["psnr_num_frames"] == 3


def test_evaluate_run_summary_dispatches_requested_metrics(monkeypatch, tmp_path: Path) -> None:
    summary = {
        "source_uri": str(tmp_path / "source.mp4"),
        "decoded_uri": str(tmp_path / "decoded.mp4"),
        "source_size_bytes": 10,
        "transport_total_size_bytes": 4,
    }
    (tmp_path / "source.mp4").write_bytes(b"source")
    (tmp_path / "decoded.mp4").write_bytes(b"decoded")

    calls: list[str] = []

    def _fake_psnr(**kwargs):
        calls.append("psnr")
        return {"psnr_mean": 30.0, "psnr_std": 0.0, "psnr_num_frames": 1, "note": None}

    def _fake_ssim(**kwargs):
        calls.append("ssim")
        return {"ssim_mean": 0.9, "ssim_std": 0.0, "ssim_num_frames": 1, "note": None}

    def _fake_vmaf(**kwargs):
        calls.append("vmaf")
        return {"vmaf_mean": 95.0, "vmaf_num_frames": 1, "note": None}

    monkeypatch.setattr(eval_module, "_compute_psnr", _fake_psnr)
    monkeypatch.setattr(eval_module, "_compute_ssim_ffmpeg", _fake_ssim)
    monkeypatch.setattr(eval_module, "_compute_vmaf_ffmpeg", _fake_vmaf)

    result = eval_module.evaluate_run_summary(
        summary=summary,
        experiment_dir=tmp_path,
        max_frames=2,
        metrics=["psnr", "ssim", "vmaf"],
    )

    assert calls == ["psnr", "ssim", "vmaf"]
    assert result["psnr_mean"] == 30.0
    assert result["ssim_mean"] == 0.9
    assert result["vmaf_mean"] == 95.0
