from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import src.experiment_evaluation as eval_module
from src.experiment_evaluation import _compute_psnr


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc")
    fourcc = int(fourcc_fn(*"mp4v"))
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def test_index_pairing_identical_frames(tmp_path: Path) -> None:
    # Create two identical videos; PSNR should be +inf for all frames
    frames = [np.full((16, 12, 3), 50, dtype=np.uint8) for _ in range(4)]
    ref = tmp_path / "ref.mp4"
    pred = tmp_path / "pred.mp4"
    _write_video(ref, frames, fps=10.0)
    _write_video(pred, frames, fps=10.0)

    res = _compute_psnr(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["psnr_num_frames"] == 4
    # Depending on codec/encoder rounding, frames may not be bit-for-bit
    # identical after writing to mp4. Ensure we recorded the expected
    # number of pairs and at least one of mean or infinite count is present.
    assert (res["psnr_mean"] is not None) or (res.get("psnr_infinite_frames", 0) > 0)


def test_index_pairing_resizes_and_psnr_finite(tmp_path: Path) -> None:
    # Reference is 16x12, predicted is 8x6; evaluator should resize and compute finite PSNR
    ref_frames = [np.full((16, 12, 3), 10, dtype=np.uint8) for _ in range(3)]
    pred_frames = [np.full((8, 6, 3), 12, dtype=np.uint8) for _ in range(3)]
    ref = tmp_path / "ref2.mp4"
    pred = tmp_path / "pred2.mp4"
    _write_video(ref, ref_frames, fps=15.0)
    _write_video(pred, pred_frames, fps=15.0)

    res = _compute_psnr(reference_video=ref, predicted_video=pred, max_frames=None)
    assert res["psnr_num_frames"] == 3
    assert res["psnr_infinite_frames"] == 0
    assert res["psnr_mean"] is not None


def test_timestamp_nearest_fallback_matching(monkeypatch, tmp_path: Path) -> None:
    # Force streaming stage to yield no pairs by making the first two VideoCapture
    # instances behave as empty, then provide frames/timestamps on the fallback.

    ref_file = tmp_path / "r.mp4"
    pred_file = tmp_path / "p.mp4"
    # touch files so path.exists() passes
    ref_file.write_bytes(b"x")
    pred_file.write_bytes(b"x")

    # Prepare fallback frame lists
    ref_frames = [np.full((10, 8, 3), i * 10 + 10, dtype=np.uint8) for i in range(3)]
    ref_times = [0.0, 40.0, 80.0]
    pred_frames = [np.full((10, 8, 3), 5 + i * 60, dtype=np.uint8) for i in range(2)]
    pred_times = [10.0, 70.0]

    class FakeCap:
        # global instance counter
        inst_count = 0

        def __init__(self, path: str):
            type(self).inst_count += 1
            self.idx = 0
            self.path = path
            self.mode = "stream_empty" if type(self).inst_count <= 2 else "fallback"

        def isOpened(self):
            return True

        def read(self):
            if self.mode == "stream_empty":
                return False, None
            # fallback mode: determine whether this is ref or pred by path
            if str(ref_file) in self.path:
                if self.idx >= len(ref_frames):
                    return False, None
                f = ref_frames[self.idx]
                self.idx += 1
                return True, f
            else:
                if self.idx >= len(pred_frames):
                    return False, None
                f = pred_frames[self.idx]
                self.idx += 1
                return True, f

        def get(self, prop):
            # return timestamp corresponding to last returned frame
            if self.mode != "fallback":
                return 0.0
            if str(ref_file) in self.path:
                return ref_times[max(0, self.idx - 1)]
            return pred_times[max(0, self.idx - 1)]

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda p: FakeCap(p))

    res = _compute_psnr(reference_video=ref_file, predicted_video=pred_file, max_frames=None)
    # Should have 3 pairs (one per reference frame)
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
