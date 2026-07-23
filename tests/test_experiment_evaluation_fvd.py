from __future__ import annotations

from pathlib import Path

import torch

import src.shared.experiment_evaluation as experiment_evaluation
from src.shared.experiment_evaluation import _normalize_evaluation_metrics, evaluate_run_summary


def test_normalize_evaluation_metrics_accepts_fvd() -> None:
    assert _normalize_evaluation_metrics("fvd") == ["fvd"]
    assert _normalize_evaluation_metrics(["psnr", "fvd"]) == ["psnr", "fvd"]


def test_compute_fvd_missing_reference_video(tmp_path: Path) -> None:
    result = experiment_evaluation._compute_fvd(
        reference_video=tmp_path / "missing_reference.mp4",
        predicted_video=tmp_path / "missing_predicted.mp4",
    )

    assert result["fvd"] is None
    assert result["note"] == "missing reference video"


def test_compute_fvd_missing_predicted_video(tmp_path: Path) -> None:
    reference_video = tmp_path / "reference.mp4"
    reference_video.write_bytes(b"reference-bytes")

    result = experiment_evaluation._compute_fvd(
        reference_video=reference_video,
        predicted_video=tmp_path / "missing_predicted.mp4",
    )

    assert result["fvd"] is None
    assert result["note"] == "missing predicted artifact"


def test_evaluate_run_summary_wires_up_fvd(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "source.mp4"
    decoded_video = tmp_path / "decoded.mp4"
    source_video.write_bytes(b"source-bytes")
    decoded_video.write_bytes(b"decoded-bytes")

    monkeypatch.setattr(
        experiment_evaluation,
        "_compute_fvd",
        lambda reference_video, predicted_video, max_frames=None: {
            "fvd": 12.5,
            "fvd_num_clips_reference": 2,
            "fvd_num_clips_predicted": 2,
            "fvd_backbone": "i3d_r50_kinetics400",
            "note": None,
        },
    )

    evaluation = evaluate_run_summary(
        summary={
            "source_uri": str(source_video),
            "decoded_uri": str(decoded_video),
            "evaluation": {"sizes_bytes": {"source": 100, "transport_total": 60}},
        },
        experiment_dir=tmp_path,
        metrics=["fvd"],
    )

    assert evaluation["fvd"] == 12.5
    assert evaluation["fvd_num_clips_reference"] == 2
    assert evaluation["fvd_backbone"] == "i3d_r50_kinetics400"


def test_compute_fvd_success_path_delegates_to_fvd_module(monkeypatch, tmp_path: Path) -> None:
    reference_video = tmp_path / "reference.mp4"
    predicted_video = tmp_path / "predicted.mp4"
    reference_video.write_bytes(b"reference-bytes")
    predicted_video.write_bytes(b"predicted-bytes")

    class DummyDecoded:
        def __init__(self, tensor):
            self.tensor = tensor

    dummy_tensor = torch.arange(10)  # Shape: [Frames] stand-in, slice-able like a real frame tensor.

    def fake_decode(video_path):
        return DummyDecoded(dummy_tensor)

    captured = {}

    def fake_compute_fvd_from_frames(reference_frames, predicted_frames):
        captured["reference_frames"] = reference_frames
        captured["predicted_frames"] = predicted_frames
        return {
            "fvd": 3.0,
            "fvd_num_clips_reference": 1,
            "fvd_num_clips_predicted": 1,
            "fvd_backbone": "i3d_r50_kinetics400",
            "note": None,
        }

    monkeypatch.setattr(experiment_evaluation, "decode_video_to_tensor", fake_decode)
    monkeypatch.setattr(experiment_evaluation, "compute_fvd_from_frames", fake_compute_fvd_from_frames)

    result = experiment_evaluation._compute_fvd(
        reference_video=reference_video,
        predicted_video=predicted_video,
    )

    assert result["fvd"] == 3.0
    assert captured["reference_frames"] is dummy_tensor
    assert captured["predicted_frames"] is dummy_tensor


def test_compute_fvd_applies_max_frames_before_delegating(monkeypatch, tmp_path: Path) -> None:
    reference_video = tmp_path / "reference.mp4"
    predicted_video = tmp_path / "predicted.mp4"
    reference_video.write_bytes(b"reference-bytes")
    predicted_video.write_bytes(b"predicted-bytes")

    class DummyDecoded:
        def __init__(self, tensor):
            self.tensor = tensor

    def fake_decode(video_path):
        return DummyDecoded(torch.arange(10))

    captured = {}

    def fake_compute_fvd_from_frames(reference_frames, predicted_frames):
        captured["reference_len"] = len(reference_frames)
        captured["predicted_len"] = len(predicted_frames)
        return {"fvd": 1.0, "fvd_num_clips_reference": 1, "fvd_num_clips_predicted": 1, "note": None}

    monkeypatch.setattr(experiment_evaluation, "decode_video_to_tensor", fake_decode)
    monkeypatch.setattr(experiment_evaluation, "compute_fvd_from_frames", fake_compute_fvd_from_frames)

    experiment_evaluation._compute_fvd(
        reference_video=reference_video,
        predicted_video=predicted_video,
        max_frames=4,
    )

    assert captured["reference_len"] == 4
    assert captured["predicted_len"] == 4


def test_compute_fvd_surfaces_decode_failure_as_note(tmp_path: Path) -> None:
    reference_video = tmp_path / "reference.mp4"
    predicted_video = tmp_path / "predicted.mp4"
    # Not real video content, so decode_video_to_tensor will fail; _compute_fvd
    # must catch that and report it via `note` rather than raising, matching
    # the PSNR/SSIM/VMAF error-handling convention.
    reference_video.write_bytes(b"not-a-real-video")
    predicted_video.write_bytes(b"not-a-real-video-either")

    result = experiment_evaluation._compute_fvd(
        reference_video=reference_video,
        predicted_video=predicted_video,
    )

    assert result["fvd"] is None
    assert result["note"] is not None
    assert "fvd computation failed" in result["note"]
