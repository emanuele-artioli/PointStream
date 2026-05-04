from __future__ import annotations

from pathlib import Path

import src.experiment_evaluation as experiment_evaluation


def test_evaluate_run_summary_combines_timings_and_savings(monkeypatch, tmp_path: Path) -> None:
    source_video = tmp_path / "source.mp4"
    decoded_video = tmp_path / "decoded.mp4"
    source_video.write_bytes(b"source-bytes")
    decoded_video.write_bytes(b"decoded-bytes")

    monkeypatch.setattr(
        experiment_evaluation,
        "_compute_psnr",
        lambda reference_video, predicted_video, max_frames=None: {
            "psnr_mean": 38.5,
            "psnr_std": 0.25,
            "psnr_num_frames": 4,
            "note": None,
        },
    )

    evaluation = experiment_evaluation.evaluate_run_summary(
        summary={
            "source_uri": str(source_video),
            "decoded_uri": str(decoded_video),
            "source_size_bytes": 100,
            "transport_total_size_bytes": 60,
            "pipeline_total_sec": 2.0,
            "encode_chunk_sec": 1.0,
            "transport_send_sec": 0.2,
            "transport_receive_sec": 0.1,
            "decode_sec": 0.3,
        },
        experiment_dir=tmp_path,
        max_frames=8,
    )

    assert evaluation["experiment_dir"] == str(tmp_path)
    assert evaluation["reference_video_size_bytes"] == 100
    assert evaluation["decoded_video_size_bytes"] == len(b"decoded-bytes")
    assert evaluation["transport_savings_percent"] == 40.0
    assert evaluation["decoded_vs_reference_percent"] == 87.0
    assert evaluation["pipeline_total_sec"] == 2.0
    assert evaluation["psnr_mean"] == 38.5
    assert evaluation["psnr_num_frames"] == 4