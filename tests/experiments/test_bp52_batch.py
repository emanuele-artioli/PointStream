"""The bounded batch must persist failures and stop before the next encode."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tier import bp52_background_search as batch
from src.contracts.config import PointstreamConfig
from src.runner import config_io


@pytest.mark.parametrize("failure", [None, "point_alarm", "control_mismatch"])
def test_batch_stops_on_alarm_and_preserves_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(batch, "load_e1_sequence", lambda *a, **k: [
        SimpleNamespace(frames=np.zeros((2, 2, 2, 3), dtype=np.uint8))
    ])
    monkeypatch.setattr(batch, "_verify_input", lambda clips: [])
    monkeypatch.setattr(batch, "_manifest_snapshot", lambda *a: {})
    monkeypatch.setattr(batch, "primary_preset", lambda codec: "0")
    monkeypatch.setattr(batch, "implementation_digest", lambda: "test-code")
    monkeypatch.setattr(batch, "stream_codec_provenance", lambda codec: {})
    monkeypatch.setattr(batch, "ffmpeg_provenance", lambda: {})
    monkeypatch.setattr(config_io, "load_tier", lambda name: PointstreamConfig())
    monkeypatch.setattr(batch, "run_metric_controls", lambda *a: {"valid": True, "alarms": []})
    monkeypatch.setattr(batch, "_bp49_comparison", lambda row: {
        "status": "available", "delta_fresh_minus_historical": {
            "coded_bytes": 1 if failure == "control_mismatch" else 0,
            "vmaf": 0, "psnr_y": 0, "ssim": 0,
            "run_seconds": 500,  # Host timing variation must not fail the control.
        },
    })

    def run_point(clips, base, point, **kwargs):
        calls.append(point.name)
        return {"name": point.name, "pointstream": {
            "n_frames": 96, "usable": True, "is_rate": True,
            "coded_bytes": 100000, "parts": {"panorama": 100000},
            "scores": {"vmaf": 70.0, "psnr_y": 30.0, "ssim": 0.9},
            "recovery_alarm": "checkpoint gap exceeded" if failure == "point_alarm" else None,
            "late_frame": {"alarms": []},
            "background_payloads": [{"payload_bytes": point.stream_crf,
                                     "decoded_plate_sha256": str(point.stream_crf)}],
        }}

    monkeypatch.setattr(batch, "run_point", run_point)
    assert batch.main(["--out-dir", str(tmp_path)]) == (1 if failure else 0)
    report = json.loads((tmp_path / "background-search.json").read_text())
    assert calls == (["bg-crf51"] if failure else list(batch.POINT_NAMES))
    assert report["outcome"] == ("partial" if failure else "complete")
    assert bool(report["alarms"]) == bool(failure)
    assert (tmp_path / "points" / "bg-crf51.json").is_file()
    assert report["points"][0]["name"] == "bg-crf51"
