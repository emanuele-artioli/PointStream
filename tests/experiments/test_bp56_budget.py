"""BP56 budget persistence and refusal to expand after a crash gap."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tier import bp56_background_effort as batch
from experiments.tier.bp53_budget import (
    POINT_RESERVE_S,
    WALL_BUDGET_S,
    begin_attempt,
    heartbeat_attempt,
    load_budget,
    over_budget,
)
from src.contracts.config import PointstreamConfig
from src.runner import config_io


def test_unknown_crash_stops_before_another_encode(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    begin_attempt(budget_path, "bg-good4-crf51")
    heartbeat_attempt(budget_path, 12.0)
    from experiments.tier.bp53_budget import recover_interrupted

    recover_interrupted(budget_path, tmp_path / "points")
    assert load_budget(budget_path)["unknown_crash_interval"] is True


def test_over_budget_cannot_be_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "POINT_RESERVE_S", 1)
    monkeypatch.setattr(batch, "HEARTBEAT_INTERVAL_S", 3600.0)
    import experiments.tier.bp53_budget as budget_mod

    monkeypatch.setattr(budget_mod, "WALL_BUDGET_S", 100.0)

    class FakeSession:
        def __init__(self, path: Path, name: str, **kwargs: object) -> None:
            self.path = path
            self.name = name
            self.kind = str(kwargs.get("kind") or "attempt")

        def __enter__(self) -> FakeSession:
            begin_attempt(self.path, self.name, kind=self.kind)
            return self

        def elapsed(self) -> float:
            return 40.0 if self.kind == "attempt" else 0.0

        def __exit__(self, *exc: object) -> None:
            heartbeat_attempt(self.path, self.elapsed())

    monkeypatch.setattr(batch, "AttemptSession", FakeSession)
    monkeypatch.setattr(
        batch,
        "load_e1_sequence",
        lambda *a, **k: [SimpleNamespace(frames=np.zeros((2, 2, 2, 3), dtype=np.uint8))],
    )
    monkeypatch.setattr(
        batch,
        "_verify_input",
        lambda clips: [
            {"context_id": batch.EXPECTED_CONTEXT, "shape": batch.EXPECTED_SHAPE}
        ],
    )
    monkeypatch.setattr(batch, "_manifest_snapshot", lambda *a: {})
    monkeypatch.setattr(batch, "primary_preset", lambda codec: "0")
    monkeypatch.setattr(batch, "implementation_digest", lambda root=None: "test-code")
    monkeypatch.setattr(batch, "stream_codec_provenance", lambda *a, **k: {})
    monkeypatch.setattr(
        batch,
        "ffmpeg_provenance",
        lambda: {
            "path": batch.BP52_FFMPEG["path"],
            "version": batch.BP52_FFMPEG["version_prefix"],
        },
    )
    monkeypatch.setattr(
        batch,
        "run_prefix_proof",
        lambda dest: {
            "probe": {"supported": True},
            "families": {},
        },
    )
    monkeypatch.setattr(config_io, "load_tier", lambda name: PointstreamConfig())
    monkeypatch.setattr(
        batch, "run_metric_controls", lambda *a: {"valid": True, "alarms": []}
    )
    monkeypatch.setattr(batch, "apply_point", lambda *a, **k: PointstreamConfig())
    monkeypatch.setattr(batch, "_control_alarms", lambda payload: [])
    monkeypatch.setattr(batch, "_tool_identity", lambda root, preset: {
        "matches_bp52_ffmpeg": True,
        "reference_preset": "0",
        "reference_pix_fmt": "yuv420p",
        "metric_code": {},
        "metric_code_unchanged_from_origin_main": True,
    })

    def run_point(clips, tuned, **kwargs):
        _ = clips, tuned, kwargs
        return {
            "n_frames": 96,
            "usable": True,
            "is_rate": True,
            "coded_bytes": 474369,
            "parts": {
                "panorama": 445513,
                "actor_reference": 8599,
                "metadata": 20257,
                "residual": 0,
            },
            "scores": {
                "vmaf": 77.417052,
                "psnr_y": 33.003064,
                "ssim": 0.96694254,
            },
            "recovery_alarm": None,
            "late_frame": {"alarms": []},
        }

    monkeypatch.setattr(batch, "pointstream_e1", run_point)
    assert batch.main(["--out-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "background-effort.json").read_text())
    assert report["outcome"] != "complete"
    assert report["completion"]["succeeded"] == 3
    assert any("exceeds wall budget" in item for item in report["alarms"])
    assert over_budget(report["budget"]) is True
    assert POINT_RESERVE_S == 3500
    assert WALL_BUDGET_S == 8 * 3600
