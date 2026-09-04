"""BP53 wall-budget persistence, crash windows, and exhaustion."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tier import bp53_background_scale as batch
from experiments.tier.bp53_budget import (
    POINT_RESERVE_S,
    WALL_BUDGET_S,
    AttemptSession,
    begin_attempt,
    finish_attempt,
    heartbeat_attempt,
    load_budget,
    longer_runs_cleared,
    over_budget,
    reconcile_checkpoints,
    reconstruct_from_events,
    recover_interrupted,
    remaining_seconds,
)
from experiments.tier.low_rate_checkpoint import save_checkpoint
from src.contracts.config import PointstreamConfig
from src.runner import config_io


def test_consumed_seconds_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    begin_attempt(path, "metric-controls", kind="controls")
    finish_attempt(path, "metric-controls", 12.5, kind="controls")
    begin_attempt(path, "first")
    finish_attempt(path, "first", 100.0)
    restarted = load_budget(path)
    assert restarted["consumed_seconds"] == 112.5
    assert remaining_seconds(restarted) == WALL_BUDGET_S - 112.5
    begin_attempt(path, "second")
    finish_attempt(path, "second", 3.0)
    assert load_budget(path)["consumed_seconds"] == 115.5


def test_idle_between_processes_is_not_consumed(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    begin_attempt(path, "a")
    finish_attempt(path, "a", 50.0)
    before = load_budget(path)["consumed_seconds"]
    after = load_budget(path)["consumed_seconds"]
    assert after == before


def test_gap_within_one_second_of_hourly_limit_does_not_clear_longer_runs() -> None:
    assert longer_runs_cleared(3599.72) is False
    assert longer_runs_cleared(3599.01) is False
    assert longer_runs_cleared(3599.0) is True
    assert longer_runs_cleared(3598.9) is True
    assert POINT_RESERVE_S == 3500


def test_reconstructed_native_walls_do_not_reset_and_do_not_clear(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"
    state = reconstruct_from_events(
        path,
        [
            {"kind": "attempt", "name": "bg-scale1-crf51", "seconds": 4099.392},
            {"kind": "attempt", "name": "bg-scale05-crf51", "seconds": 6242.19},
            {"kind": "attempt", "name": "bg-scale05-crf63", "seconds": 4039.722},
        ],
        max_checkpoint_gap_seconds=3599.7179984620307,
        note="controls duration was not stored",
    )
    assert state["consumed_seconds"] == 4099.392 + 6242.19 + 4039.722
    assert state["longer_runs_operationally_cleared"] is False
    assert state["consumed_seconds_is_lower_bound"] is True
    assert remaining_seconds(state) < WALL_BUDGET_S
    reloaded = load_budget(path)
    assert reloaded["consumed_seconds"] == state["consumed_seconds"]


def test_interrupted_work_flags_unknown_crash_interval(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    begin_attempt(path, "bg-scale05-crf51")
    heartbeat_attempt(path, 8.0)
    recover_interrupted(path, tmp_path / "points")
    state = load_budget(path)
    assert state["consumed_seconds"] == 8.0
    assert state["unknown_crash_interval"] is True
    assert state["consumed_seconds_is_lower_bound"] is True
    assert state["active_attempt"] is None
    kinds = [item["kind"] for item in state["events"]]
    assert kinds == ["interrupted"]
    recover_interrupted(path, tmp_path / "points")
    assert load_budget(path)["consumed_seconds"] == 8.0

    path2 = tmp_path / "budget-no-heartbeat.json"
    begin_attempt(path2, "bg-scale05-crf63")
    recover_interrupted(path2, tmp_path / "points")
    crashed = load_budget(path2)
    assert crashed["consumed_seconds"] == 0.0
    assert crashed["unknown_crash_interval"] is True
    assert crashed["consumed_seconds_is_lower_bound"] is True


def test_checkpoint_charge_crash_window_is_reconciled_once(tmp_path: Path) -> None:
    budget_path = tmp_path / "budget.json"
    points_dir = tmp_path / "points"
    begin_attempt(budget_path, "bg-scale1-crf51")
    heartbeat_attempt(budget_path, 10.0)
    save_checkpoint(
        points_dir,
        "bg-scale1-crf51",
        {"name": "bg-scale1-crf51", "attempt_wall_seconds": 12.0},
    )
    recover_interrupted(budget_path, points_dir)
    first = reconcile_checkpoints(budget_path, points_dir, ["bg-scale1-crf51"])
    second = reconcile_checkpoints(budget_path, points_dir, ["bg-scale1-crf51"])
    finish_attempt(budget_path, "bg-scale1-crf51", 12.0)
    state = load_budget(budget_path)
    assert first["consumed_seconds"] == 12.0
    assert second["consumed_seconds"] == 12.0
    assert state["consumed_seconds"] == 12.0
    assert state["unknown_crash_interval"] is False
    attempt_events = [item for item in state["events"] if item["kind"] == "attempt"]
    assert len(attempt_events) == 1
    assert attempt_events[0]["seconds"] == 12.0


def test_over_budget_final_point_cannot_be_complete(
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

    monkeypatch.setattr(batch, "load_e1_sequence", lambda *a, **k: [
        SimpleNamespace(frames=np.zeros((2, 2, 2, 3), dtype=np.uint8))
    ])
    monkeypatch.setattr(
        batch,
        "_verify_input",
        lambda clips: [
            {"context_id": batch.EXPECTED_CONTEXT, "shape": batch.EXPECTED_SHAPE}
        ],
    )
    monkeypatch.setattr(batch, "_manifest_snapshot", lambda *a: {})
    monkeypatch.setattr(batch, "primary_preset", lambda codec: "0")
    monkeypatch.setattr(batch, "implementation_digest", lambda: "test-code")
    monkeypatch.setattr(batch, "stream_codec_provenance", lambda codec: {})
    monkeypatch.setattr(
        batch,
        "ffmpeg_provenance",
        lambda: {
            "path": batch.BP52_FFMPEG["path"],
            "version": batch.BP52_FFMPEG["version_prefix"],
        },
    )
    monkeypatch.setattr(config_io, "load_tier", lambda name: PointstreamConfig())
    monkeypatch.setattr(
        batch, "run_metric_controls", lambda *a: {"valid": True, "alarms": []}
    )
    monkeypatch.setattr(batch, "apply_point", lambda *a, **k: PointstreamConfig())
    monkeypatch.setattr(batch, "_control_alarms", lambda payload: [])

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
    report = json.loads((tmp_path / "background-scale.json").read_text())
    assert report["outcome"] != "complete"
    assert report["completion"]["succeeded"] == 3
    assert any("exceeds wall budget" in item for item in report["alarms"])
    assert over_budget(report["budget"]) is True


def test_attempt_session_writes_active_record_before_work(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    seen: list[object] = []

    class JumpClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 1.0
            return self.t

    with AttemptSession(path, "probe", interval_s=3600.0, clock=JumpClock()) as session:
        seen.append(load_budget(path)["active_attempt"]["name"])
        session.pulse()
    assert seen == ["probe"]
    active = load_budget(path)["active_attempt"]
    assert active is not None
    assert active["last_elapsed_seconds"] >= 1.0
    finish_attempt(path, "probe", active["last_elapsed_seconds"])
    assert load_budget(path)["active_attempt"] is None
