"""Cheap health checks must not become repeated model invocations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import monitor


def setup_job(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    monitor.write_json(
        tmp_path / "policy.json",
        {
            "report_at": None,
            "repeat_seconds": None,
            "quiet_until": 0,
            "urgent_during_quiet": False,
            "stall_seconds": 100,
            **overrides,
        },
    )
    return {"status": "running", "started": 0, "last_progress": 0, "emitted": []}


def events(tmp_path: Path) -> list[dict[str, Any]]:
    return [monitor.read_json(p) for p in (tmp_path / "events").glob("*.json")]


def test_bedtime_digest_is_one_shot_then_hourly_schedule_can_replace_it(tmp_path: Path) -> None:
    state = setup_job(tmp_path, report_at=800, quiet_until=800)
    monitor.tick(tmp_path, state, 799)
    assert events(tmp_path) == []
    monitor.tick(tmp_path, state, 800)
    assert sum(e["kind"].startswith("report:") for e in events(tmp_path)) == 1
    monitor.tick(tmp_path, state, 900)
    assert sum(e["kind"].startswith("report:") for e in events(tmp_path)) == 1
    policy = monitor.read_json(tmp_path / "policy.json")
    policy.update(report_at=1000, repeat_seconds=100, quiet_until=0)
    monitor.write_json(tmp_path / "policy.json", policy)
    monitor.tick(tmp_path, state, 1000)
    # Reload persisted state as a restarted supervisor would.
    state = monitor.read_json(tmp_path / "status.json")
    monitor.tick(tmp_path, state, 1099)
    assert sum(e["kind"].startswith("report:") for e in events(tmp_path)) == 2
    monitor.tick(tmp_path, state, 1100)
    assert sum(e["kind"].startswith("report:") for e in events(tmp_path)) == 3
    policy.update(report_at=None, repeat_seconds=None)
    monitor.write_json(tmp_path / "policy.json", policy)
    monitor.tick(tmp_path, state, 5000)
    assert sum(e["kind"].startswith("report:") for e in events(tmp_path)) == 3


def test_heartbeat_timestamp_does_not_hide_stalled_work(tmp_path: Path) -> None:
    state = setup_job(tmp_path)
    progress = {"stage": "train", "completed": 1, "updated": 1, "decision": None}
    monitor.write_json(tmp_path / "progress.json", progress)
    monitor.tick(tmp_path, state, 1)
    progress["updated"] = 99
    monitor.write_json(tmp_path / "progress.json", progress)
    monitor.tick(tmp_path, state, 101)
    monitor.tick(tmp_path, state, 102)
    assert [e["kind"] for e in events(tmp_path)] == ["stall:1"]
    progress.update(completed=2, updated=103)
    monitor.write_json(tmp_path / "progress.json", progress)
    monitor.tick(tmp_path, state, 103)
    monitor.tick(tmp_path, state, 202)
    assert len(events(tmp_path)) == 1


def test_republishing_same_decision_does_not_wake_again(tmp_path: Path) -> None:
    state = setup_job(tmp_path)
    for now in (1, 2, 3):
        monitor.write_json(
            tmp_path / "progress.json",
            {
                "stage": "pilot",
                "completed": 1,
                "updated": now,
                "decision": "spacing unresolved",
            },
        )
        monitor.tick(tmp_path, state, now)
    assert len(events(tmp_path)) == 1


@pytest.mark.parametrize("urgent", [False, True])
def test_quiet_hours_apply_to_failure_and_decision(tmp_path: Path, urgent: bool) -> None:
    state = setup_job(tmp_path, report_at=50, quiet_until=100, urgent_during_quiet=urgent)
    state["status"] = "failed"
    state["progress"] = {"stage": "pilot", "decision": "invalid evidence", "updated": 1}
    kinds = monitor.due_events(state, monitor.read_json(tmp_path / "policy.json"), 50)
    assert len(kinds) == (2 if urgent else 0)
    assert not any(k.startswith("report:") for k in kinds)


def test_delivery_batches_events_retries_then_acknowledges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = setup_job(tmp_path, report_at=1)
    state["status"] = "complete"
    monitor.tick(tmp_path, state, 1)
    calls = []
    now = [1.0]
    monkeypatch.setattr(monitor.time, "time", lambda: now[0])

    def queue(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0, stderr="offline")

    monkeypatch.setattr(monitor.subprocess, "run", queue)
    monitor.deliver(tmp_path, "owner", "/bin/codex")
    assert not list((tmp_path / "events").glob("*.sent"))
    monitor.deliver(tmp_path, "owner", "/bin/codex")
    assert len(calls) == 1  # backoff without another model call
    now[0] = 100
    monitor.deliver(tmp_path, "owner", "/bin/codex")
    assert len(calls) == 2
    assert calls[-1][:4] == ["/bin/codex", "queue", "--thread", "owner"]
    assert "terminal:complete" in calls[-1][-1] and "report:1" in calls[-1][-1]
    assert len(list((tmp_path / "events").glob("*.sent"))) == 2
    monitor.deliver(tmp_path, "owner", "/bin/codex")
    assert len(calls) == 2


def test_event_replay_reuses_identity_after_state_write_loss(tmp_path: Path) -> None:
    state = setup_job(tmp_path, report_at=1)
    monitor.tick(tmp_path, state, 1)
    first = {e["id"] for e in events(tmp_path)}
    state["emitted"] = []  # event file survived, status update did not
    monitor.tick(tmp_path, state, 1)
    assert {e["id"] for e in events(tmp_path)} == first


def test_supervisor_restart_never_relaunches_unknown_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = setup_job(tmp_path)
    monitor.write_json(tmp_path / "status.json", state)
    monitor.write_json(tmp_path / "request.json", {"thread": None, "codex": "codex"})

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("restart must not spawn a possibly duplicate job")

    monkeypatch.setattr(monitor.subprocess, "Popen", forbidden)
    assert monitor.supervise(tmp_path) == 0
    assert monitor.read_json(tmp_path / "status.json")["status"] == "interrupted"


def test_supervisor_enforces_budget_without_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_job(tmp_path)
    monitor.write_json(
        tmp_path / "request.json",
        {
            "command": ["synthetic-worker"],
            "cwd": str(tmp_path),
            "budget_seconds": 5,
            "thread": None,
            "codex": "codex",
        },
    )
    now = [0.0]
    child = SimpleNamespace(pid=123, poll=lambda: None)
    stopped = []
    monkeypatch.setattr(monitor.time, "time", lambda: now[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(monitor, "stop_child", lambda proc: stopped.append(proc))
    assert monitor.supervise(tmp_path) == 0
    assert stopped and stopped[0] is child
    assert monitor.read_json(tmp_path / "status.json")["status"] == "budget_exhausted"


def test_schedule_cli_changes_deadline_without_restarting_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_job(tmp_path)
    monkeypatch.setattr(monitor.time, "time", lambda: 100)
    monkeypatch.setattr(
        monitor.sys,
        "argv",
        [
            "monitor",
            "schedule",
            str(tmp_path),
            "--report-in-hours",
            "8",
            "--quiet-hours",
            "8",
        ],
    )
    assert monitor.main() == 0
    policy = monitor.read_json(tmp_path / "policy.json")
    assert policy["report_at"] == policy["quiet_until"] == 28900
    assert policy["repeat_seconds"] is None
    monkeypatch.setattr(
        monitor.sys,
        "argv",
        [
            "monitor",
            "schedule",
            str(tmp_path),
            "--every-hours",
            "1",
        ],
    )
    assert monitor.main() == 0
    policy = monitor.read_json(tmp_path / "policy.json")
    assert policy["repeat_seconds"] == 3600 and policy["report_at"] == 3700
    assert policy["quiet_until"] == 100


def test_pending_events_are_not_delivered_during_quiet_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = setup_job(tmp_path, report_at=1)
    monitor.tick(tmp_path, state, 1)
    policy = monitor.read_json(tmp_path / "policy.json")
    policy["quiet_until"] = 100
    monitor.write_json(tmp_path / "policy.json", policy)
    monkeypatch.setattr(monitor.time, "time", lambda: 50)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("quiet hours must not invoke the agent")

    monkeypatch.setattr(monitor.subprocess, "run", forbidden)
    monitor.deliver(tmp_path, "owner", "codex")
    assert len(events(tmp_path)) == 1
    assert not list((tmp_path / "events").glob("*.sent"))
