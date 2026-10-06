"""A contended GPU must cost at most the unfinished part of a stage, never finished work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import claims, fleet, inbox, monitor
from tests.fleet import test_fleet_inbox
from tests.fleet.test_fleet_inbox import specification

campaign = test_fleet_inbox.campaign
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("declared, expected", [
    (None, {"policy": "stop", "resume_attempts": 0}),
    ({"policy": "pause", "pause_seconds": 20, "resume_attempts": 2}, {"policy": "pause", "pause_seconds": 20.0, "resume_attempts": 2}),
    ({"policy": "continue", "basis": "decoded frames do not depend on wall time"},
     {"policy": "continue", "basis": "decoded frames do not depend on wall time", "resume_attempts": 0}),
])
def test_contention_policy_is_normalized_into_the_spec(tmp_path, declared, expected):
    spec = specification(tmp_path)
    if declared is not None:
        spec["contention"] = declared
    assert inbox.validate_spec(spec)["contention"] == expected


@pytest.mark.parametrize("declared", [
    {"policy": "ignore"},
    {"policy": "pause"},
    {"policy": "pause", "pause_seconds": 0},
    {"policy": "pause", "pause_seconds": 31},  # exceeds the 30 s budget
    {"policy": "stop", "pause_seconds": 10},
    {"policy": "continue"},
    {"policy": "continue", "basis": "timing-free", "resume_attempts": 1},
    {"policy": "stop", "resume_attempts": -1},
    {"policy": "stop", "resume_attempts": True},
    {"policy": "stop", "replay": True},
    "stop",
])
def test_bad_contention_declarations_are_rejected(tmp_path, declared):
    spec = specification(tmp_path)
    spec["contention"] = declared
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec)


@pytest.fixture
def signals(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(monitor.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    stopped: list[Any] = []
    monkeypatch.setattr(monitor, "stop_child", lambda child: stopped.append(child) or True)
    return SimpleNamespace(sent=sent, stopped=stopped, child=SimpleNamespace(pid=4242))


def running() -> dict[str, Any]:
    return {"status": "running", "started": 0, "last_progress": 0, "emitted": []}


def test_stop_policy_stops_and_records_the_episode(signals):
    state = running()
    assert monitor.contention_step(signals.child, state, {"policy": "stop"}, [77], 10) is True
    assert state["status"] == "contended" and state["timing_contaminated"]
    assert state["contention"] == [{"detected": 10, "action": "stop", "foreign_gpu_pids": [77], "ended": 10}]
    assert signals.stopped == [signals.child] and signals.sent == []


def test_continue_policy_keeps_running_and_marks_timing(signals, tmp_path):
    state = running()
    policy = {"policy": "continue", "basis": "timing-free"}
    for now, foreign in ((10, [77]), (20, [77, 78]), (30, []), (40, [])):
        assert monitor.contention_step(signals.child, state, policy, foreign, now) is None
    assert state["status"] == "running" and state["timing_contaminated"]
    assert state["contention"] == [{"detected": 10, "action": "continue", "foreign_gpu_pids": [77, 78], "cleared": 30}]
    assert not signals.stopped and not signals.sent
    monitor.write_json(tmp_path / "policy.json", {"stall_seconds": 10**6, "report_at": None, "quiet_until": 0})
    monitor.tick(tmp_path, state, 41)
    monitor.tick(tmp_path, state, 42)
    kinds = [monitor.read_json(p)["kind"] for p in (tmp_path / "events").glob("*.json")]
    assert kinds == ["contention:10"]


def test_pause_policy_continues_when_the_gpu_clears(signals):
    state = running()
    policy = {"policy": "pause", "pause_seconds": 100}
    assert monitor.contention_step(signals.child, state, policy, [77], 10) is None
    assert monitor.contention_step(signals.child, state, policy, [77], 60) is None
    assert monitor.contention_step(signals.child, state, policy, [], 70) is None
    assert signals.sent == [(4242, signal.SIGSTOP), (4242, signal.SIGCONT)]
    assert state["status"] == "running" and not signals.stopped
    assert monitor.paused_seconds(state, 0, 1000) == 60
    assert monitor.paused_seconds(state, 30, 1000) == 40
    # A new occupant opens a new episode and pauses again.
    monitor.contention_step(signals.child, state, policy, [90], 80)
    assert len(state["contention"]) == 2 and signals.sent[-1] == (4242, signal.SIGSTOP)


def test_pause_policy_stops_when_the_gpu_stays_busy(signals):
    state = running()
    policy = {"policy": "pause", "pause_seconds": 100}
    monitor.contention_step(signals.child, state, policy, [77], 10)
    assert monitor.contention_step(signals.child, state, policy, [77], 109) is None
    assert monitor.contention_step(signals.child, state, policy, [77], 110) is True
    assert state["status"] == "contended" and "100" in state["error"]
    assert state["contention"][0]["ended"] == 110
    assert monitor.paused_seconds(state, 0, 1000) == 100


def process_state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def test_paused_process_group_is_really_suspended_and_still_stoppable(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.05)"], start_new_session=True)
    try:
        state = running()
        policy = {"policy": "pause", "pause_seconds": 100}
        monitor.contention_step(child, state, policy, [77], time.time())
        time.sleep(0.3)
        assert process_state(child.pid).startswith("T")
        monitor.contention_step(child, state, policy, [], time.time())
        time.sleep(0.3)
        assert not process_state(child.pid).startswith("T")
        # A group still paused when its pause expires must terminate on SIGTERM, not hang.
        monitor.contention_step(child, state, policy, [77], time.time())
        started = time.monotonic()
        assert monitor.contention_step(child, state, policy, [77], time.time() + 100) is True
        assert time.monotonic() - started < 10
        assert child.poll() == -signal.SIGTERM
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def gpu_request(tmp_path: Path, **extra: Any) -> None:
    monitor.write_json(tmp_path / "policy.json", {"stall_seconds": 10**6, "report_at": None, "quiet_until": 0})
    monitor.write_json(tmp_path / "request.json", {
        "command": ["synthetic-worker"], "cwd": str(tmp_path), "budget_seconds": 600,
        "thread": None, "codex": "codex",
        "claims": {"gpu": "GPU-contended", "min_free_memory_mb": 1000, "cpu_threads": 1,
                   "claims_dir": str(tmp_path / "claims"), "available_cores": 4},
        **extra,
    })


@pytest.fixture
def fake_gpu(monkeypatch):
    device = {"index": 0, "uuid": "GPU-contended", "name": "Mock", "memory_free_mb": 40000, "memory_total_mb": 48000,
              "memory_used_mb": 0, "utilization_pct": 0, "active_pids": []}
    monkeypatch.setattr(claims, "query_gpus", lambda probe_fn=None: [dict(device)])
    now = [1000.0]
    monkeypatch.setattr(monitor.time, "time", lambda: now[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    child = SimpleNamespace(pid=4242, poll=lambda: None)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(monitor, "stop_child", lambda proc: True)
    return now


def test_supervisor_releases_the_gpu_then_salvages_a_contended_stage(tmp_path, monkeypatch, fake_gpu):
    gpu_request(tmp_path, salvage=["salvage-command"], contention={"policy": "stop"})
    monkeypatch.setattr(monitor, "foreign_gpu_processes", lambda device, group: [77])
    calls = []

    def salvage(directory, request, reason):
        calls.append(reason)
        status = claims.get_claims_status(tmp_path / "claims")
        assert not status["devices"], "the GPU claim must be released before salvage"
        return {"exit_code": 0, "reason": reason}

    monkeypatch.setattr(monitor, "salvage", salvage)
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "contended" and status["salvage"] == {"exit_code": 0, "reason": "contended"}
    assert status["ended"] == status["contention"][0]["ended"]
    assert calls == ["contended"]


def test_completed_job_is_not_salvaged(tmp_path, monkeypatch, fake_gpu):
    gpu_request(tmp_path, salvage=["salvage-command"])
    monkeypatch.setattr(monitor, "foreign_gpu_processes", lambda device, group: [])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4242, poll=lambda: 0))
    monkeypatch.setattr(monitor, "salvage", lambda *a: pytest.fail("a completed job publishes normally"))
    assert monitor.supervise(tmp_path) == 0
    assert monitor.read_json(tmp_path / "status.json")["status"] == "complete"


def test_continue_policy_runs_to_completion_with_contaminated_timing(tmp_path, monkeypatch, fake_gpu):
    gpu_request(tmp_path, contention={"policy": "continue", "basis": "timing-free"})
    polls = iter([None, None, 0])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4242, poll=lambda: next(polls)))
    monkeypatch.setattr(monitor, "foreign_gpu_processes", lambda device, group: [77])
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "complete" and status["timing_contaminated"]
    assert status["contention"][0]["action"] == "continue"


def test_salvage_command_receives_the_reason(tmp_path):
    marker = tmp_path / "reason.txt"
    request = {"cwd": str(tmp_path), "salvage": [sys.executable, "-c", f"import sys,pathlib; pathlib.Path({str(marker)!r}).write_text(sys.argv[1])"]}
    record = monitor.salvage(tmp_path, request, "budget_exhausted")
    assert record["exit_code"] == 0 and marker.read_text() == "budget_exhausted"


def members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as bundle:
        files = {m.name: bundle.extractfile(m) for m in bundle.getmembers() if m.isfile()}
        return {name: handle.read() for name, handle in files.items() if handle is not None}


def test_salvage_publishes_partial_outputs_and_finished_checkpoint_files(tmp_path):
    directory = tmp_path / "job"
    for stage in ("smoke", "full"):
        scratch = tmp_path / "scratch" / stage
        (scratch / "publish").mkdir(parents=True)
        (scratch / "checkpoint").mkdir()
        (scratch / "publish" / "clip-0.json").write_text(stage)
        (scratch / "checkpoint" / "state.json").write_text('{"done": 4}')
        (scratch / "checkpoint" / ".state.json.9.tmp").write_text("half-writ")
        record = {"scratch": str(scratch), **({"released": 1.0} if stage == "smoke" else {})}
        monitor.write_json(directory / stage / "staging.json", record)
    assert inbox.salvage(directory, "contended") == 0
    full = monitor.read_json(directory / "full" / "staging.json")
    assert full["partial"]["partial"] is True and full["salvaged"]["reason"] == "contended"
    assert full["partial"]["sha256"] == inbox.file_digest(directory / "full" / "partial.tar")
    assert members(directory / "full" / "partial.tar") == {"publish/clip-0.json": b"full"}
    assert members(directory / "full" / "checkpoint.tar") == {"checkpoint/state.json": b'{"done": 4}'}
    assert full["checkpoint"]["sha256"] == inbox.file_digest(directory / "full" / "checkpoint.tar")
    assert not (directory / "full" / "published.tar").exists()
    assert "salvaged" not in monitor.read_json(directory / "smoke" / "staging.json")
    # Salvage runs once; scratch stays on the host for inspection.
    assert inbox.salvage(directory, "again") == 0
    assert monitor.read_json(directory / "full" / "staging.json")["salvaged"]["reason"] == "contended"
    assert (tmp_path / "scratch" / "full" / "publish" / "clip-0.json").exists()


def test_salvage_reports_scratch_left_on_another_host(tmp_path):
    monitor.write_json(tmp_path / "full" / "staging.json", {"scratch": str(tmp_path / "elsewhere")})
    assert inbox.salvage(tmp_path, "contended") == 1
    assert "not present" in monitor.read_json(tmp_path / "full" / "staging.json")["salvaged"]["error"]


def test_save_checkpoint_is_atomic_and_rejects_hidden_names(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CHECKPOINT_DIR", str(tmp_path))
    assert monitor.save_checkpoint("state.json", b"{}").read_bytes() == b"{}"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    for name in (".state", "a/b", ""):
        with pytest.raises(ValueError):
            monitor.save_checkpoint(name, b"")


def test_pause_time_extends_only_the_command_allowance(tmp_path):
    run = tmp_path / "run"
    command = [sys.executable, "-c", "import time; time.sleep(1.5)"]
    with pytest.raises(fleet.FleetError):
        with (tmp_path / "a.log").open("w") as log:
            inbox.run_bounded(command, cwd=tmp_path, env=dict(os.environ), log=log, seconds=0.5, budget_end=time.time() + 60, run=run)
    now = time.time()
    monitor.write_json(run / "status.json", {"contention": [{"detected": now, "paused_at": now, "resumed_at": now + 60}]})
    with (tmp_path / "b.log").open("w") as log:
        assert inbox.run_bounded(command, cwd=tmp_path, env=dict(os.environ), log=log, seconds=0.5, budget_end=time.time() + 60, run=run) == 0
    with pytest.raises(fleet.FleetError):
        with (tmp_path / "c.log").open("w") as log:
            inbox.run_bounded(command, cwd=tmp_path, env=dict(os.environ), log=log, seconds=0.5, budget_end=time.time() + 0.5, run=run)


WORKLOAD = '''import json,os,pathlib,sys,time
from experiments.jobs import monitor
stage, frames = os.environ["PS_STAGE"], int(sys.argv[-1])
checkpoint = pathlib.Path(os.environ["PS_CHECKPOINT_DIR"]) / "done.json"
start = json.loads(checkpoint.read_text())["done"] if checkpoint.exists() else 0
publish = pathlib.Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
publish.mkdir(exist_ok=True)
for frame in range(start, frames):
    (publish / f"{frame}.txt").write_text(os.environ["PS_ATTEMPT"])
    monitor.save_checkpoint("done.json", json.dumps({"done": frame + 1}).encode())
    if stage == "full" and frame == 3 and os.environ["PS_ATTEMPT"] == "1":
        pathlib.Path(os.environ["PS_STAGE_DIR"], "waiting").write_text("")
        time.sleep(120)
pathlib.Path(os.environ["PS_STAGE_DIR"], "result.json").write_text(json.dumps({"frames": frames, "resumed_at": start}))
'''


def contended_job(directory: Path, resume_attempts: int, monkeypatch) -> dict[str, Any]:
    """Run attempt 1 for real until full is mid-stage, then stop it as the supervisor would."""
    (directory / "source" / "work.py").write_text(WORKLOAD)
    spec = specification(directory.parents[3])
    spec.update(full={"seconds": 20}, budget_seconds=100, contention={"policy": "stop", "resume_attempts": resume_attempts})
    spec = inbox.validate_spec(spec)
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec), "source_sha256": inbox.source_identity(directory / "source")})
    inbox.acquire_request(directory, "gpu1")
    inbox.transition(directory, "running")
    started = time.time()
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY)}
    child = subprocess.Popen([sys.executable, "-m", "experiments.jobs.inbox", "campaign", str(directory)], cwd=directory / "source", env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while not (directory / "full" / "waiting").exists():
            assert child.poll() is None and time.monotonic() < deadline, "attempt 1 never reached the full stage"
            time.sleep(0.05)
    finally:
        assert claims.terminate_and_reap_process_group(child)
    monitor.write_json(directory / "run" / "status.json", {"status": "contended", "started": started, "ended": time.time(), "contention": [{"detected": time.time(), "action": "stop", "foreign_gpu_pids": [77]}]})
    monitor.write_json(directory / "run" / "supervisor.json", {"pid": 999999, "proc_start_time": 1})
    assert inbox.salvage(directory, "contended") == 0
    monkeypatch.setattr(inbox, "is_pid_alive", lambda *args: False)
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY))
    return spec


def test_declared_resume_continues_a_contended_stage_from_its_checkpoint(campaign, monkeypatch):
    spec = contended_job(campaign, 1, monkeypatch)
    assert inbox.rpc(campaign.parents[1], "status", {"job_id": campaign.name})["status"] == "resuming"
    inbox.reconcile(campaign, "gpu1")
    state = monitor.read_json(campaign / "state.json")
    assert state["status"] == "pending" and state["attempt"] == 2 and 0 < state["consumed_seconds"] < spec["budget_seconds"]
    archive = campaign / "resumes" / "1"
    assert {p.name for p in archive.iterdir()} >= {"run", "full", "owner"}
    assert not (campaign / "owner").exists() and not (campaign / "full").exists() and (campaign / "smoke").is_dir()
    assert members(archive / "full" / "partial.tar") == {f"publish/{i}.txt": b"1" for i in range(4)}
    resume = monitor.read_json(campaign / "resume.json")
    assert resume["completed_stages"] == ["smoke"] and resume["previous"]["stage"] == "full"
    assert resume["previous"]["checkpoint"]["path"] == str(archive / "full" / "checkpoint.tar")
    events = [monitor.read_json(p)["kind"] for p in (campaign / "events").glob("*.json")]
    assert "resume:attempt-2" in events and not any(k.startswith("terminal:") for k in events)
    # The next attempt is admitted like any pending request, then starts from the checkpoint.
    assert inbox.acquire_request(campaign, "gpu2")
    (campaign / "run").mkdir()
    monitor.write_json(campaign / "run" / "request.json", {"attempt": 2, "budget_seconds": spec["budget_seconds"] - state["consumed_seconds"]})
    assert inbox.campaign(campaign) == 0, monitor.read_json(campaign / "campaign-error.json")
    assert json.loads((campaign / "full" / "result.json").read_text()) == {"frames": 8, "resumed_at": 4}
    dispatch = monitor.read_json(campaign / "full" / "dispatch.json")
    assert dispatch["attempt"] == 2 and dispatch["resumed_from"]["checkpoint"]["sha256"] == resume["previous"]["checkpoint"]["sha256"]
    assert members(Path(monitor.read_json(campaign / "full" / "staging.json")["published"]["path"])) == {
        **{f"publish/{i}.txt": b"2" for i in range(4, 8)},
    }
    # Finished attempts are never replayed: a second campaign for the same attempt refuses.
    assert inbox.campaign(campaign) == 1


def test_contended_job_without_declared_resume_stays_terminal(campaign, monkeypatch):
    contended_job(campaign, 0, monkeypatch)
    assert inbox.rpc(campaign.parents[1], "status", {"job_id": campaign.name})["status"] == "contended"
    inbox.reconcile(campaign, "gpu1")
    state = monitor.read_json(campaign / "state.json")
    assert state["status"] == "contended" and "no declared resume" in state["resume_declined"]
    assert (campaign / "owner").exists() and not (campaign / "resumes").exists()
    # The finished smoke and the partial full stage are both preserved.
    assert (campaign / "full" / "partial.tar").exists() and (campaign / "smoke" / "published.tar").exists()


def test_resume_needs_a_checkpoint_and_remaining_budget(campaign, monkeypatch):
    spec = contended_job(campaign, 2, monkeypatch)
    state = monitor.read_json(campaign / "state.json")
    status = monitor.read_json(campaign / "run" / "status.json")
    decision, _ = inbox.resume_decision(campaign, spec, state, status, time.time())
    assert decision is not None and decision["stage"] == "full"
    decision, reason = inbox.resume_decision(campaign, spec, {**state, "consumed_seconds": 90}, status, time.time())
    assert decision is None and "budget" in reason
    decision, reason = inbox.resume_decision(campaign, spec, {**state, "attempt": 3}, status, time.time())
    assert decision is None and "no declared resume" in reason
    staged = monitor.read_json(campaign / "full" / "staging.json")
    monitor.write_json(campaign / "full" / "staging.json", {**staged, "checkpoint": None})
    decision, reason = inbox.resume_decision(campaign, spec, state, status, time.time())
    assert decision is None and "no checkpoint" in reason


def test_resumed_attempt_refuses_a_changed_checkpoint(campaign, monkeypatch):
    spec = contended_job(campaign, 1, monkeypatch)
    inbox.reconcile(campaign, "gpu1")
    checkpoint = Path(monitor.read_json(campaign / "resume.json")["previous"]["checkpoint"]["path"])
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"tampered")
    (campaign / "run").mkdir()
    monitor.write_json(campaign / "run" / "request.json", {"attempt": 2, "budget_seconds": spec["budget_seconds"]})
    assert inbox.campaign(campaign) == 1
    assert "checkpoint identity changed" in monitor.read_json(campaign / "campaign-error.json")["error"]
