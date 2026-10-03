"""Ownership and smoke promotion must fail closed across process loss."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest

from experiments.jobs import fleet, inbox, monitor


def specification(data: Path):
    identity = data / "input.json"
    identity.write_text('{"immutable":true}')
    return {
        "schema": 1, "hosts": ["gpu1"], "gpu_models": [], "gpu_memory_mib": 100,
        "cpu_threads": 1, "entrypoint": ["work.py"],
        "arguments": ["--frames", "{frames}"],
        "scale": {"frames": {"smoke": 2, "full": 8}},
        "inputs": [{"path": str(identity), "sha256": inbox.file_digest(identity)}],
        "smoke": {"seconds": 2, "representative_basis": "same immutable synthetic input"},
        "full": {"seconds": 2}, "budget_seconds": 30,
        "deadline": datetime.fromtimestamp(time.time() + 300, timezone.utc).isoformat(),
        "validator": ["{python}", "validate.py"], "validator_seconds": 2,
        "required_commands": [],
    }


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    directory = tmp_path / "jobs" / "fleet" / "inbox" / "20261003T000000Z-12345678"
    directory.mkdir(parents=True)
    source = directory / "source"
    source.mkdir()
    (directory / "run").mkdir()
    (source / "work.py").write_text('''import json,os,pathlib,sys
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
p.joinpath('result.json').write_text(json.dumps({'frames':int(sys.argv[-1])}))
''')
    (source / "validate.py").write_text('''import json,os,pathlib
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
value=json.loads(p.joinpath('result.json').read_text())
pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed':value['frames']==2,'checks':['representative frame count']}))
''')
    spec = inbox.validate_spec(specification(tmp_path))
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec), "source_sha256": inbox.source_identity(source)})
    monkeypatch.setenv("PS_JOB_DIR", str(directory / "run"))
    return directory


def test_smoke_validates_then_full_uses_same_entrypoint(campaign):
    assert inbox.campaign(campaign) == 0
    assert json.loads((campaign / "full" / "result.json").read_text())["frames"] == 8
    gate = json.loads((campaign / "gate.json").read_text())
    assert gate["passed"] and gate["validation"]["checks"]
    assert json.loads((campaign / "smoke" / "dispatch.json").read_text())["citable"] is False


@pytest.mark.parametrize("validator", [
    "raise SystemExit(1)",
    "import os,pathlib; pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text('{}')",
    "import os,pathlib; pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text('{\"passed\":true}')",
])
def test_successful_smoke_is_insufficient_without_validator(campaign, validator):
    (campaign / "source" / "validate.py").write_text(validator)
    ready = monitor.read_json(campaign / "ready.json")
    ready["source_sha256"] = inbox.source_identity(campaign / "source")
    monitor.write_json(campaign / "ready.json", ready)
    assert inbox.campaign(campaign) == 1
    assert not (campaign / "full").exists()


@pytest.mark.parametrize("changed", ["input", "code", "spec"])
def test_changed_identity_blocks_full(campaign, changed):
    source = campaign / "source"
    validator = (source / "validate.py").read_text()
    spec = monitor.read_json(campaign / "spec.json")
    if changed == "input":
        validator += f"\npathlib.Path({spec['inputs'][0]['path']!r}).write_text('changed')\n"
    elif changed == "code":
        validator += "\npathlib.Path('work.py').write_text('changed')\n"
    else:
        validator += f"\npathlib.Path({str(campaign / 'spec.json')!r}).write_text('{{}}')\n"
    (source / "validate.py").write_text(validator)
    ready = monitor.read_json(campaign / "ready.json")
    ready["source_sha256"] = inbox.source_identity(source)
    monitor.write_json(campaign / "ready.json", ready)
    assert inbox.campaign(campaign) == 1
    assert not (campaign / "full").exists()


def test_budget_prevents_promotion(campaign, monkeypatch):
    actual = inbox.time.time
    calls = [actual()]
    def clock():
        calls[0] += 1
        return calls[0]
    monkeypatch.setattr(inbox.time, "time", clock)
    spec = monitor.read_json(campaign / "spec.json")
    spec["budget_seconds"] = 5
    monitor.write_json(campaign / "spec.json", spec)
    ready = monitor.read_json(campaign / "ready.json")
    ready["spec_sha256"] = inbox.digest(spec)
    monitor.write_json(campaign / "ready.json", ready)
    assert inbox.campaign(campaign) == 1
    assert not (campaign / "full").exists()


def test_request_ownership_has_exactly_one_winner(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = list(pool.map(lambda _: inbox.acquire_request(tmp_path, "gpu1"), range(8)))
    assert sum(winners) == 1
    assert not inbox.acquire_request(tmp_path, "gpu2")


def test_owner_loss_never_replays(tmp_path, monkeypatch):
    inbox.acquire_request(tmp_path, "gpu1")
    inbox.transition(tmp_path, "launching")
    monkeypatch.setattr(inbox, "is_pid_alive", lambda *args: False)
    inbox.reconcile(tmp_path, "gpu2")
    assert monitor.read_json(tmp_path / "state.json")["status"] == "launching"
    inbox.reconcile(tmp_path, "gpu1")
    assert monitor.read_json(tmp_path / "state.json")["status"] == "attention"
    assert not inbox.acquire_request(tmp_path, "gpu2")


def test_supervisor_acknowledgement_loss_preserves_live_execution(tmp_path, monkeypatch):
    inbox.acquire_request(tmp_path, "gpu1")
    inbox.transition(tmp_path, "launching")
    monitor.write_json(tmp_path / "run" / "supervisor.json", {"pid": 4321, "proc_start_time": 1})
    monkeypatch.setattr(inbox, "is_pid_alive", lambda pid, start: pid == 4321)
    inbox.reconcile(tmp_path, "gpu1")
    assert monitor.read_json(tmp_path / "state.json")["status"] == "launching"


@pytest.mark.parametrize("cancel", [True, False])
def test_pending_cancel_and_deadline_need_no_gpu(tmp_path, monkeypatch, cancel):
    base = tmp_path / "jobs" / "fleet"
    directory = base / "inbox" / "20261003T000000Z-12345678"
    directory.mkdir(parents=True)
    spec = inbox.validate_spec(specification(tmp_path))
    spec["deadline_epoch"] = time.time() - 1
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {})
    inbox.transition(directory, "pending")
    if cancel:
        monitor.write_json(directory / "cancel.json", {})
    monkeypatch.setattr(inbox, "local_probe", lambda _: pytest.fail("expired/cancelled job probed a GPU"))
    inbox.worker_tick(base, "gpu1")
    assert monitor.read_json(directory / "state.json")["status"] == ("cancelled" if cancel else "expired")


def test_only_published_requests_are_visible(tmp_path):
    base = tmp_path / "jobs" / "fleet"
    response = inbox.rpc(base, "prepare", {"job_id": "20261003T000000Z-12345678"})
    assert Path(response["directory"]).is_dir()
    assert inbox.rpc(base, "status", {}) == []
    with pytest.raises(fleet.FleetError, match="unpublished"):
        inbox.rpc(base, "status", {"job_id": "20261003T000000Z-12345678"})


def test_events_repeat_until_explicit_ack(tmp_path, monkeypatch):
    job = "20261003T000000Z-12345678"
    monitor.write_json(tmp_path / (job + ".json"), {"config": {}})
    event_record = {"id": "abc", "kind": "terminal:complete"}
    monkeypatch.setattr(inbox, "remote_rpc", lambda *args: {"events": [event_record.copy()]})
    from argparse import Namespace
    args = Namespace(state_dir=tmp_path, job_id=job, action="events")
    assert inbox.remote_job(args)["events"] == [event_record]
    assert inbox.remote_job(args)["events"] == [event_record]
    monitor.write_json(tmp_path / "event-acks.json", {"abc": time.time()})
    assert inbox.remote_job(args)["events"] == []


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(entrypoint=["-c", "arbitrary()"]),
    lambda s: s.update(cpu_threads=True),
    lambda s: s.update(deadline="tomorrow"),
    lambda s: s.update(arguments=["{unknown}"]),
    lambda s: s["smoke"].update(representative_basis=""),
    lambda s: s["smoke"].update(seconds=601),
])
def test_bad_spec_is_rejected_before_dispatch(tmp_path, mutation):
    spec = specification(tmp_path)
    mutation(spec)
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec, now=time.time())


def test_legacy_unrestricted_launch_is_not_a_public_command(capsys):
    with pytest.raises(SystemExit):
        fleet.main(["launch", "--", "python", "-c", "print(1)"])
    assert "invalid choice" in capsys.readouterr().err


def test_monitor_does_not_launch_pre_cancelled_job(tmp_path, monkeypatch):
    monitor.write_json(tmp_path / "request.json", {"cancel_path": str(tmp_path / "job-cancel.json")})
    monitor.write_json(tmp_path / "job-cancel.json", {})
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: pytest.fail("cancelled job launched"))
    assert monitor.supervise(tmp_path) == 0
    assert monitor.read_json(tmp_path / "status.json")["status"] == "cancelled"


def test_pending_requests_start_in_age_order_with_fresh_probes(tmp_path, monkeypatch):
    base = tmp_path / "jobs" / "fleet"
    names = ["20261003T000001Z-12345678", "20261003T000000Z-12345678"]
    spec = inbox.validate_spec(specification(tmp_path))
    for name in names:
        directory = base / "inbox" / name
        directory.mkdir(parents=True)
        monitor.write_json(directory / "ready.json", {})
        monitor.write_json(directory / "spec.json", spec)
        inbox.transition(directory, "pending")
    order = []
    probes = []
    monkeypatch.setattr(inbox, "local_probe", lambda alias: probes.append(alias) or {})
    monkeypatch.setattr(inbox, "eligible", lambda *a: {})
    monkeypatch.setattr(inbox, "start_request", lambda directory, *a: order.append(directory.name))
    inbox.worker_tick(base, "gpu1")
    assert order == sorted(names)
    assert len(probes) == 2


def test_expired_before_supervisor_start_never_executes(tmp_path, monkeypatch):
    monitor.write_json(tmp_path / "request.json", {"deadline_epoch": time.time() - 1})
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: pytest.fail("expired job launched"))
    assert monitor.supervise(tmp_path) == 1
    assert monitor.read_json(tmp_path / "status.json")["status"] == "expired"


def test_terminal_event_identity_is_shared_between_queue_and_supervisor(tmp_path):
    base = tmp_path / "jobs" / "fleet"
    name = "20261003T000000Z-12345678"
    directory = base / "inbox" / name
    monitor.write_json(directory / "ready.json", {})
    inbox.transition(directory, "complete")
    monitor.write_json(directory / "run" / "events" / "terminal.json", {"id": "other", "kind": "terminal:complete"})
    events = inbox.rpc(base, "events", {"job_id": name})["events"]
    assert len(events) == 1
    assert len(events[0]["id"]) == 32


@pytest.mark.parametrize("field,value", [
    ("memory_used_mb", 257), ("utilization_pct", 6), ("memory_used_mb", float("nan")),
])
def test_claim_time_policy_matches_admission(field, value):
    from experiments.jobs.claims import is_device_free
    device = {"memory_free_mb": 16000, "memory_used_mb": 0, "utilization_pct": 0, "active_pids": []}
    device[field] = value
    assert not is_device_free(device)


def test_verified_pre_child_admission_race_is_archived_and_requeued(tmp_path, monkeypatch):
    inbox.acquire_request(tmp_path, "gpu1")
    inbox.transition(tmp_path, "running")
    monitor.write_json(tmp_path / "run" / "status.json", {"status": "failed", "admission_rejected": True, "error": "GPU was claimed"})
    monitor.write_json(tmp_path / "run" / "supervisor.json", {"pid": 4321, "proc_start_time": 1})
    monkeypatch.setattr(inbox, "is_pid_alive", lambda *args: False)
    inbox.reconcile(tmp_path, "gpu1")
    assert monitor.read_json(tmp_path / "state.json")["status"] == "pending"
    assert not (tmp_path / "owner").exists()
    assert len(list((tmp_path / "attempts").glob("*/owner/identity.json"))) == 1
    assert inbox.acquire_request(tmp_path, "gpu2")


def test_admission_rejection_with_child_identity_never_requeues(tmp_path, monkeypatch):
    inbox.acquire_request(tmp_path, "gpu1")
    inbox.transition(tmp_path, "running")
    monitor.write_json(tmp_path / "run" / "status.json", {"status": "failed", "admission_rejected": True, "pid": 1234})
    monkeypatch.setattr(inbox, "is_pid_alive", lambda *args: False)
    inbox.reconcile(tmp_path, "gpu1")
    assert monitor.read_json(tmp_path / "state.json")["status"] == "failed"
    assert (tmp_path / "owner").exists()


def test_lost_publication_reply_retains_id_and_does_not_resubmit(tmp_path, monkeypatch):
    from argparse import Namespace
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specification(tmp_path)))
    snapshot = tmp_path / "snapshot.tar"
    snapshot.write_bytes(b"snapshot")
    config = {"hosts": ["gpu1"]}
    calls = []
    monkeypatch.setattr(inbox, "load_config", lambda _: config)
    monkeypatch.setattr(fleet, "_build_snapshot", lambda *a, **kw: (snapshot, "hash", {}))
    monkeypatch.setattr(fleet, "_send_snapshot", lambda *a: None)
    def remote(config, action, payload):
        calls.append(action)
        if action == "health":
            return {"workers": {"gpu1": {"updated": time.time()}}}
        if action == "prepare":
            return {"directory": "/data/jobs/fleet/inbox/" + payload["job_id"]}
        raise TimeoutError("reply lost")
    monkeypatch.setattr(inbox, "remote_rpc", remote)
    args = Namespace(state_dir=tmp_path / "state", spec=spec_path, include_change=[], include_untracked=[], chat_id="chat")
    with pytest.raises(fleet.FleetError, match="never resubmit"):
        inbox.submit(args, tmp_path)
    records = list(args.state_dir.glob("*.json"))
    assert len(records) == 1
    assert monitor.read_json(records[0])["status"] == "submission_unknown"
    assert calls == ["health", "prepare", "publish"]


def test_campaign_budget_includes_supervisor_startup(campaign):
    monitor.write_json(campaign / "run" / "status.json", {"started": time.time() - 100})
    monitor.write_json(campaign / "run" / "request.json", {"budget_seconds": 101})
    assert inbox.campaign(campaign) == 1
    assert not (campaign / "smoke").exists()
