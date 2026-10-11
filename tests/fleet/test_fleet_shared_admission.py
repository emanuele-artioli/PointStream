"""The opt-in shared GPU class: admitted beside other users when memory suffices,
never two of our own claims beyond what fits, and stopped on its own overrun."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import claims, fleet, inbox, monitor
from tests.fleet import test_fleet_inbox
from tests.fleet.test_fleet_contention import fake_gpu, gpu_request  # noqa: F401 -- fixture
from tests.fleet.test_fleet_inbox import specification

campaign = test_fleet_inbox.campaign
BASIS = "decoded frames and scores do not depend on wall time"


def shared_specification(data: Path, **sharing: Any) -> dict[str, Any]:
    return {**specification(data), "gpu_memory_mib": 12000, "gpu_sharing": {"mode": "shared", "basis": BASIS, **sharing}}


# --- specification -----------------------------------------------------------


def test_spec_without_the_field_is_unchanged(tmp_path):
    spec = inbox.validate_spec(specification(tmp_path))
    assert "gpu_sharing" not in spec
    assert spec["contention"] == {"policy": "stop", "resume_attempts": 0}
    assert inbox.gpu_sharing(spec) == {"mode": "exclusive"}


def test_shared_spec_is_normalized_with_continue_contention(tmp_path):
    spec = inbox.validate_spec(shared_specification(tmp_path))
    assert spec["gpu_sharing"] == {"mode": "shared", "basis": BASIS, "margin_mib": fleet.DEFAULT_SHARING_MARGIN_MIB}
    assert spec["contention"] == {"policy": "continue", "basis": BASIS, "resume_attempts": 0}
    # The host re-validates the normalized spec at publication.
    assert inbox.validate_spec(spec) == spec


def test_shared_spec_keeps_a_declared_continue_and_margin(tmp_path):
    spec = shared_specification(tmp_path, margin_mib=4096)
    spec["contention"] = {"policy": "continue", "basis": "own basis"}
    normalized = inbox.validate_spec(spec)
    assert normalized["gpu_sharing"]["margin_mib"] == 4096
    assert normalized["contention"]["basis"] == "own basis"


def test_explicit_exclusive_is_accepted(tmp_path):
    spec = {**specification(tmp_path), "gpu_sharing": {"mode": "exclusive"}}
    assert inbox.validate_spec(spec)["gpu_sharing"] == {"mode": "exclusive"}


@pytest.mark.parametrize("sharing", [
    {"mode": "shared"},
    {"mode": "shared", "basis": "  "},
    {"mode": "shared", "basis": 3},
    {"mode": "pooled", "basis": BASIS},
    {"mode": "exclusive", "basis": BASIS},
    {"mode": "shared", "basis": BASIS, "margin_mib": 512},
    {"mode": "shared", "basis": BASIS, "margin_mib": True},
    {"mode": "shared", "basis": BASIS, "margin_mib": 8192.0},
    {"mode": "shared", "basis": BASIS, "limit": 1},
    "shared",
])
def test_bad_sharing_declarations_are_rejected(tmp_path, sharing):
    spec = {**specification(tmp_path), "gpu_sharing": sharing}
    with pytest.raises(fleet.FleetError, match="gpu_sharing"):
        inbox.validate_spec(spec)


@pytest.mark.parametrize("contention", [
    {"policy": "stop"},
    {"policy": "pause", "pause_seconds": 10},
    {"policy": "stop", "resume_attempts": 1},
])
def test_shared_spec_cannot_stop_or_pause_for_contention(tmp_path, contention):
    spec = shared_specification(tmp_path)
    spec["contention"] = contention
    with pytest.raises(fleet.FleetError, match="contention"):
        inbox.validate_spec(spec)


def test_cpu_job_cannot_share_a_gpu(tmp_path):
    spec = {**specification(tmp_path), "device": "cpu", "gpu_memory_mib": 0, "gpu_sharing": {"mode": "shared", "basis": BASIS}}
    with pytest.raises(fleet.FleetError, match="cpu job"):
        inbox.validate_spec(spec)


# --- selection on the probe --------------------------------------------------


def device(uuid: str, *, free: int = 40000, used: int = 1, util: int = 0, pids: int = 0, shared: list[dict[str, Any]] | None = None,
           name: str = "NVIDIA RTX A6000", claimed: bool = False) -> dict[str, Any]:
    return {"uuid": uuid, "name": name, "memory_used_mib": used, "memory_free_mib": free, "memory_total_mib": 49140,
            "utilization_pct": util, "compute_processes": [{"pid": 100 + n, "memory_mib": 1000} for n in range(pids)],
            "pointstream_claimed": claimed, "pointstream_shared_claims": shared or []}


def host(*gpus: dict[str, Any]) -> dict[str, Any]:
    return {"alias": "gpu1", "host": "gpu1.example", "available": True, "gpus": list(gpus), "cpu_headroom": 32,
            "python_available": True, "data_root_available": True, "tools": {}}


def shared_select(*gpus: dict[str, Any], memory: int = 12000, margin: int = 8192) -> dict[str, Any]:
    _, gpu = fleet.select_gpu([host(*gpus)], required_memory_mib=memory, cpu_threads=1, sharing="shared", sharing_margin_mib=margin)
    return gpu


def test_exclusive_still_refuses_a_gpu_with_a_foreign_process_or_a_shared_claim():
    busy = device("GPU-busy", used=9000, util=60, pids=1)
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([host(busy)], required_memory_mib=12000, cpu_threads=1)
    with_share = device("GPU-shared", shared=[{"token": "a", "memory_mib": 1000, "margin_mib": 1024}])
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([host(with_share)], required_memory_mib=1000, cpu_threads=1)


def test_shared_is_admitted_beside_a_foreign_process_when_memory_suffices():
    gpu = shared_select(device("GPU-busy", free=12000 + 8192, used=28000, util=97, pids=3))
    assert gpu["uuid"] == "GPU-busy"


def test_shared_is_refused_when_memory_does_not_cover_peak_plus_margin():
    with pytest.raises(fleet.FleetError, match="sharing margin"):
        shared_select(device("GPU-busy", free=12000 + 8191, pids=1))


def test_shared_prefers_an_idle_gpu_over_a_busy_one_with_more_memory():
    busy = device("GPU-busy", free=45000, pids=1, name="NVIDIA RTX 6000 Ada Generation")
    idle = device("GPU-idle", free=30000)
    assert shared_select(busy, idle)["uuid"] == "GPU-idle"


def test_shared_never_takes_an_exclusively_claimed_gpu():
    with pytest.raises(fleet.FleetError):
        shared_select(device("GPU-ours", claimed=True))


def test_two_shared_claims_fit_only_with_both_peaks_and_margins():
    first = [{"token": "a", "memory_mib": 10000, "margin_mib": 8192}]
    assert shared_select(device("GPU-x", free=18192 + 20192, shared=first))["uuid"] == "GPU-x"
    with pytest.raises(fleet.FleetError):
        shared_select(device("GPU-x", free=18192 + 20191, shared=first))


@pytest.mark.parametrize("claims_list", [[{"token": "a"}], ["junk"], {"a": 1}, [{"token": "a", "memory_mib": -1, "margin_mib": 1}]])
def test_malformed_shared_claims_fail_closed(claims_list):
    with pytest.raises(fleet.FleetError):
        shared_select(device("GPU-x", free=48000, shared=claims_list))


def test_shared_still_needs_a_complete_probe():
    with pytest.raises(fleet.FleetError, match="probe failed"):
        fleet.select_gpu([{"alias": "gpu1", "available": False, "error": "probe failed"}],
                         required_memory_mib=1, cpu_threads=1, sharing="shared")


def test_eligible_passes_the_declared_margin(tmp_path):
    spec = inbox.validate_spec(shared_specification(tmp_path, margin_mib=2048))
    assert inbox.eligible(spec, host(device("GPU-busy", free=12000 + 2048, pids=2)))["uuid"] == "GPU-busy"
    with pytest.raises(fleet.FleetError):
        inbox.eligible(inbox.validate_spec(specification(tmp_path)), host(device("GPU-busy", free=40000, pids=2)))


# --- claims on the host ------------------------------------------------------


def gpu_reading(uuid: str = "GPU-share", free: float = 40000.0) -> dict[str, Any]:
    # Busy with another user's work: exclusive admission would refuse it.
    return {"index": 0, "uuid": uuid, "name": "Mock", "memory_free_mb": free, "memory_total_mb": 48000.0,
            "memory_used_mb": 8000.0, "utilization_pct": 95.0, "active_pids": [31337]}


def shared_claim(tmp_path: Path, memory: int = 10000, margin: int = 8192, free: float = 40000.0, job: str = "job") -> claims.DeviceClaim:
    return claims.acquire_shared_device_claim(
        device_uuid="GPU-share", memory_mib=memory, margin_mib=margin, claims_dir=tmp_path,
        job_id=job, probe_fn=lambda: [gpu_reading(free=free)],
    )


def test_shared_claim_ignores_foreign_processes_and_utilization(tmp_path):
    assert not claims.is_device_free(gpu_reading())
    claim = shared_claim(tmp_path)
    assert claim.sharing == "shared" and claim.memory_mib == 10000
    assert claims.device_claim_file(tmp_path, claim).is_file()


def test_shared_claim_is_refused_without_memory(tmp_path):
    with pytest.raises(claims.DeviceBusyError):
        shared_claim(tmp_path, free=18191.0)


def test_shared_claim_fails_closed_on_an_incomplete_probe(tmp_path):
    with pytest.raises(claims.DeviceUnavailableError):
        claims.acquire_shared_device_claim(device_uuid="GPU-share", memory_mib=1, margin_mib=1024, claims_dir=tmp_path, probe_fn=lambda: [])
    bad = {**gpu_reading(), "memory_free_mb": float("nan")}
    with pytest.raises(claims.DeviceBusyError):
        claims.acquire_shared_device_claim(device_uuid="GPU-share", memory_mib=1, margin_mib=1024, claims_dir=tmp_path, probe_fn=lambda: [bad])


def test_two_shared_claims_on_one_gpu_only_when_both_fit(tmp_path):
    first = shared_claim(tmp_path, job="a")
    second = shared_claim(tmp_path, free=2 * 18192.0, job="b")
    assert first.device_uuid == second.device_uuid and first.token != second.token
    with pytest.raises(claims.DeviceBusyError, match="2 other shared claims"):
        shared_claim(tmp_path, free=3 * 18192.0 - 1, job="c")
    claims.release_device_claim(first.host, first.device_uuid, first.token, tmp_path)
    assert shared_claim(tmp_path, free=2 * 18192.0, job="c")


def test_verified_dead_shared_claim_is_reclaimed(tmp_path):
    stale = shared_claim(tmp_path)
    record = claims.device_claim_file(tmp_path, stale)
    record.write_text(json.dumps({**asdict(stale), "pid": 2**22 + 12345, "proc_start_time": None}))
    assert shared_claim(tmp_path, free=18192.0, job="next")
    assert not record.exists()


def test_shared_and_exclusive_claims_exclude_each_other(tmp_path):
    shared = shared_claim(tmp_path)
    idle = {**gpu_reading(), "memory_used_mb": 0.0, "utilization_pct": 0.0, "active_pids": []}
    with pytest.raises(claims.DeviceBusyError):
        claims.acquire_device_claim(device_uuid="GPU-share", claims_dir=tmp_path, probe_fn=lambda: [idle])
    claims.release_device_claim(shared.host, shared.device_uuid, shared.token, tmp_path)
    exclusive = claims.acquire_device_claim(device_uuid="GPU-share", claims_dir=tmp_path, probe_fn=lambda: [idle])
    with pytest.raises(claims.DeviceClaimConflictError):
        shared_claim(tmp_path)
    assert not list(claims.shared_claims_dir(tmp_path, exclusive.host, "GPU-share").glob("*.json"))


def test_pre_start_recheck_uses_the_shared_rule(tmp_path):
    claim = shared_claim(tmp_path)
    assert claims.is_shared_claim_admissible(claim, gpu_reading(free=18192.0), tmp_path)
    assert not claims.is_shared_claim_admissible(claim, gpu_reading(free=18191.0), tmp_path)
    assert not claims.is_shared_claim_admissible(claim, None, tmp_path)
    shared_claim(tmp_path, free=2 * 18192.0, job="other")
    assert not claims.is_shared_claim_admissible(claim, gpu_reading(free=2 * 18192.0 - 1), tmp_path)


def test_child_env_exports_the_memory_limit_only_when_shared(tmp_path):
    claim = shared_claim(tmp_path)
    session = claims.ClaimSession(token=claim.token, device_claim=claim, cpu_claim=None, claims_dir=tmp_path)
    env = claims.build_child_env(session, {})
    assert env["PS_GPU_MEMORY_LIMIT_MIB"] == "10000" and env["PS_GPU_SHARING"] == "shared"
    exclusive = claims.DeviceClaim(token="ab", host="h", device_uuid="GPU-x", device_ordinal=0, child_ordinals={}, pid=1, created_at=0, job_id="j")
    plain = claims.build_child_env(claims.ClaimSession(token="ab", device_claim=exclusive, cpu_claim=None, claims_dir=tmp_path), {})
    assert "PS_GPU_MEMORY_LIMIT_MIB" not in plain and "PS_GPU_SHARING" not in plain


# --- the supervisor ----------------------------------------------------------


SHARING = {"mode": "shared", "basis": BASIS, "margin_mib": 8192}


def shared_request(tmp_path: Path, **extra: Any) -> None:
    gpu_request(tmp_path, contention={"policy": "continue", "basis": BASIS}, gpu_sharing=SHARING, **extra)
    request = monitor.read_json(tmp_path / "request.json")
    request["claims"].update(sharing="shared", memory_mib=10000, margin_mib=8192, min_free_memory_mb=18192)
    monitor.write_json(tmp_path / "request.json", request)


@pytest.fixture
def busy_gpu(monkeypatch, fake_gpu):  # noqa: F811 -- fixture parameter
    reading = {"index": 0, "uuid": "GPU-contended", "name": "Mock", "memory_free_mb": 20000, "memory_total_mb": 48000,
               "memory_used_mb": 28000, "utilization_pct": 99, "active_pids": [77]}
    monkeypatch.setattr(claims, "query_gpus", lambda probe_fn=None: [dict(reading)])
    monkeypatch.setattr(monitor, "foreign_gpu_processes", lambda device, group: [77])
    return reading


def test_shared_job_runs_beside_a_foreign_process_with_contaminated_timing(tmp_path, monkeypatch, busy_gpu):
    shared_request(tmp_path)
    polls = iter([None, None, 0])
    environments: list[dict[str, str]] = []

    def spawn(*args: Any, **kwargs: Any) -> Any:
        environments.append(kwargs["env"])
        return SimpleNamespace(pid=4242, poll=lambda: next(polls))

    monkeypatch.setattr(monitor.subprocess, "Popen", spawn)
    monkeypatch.setattr(monitor, "owned_gpu_memory_mib", lambda device, group: 9000)
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "complete" and status["timing_contaminated"] is True
    assert status["gpu_sharing"] == SHARING
    assert status["contention"][0]["action"] == "continue"
    assert environments[0]["PS_GPU_MEMORY_LIMIT_MIB"] == "10000"
    assert not claims.get_claims_status(tmp_path / "claims")["devices"]


def test_shared_job_is_refused_at_pre_start_recheck_when_memory_drops(tmp_path, monkeypatch, busy_gpu):
    shared_request(tmp_path)
    readings = iter([20000, 18191])
    monkeypatch.setattr(claims, "query_gpus", lambda probe_fn=None: [{**busy_gpu, "memory_free_mb": next(readings)}])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: pytest.fail("no child may start"))
    monitor.supervise(tmp_path)
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "failed" and status["admission_rejected"] and "busy before launch" in status["error"]
    assert not claims.get_claims_status(tmp_path / "claims")["devices"]


def test_memory_overrun_stops_the_job_releases_the_gpu_then_salvages(tmp_path, monkeypatch, busy_gpu):
    shared_request(tmp_path, salvage=["salvage-command"])
    monkeypatch.setattr(monitor, "owned_gpu_memory_mib", lambda device, group: 10000 + 1025)
    stopped: list[Any] = []
    monkeypatch.setattr(monitor, "stop_child", lambda child: stopped.append(child) or True)
    signalled: list[Any] = []
    monkeypatch.setattr(monitor.os, "killpg", lambda *a: signalled.append(a))

    def salvage(directory, request, reason):
        assert not claims.get_claims_status(tmp_path / "claims")["devices"], "release before salvage"
        return {"exit_code": 0, "reason": reason}

    monkeypatch.setattr(monitor, "salvage", salvage)
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "failed" and "GPU memory overrun" in status["error"]
    assert status["gpu_memory_overrun"]["used_mib"] == 11025 and status["gpu_memory_overrun"]["limit_mib"] == 11024
    assert status["salvage"] == {"exit_code": 0, "reason": "failed"}
    assert stopped and signalled == [], "only the owned group is stopped; foreign PID 77 is never signalled"


def test_usage_at_the_tolerance_keeps_running(tmp_path, monkeypatch, busy_gpu):
    shared_request(tmp_path)
    polls = iter([None, 0])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4242, poll=lambda: next(polls)))
    monkeypatch.setattr(monitor, "owned_gpu_memory_mib", lambda device, group: 11024)
    assert monitor.supervise(tmp_path) == 0
    assert monitor.read_json(tmp_path / "status.json")["status"] == "complete"


def test_unverifiable_own_memory_stops_a_shared_job(tmp_path, monkeypatch, busy_gpu):
    shared_request(tmp_path)
    monkeypatch.setattr(monitor, "owned_gpu_memory_mib", lambda device, group: None)
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "failed" and "could not be verified" in status["error"]


def test_exclusive_job_never_queries_its_own_memory(tmp_path, monkeypatch, fake_gpu):  # noqa: F811
    gpu_request(tmp_path)
    monkeypatch.setattr(monitor, "foreign_gpu_processes", lambda device, group: [])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4242, poll=lambda: 0))
    monkeypatch.setattr(monitor, "owned_gpu_memory_mib", lambda *a: pytest.fail("exclusive jobs are not memory-policed"))
    assert monitor.supervise(tmp_path) == 0
    status = monitor.read_json(tmp_path / "status.json")
    assert status["status"] == "complete" and "timing_contaminated" not in status and "gpu_sharing" not in status


def test_owned_gpu_memory_sums_only_this_process_group(monkeypatch):
    rows = "GPU-a, 10, 3000\nGPU-a, 11, 2000\nGPU-a, 99, 40000\nGPU-b, 12, 7000\n"
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=rows))
    monkeypatch.setattr(monitor.os, "getpgid", lambda pid: 500 if pid in (10, 11, 12) else 900)
    assert monitor.owned_gpu_memory_mib("GPU-a", 500) == 5000
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="GPU-a, 10, [N/A]\n"))
    assert monitor.owned_gpu_memory_mib("GPU-a", 500) is None

    def failed(*a: Any, **kw: Any) -> Any:
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(monitor.subprocess, "run", failed)
    assert monitor.owned_gpu_memory_mib("GPU-a", 500) is None


# --- the worker and the stage records ----------------------------------------


def test_shared_request_carries_the_mode_to_the_supervisor(tmp_path, monkeypatch):
    base = tmp_path / "jobs" / "fleet"
    directory = base / "inbox" / "20261011T000000Z-12345678"
    directory.mkdir(parents=True)
    spec = inbox.validate_spec(shared_specification(tmp_path))
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec)})
    inbox.transition(directory, "pending")
    monkeypatch.setattr(inbox.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=os.getpid()))
    inbox.start_request(directory, spec, {"uuid": "GPU-busy"}, host(), base)
    request = monitor.read_json(directory / "run" / "request.json")
    assert request["gpu_sharing"] == spec["gpu_sharing"]
    assert request["contention"]["policy"] == "continue"
    assert request["claims"].items() >= {"sharing": "shared", "memory_mib": 12000, "margin_mib": 8192, "min_free_memory_mb": 20192}.items()
    assert monitor.read_json(directory / "state.json")["gpu_sharing"] == spec["gpu_sharing"]


def test_shared_stages_record_the_mode_and_contaminated_timing(campaign):
    spec = inbox.validate_spec(shared_specification(campaign.parents[3]))
    monitor.write_json(campaign / "spec.json", spec)
    ready = monitor.read_json(campaign / "ready.json")
    monitor.write_json(campaign / "ready.json", {**ready, "spec_sha256": inbox.digest(spec)})
    assert inbox.campaign(campaign) == 0
    for stage in ("smoke", "full"):
        assert monitor.read_json(campaign / stage / "dispatch.json")["gpu_sharing"] == spec["gpu_sharing"]
        execution = monitor.read_json(campaign / stage / "execution.json")
        assert execution["gpu_sharing"] == spec["gpu_sharing"] and execution["timing_contaminated"] is True


def test_exclusive_stages_record_the_default_mode(campaign):
    assert inbox.campaign(campaign) == 0
    execution = monitor.read_json(campaign / "full" / "execution.json")
    assert execution["gpu_sharing"] == {"mode": "exclusive"} and execution["timing_contaminated"] is False
