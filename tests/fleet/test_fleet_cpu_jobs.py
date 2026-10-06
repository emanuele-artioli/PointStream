"""A cpu job is admitted on CPU headroom alone and claims no GPU."""
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import fleet, inbox, monitor
from tests.fleet.test_fleet_inbox import specification


def cpu_specification(data: Path) -> dict[str, Any]:
    return {**specification(data), "device": "cpu", "gpu_models": [], "gpu_memory_mib": 0, "cpu_threads": 4}


def host(**fields: Any) -> dict[str, Any]:
    busy = {"uuid": "GPU-busy", "name": "NVIDIA RTX A6000", "compute_processes": [{"pid": 1}],
            "memory_used_mib": 40000, "utilization_pct": 99, "memory_free_mib": 100}
    return {"alias": "gpu1", "host": "gpu1", "available": True, "python_available": True,
            "data_root_available": True, "cpu_headroom": 32, "gpus": [busy], **fields}


def test_cpu_spec_is_valid_and_gpu_spec_keeps_its_default(tmp_path):
    assert inbox.validate_spec(cpu_specification(tmp_path))["device"] == "cpu"
    assert inbox.validate_spec(specification(tmp_path))["device"] == "gpu"


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(gpu_models=["RTX A6000"]),
    lambda s: s.update(gpu_memory_mib=1024),
    lambda s: s.update(gpu_memory_mib=False),
    lambda s: s.update(contention={"policy": "stop"}),
    lambda s: s.update(device="tpu"),
    lambda s: s.update(cpu_threads=0),
])
def test_cpu_spec_cannot_describe_a_gpu(tmp_path, mutation):
    spec = cpu_specification(tmp_path)
    mutation(spec)
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec)


def test_gpu_spec_still_requires_gpu_memory(tmp_path):
    spec = specification(tmp_path)
    spec["gpu_memory_mib"] = 0
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec)


def test_cpu_job_is_admitted_while_every_gpu_is_busy(tmp_path):
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    assert inbox.eligible(spec, host()) is None
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        inbox.eligible(inbox.validate_spec(specification(tmp_path)), host())


@pytest.mark.parametrize("fields", [{"cpu_headroom": 4}, {"available": False}, {"data_root_available": False}])
def test_cpu_job_still_needs_cpu_headroom_and_a_usable_host(tmp_path, fields):
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    with pytest.raises(fleet.FleetError, match="No eligible CPU host"):
        inbox.eligible(spec, host(**fields))


def test_select_cpu_prefers_the_most_headroom():
    chosen = fleet.select_cpu([host(alias="gpu1", cpu_headroom=16), host(alias="gpu2", cpu_headroom=40)], cpu_threads=4)
    assert chosen["alias"] == "gpu2"


def test_cpu_request_asks_the_supervisor_for_threads_only(tmp_path, monkeypatch):
    base = tmp_path / "jobs" / "fleet"
    directory = base / "inbox" / "20261006T000000Z-12345678"
    directory.mkdir(parents=True)
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec)})
    inbox.transition(directory, "pending")
    launched: list[list[str]] = []

    def spawn(command: list[str], **kwargs: Any) -> Any:
        launched.append(command)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(inbox.subprocess, "Popen", spawn)
    inbox.start_request(directory, spec, None, host(), base)
    request = monitor.read_json(directory / "run" / "request.json")
    assert request["claims"]["gpu"] is None
    assert request["claims"]["cpu_threads"] == 4
    assert launched and monitor.read_json(directory / "state.json")["status"] == "running"
    assert monitor.read_json(directory / "environment.json")["gpu"] is None
