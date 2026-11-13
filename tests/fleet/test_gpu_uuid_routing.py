"""Physical UUID admission is opt-in; legacy required_commands blocks it safely."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import pytest
from experiments.jobs import fleet, inbox, monitor
from tests.fleet.test_fleet_inbox import specification

TARGET = "GPU-11111111-2222-3333-4444-555555555555"
OTHER = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SENTINEL = "ps-fleet-gpu-uuid-v1-6ccde7f4"


def pinned_spec(tmp_path):
    return {**specification(tmp_path), "gpu_uuid": TARGET, "required_commands": [SENTINEL]}


def test_uuid_requires_legacy_safe_command(tmp_path):
    spec = pinned_spec(tmp_path)
    spec["required_commands"] = []
    with pytest.raises(fleet.FleetError, match="routing sentinel"):
        inbox.validate_spec(spec)


@pytest.mark.parametrize(
    "value", [None, "", False, [TARGET], "0", OTHER.upper(), "GPU-11111111222233334444555555555555"]
)
def test_rejects_noncanonical_uuid(tmp_path, value):
    spec = pinned_spec(tmp_path)
    spec["gpu_uuid"] = value
    with pytest.raises(fleet.FleetError, match="gpu_uuid"):
        inbox.validate_spec(spec)


def gpu(identifier, **fields):
    return {
        "uuid": identifier,
        "name": "RTX 6000 Ada",
        "compute_processes": [],
        "memory_used_mib": 0,
        "memory_free_mib": 48000,
        "utilization_pct": 0,
        **fields,
    }


def host(gpus):
    return {
        "alias": "gpu1",
        "host": "gpu1",
        "available": True,
        "python_available": True,
        "data_root_available": True,
        "cpu_headroom": 32,
        "gpus": gpus,
    }


def test_exact_target_selected_without_mutating_host_or_environment(tmp_path, monkeypatch):
    spec = inbox.validate_spec(pinned_spec(tmp_path))
    inventory = host([gpu(OTHER), gpu(TARGET)])
    monkeypatch.setenv("PATH", str(tmp_path))
    before = dict(os.environ)
    assert inbox.eligible(spec, inventory)["uuid"] == TARGET
    assert [g["uuid"] for g in inventory["gpus"]] == [OTHER, TARGET]
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "target",
    [
        None,
        {"compute_processes": [{"pid": 1}]},
        {"memory_free_mib": 1},
        {"pointstream_claimed": True},
    ],
)
def test_missing_busy_claimed_or_low_memory_target_never_falls_back(tmp_path, target):
    spec = inbox.validate_spec(pinned_spec(tmp_path))
    gpus = [gpu(OTHER)]
    if target is not None:
        gpus.append(gpu(TARGET, **target))
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        inbox.eligible(spec, host(gpus))


def test_model_filter_and_unpinned_defaults_remain_active(tmp_path):
    spec = inbox.validate_spec(specification(tmp_path))
    assert "gpu_uuid" not in spec
    assert inbox.eligible(spec, host([gpu(OTHER)]))["uuid"] == OTHER
    spec = inbox.validate_spec(pinned_spec(tmp_path))
    spec["gpu_models"] = ["A6000"]
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        inbox.eligible(spec, host([gpu(TARGET)]))


def test_cpu_uuid_is_rejected_and_valid_spec_roundtrips(tmp_path):
    spec = pinned_spec(tmp_path)
    once = inbox.validate_spec(spec)
    assert inbox.validate_spec(once) == once
    spec.update(device="cpu", gpu_memory_mib=0)
    with pytest.raises(fleet.FleetError, match="gpu_uuid"):
        inbox.validate_spec(spec)


@pytest.mark.parametrize("damage", ["payload", "nonexec", "extra", "symlink", "owner"])
def test_only_exact_owned_private_marker_is_accepted(tmp_path, monkeypatch, damage):
    code = tmp_path / "inbox.py"
    code.write_text("")
    directory = tmp_path / "uuid-routing-bin"
    directory.mkdir()
    token = directory / SENTINEL
    token.write_bytes(inbox.GPU_UUID_SENTINEL_BYTES)
    token.chmod(493)
    monkeypatch.setattr(inbox, "__file__", str(code))
    if damage == "payload":
        token.write_text("different")
    elif damage == "nonexec":
        token.chmod(420)
    elif damage == "extra":
        (directory / "python").write_text("shadow")
    elif damage == "symlink":
        token.unlink()
        token.symlink_to(code)
    else:
        monkeypatch.setattr(inbox.os, "getuid", lambda: token.stat().st_uid + 1)
    with pytest.raises(fleet.FleetError, match="sentinel"):
        inbox.uuid_routing_command()


def test_client_rejects_legacy_capability_before_snapshot_or_publication(tmp_path, monkeypatch):
    import argparse
    import time

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(pinned_spec(tmp_path)))
    calls = []
    monkeypatch.setattr(inbox, "snapshot_root", lambda root, selected: root)
    monkeypatch.setattr(inbox, "load_config", lambda _: {"hosts": ["gpu1"]})

    def rpc(config, action, payload):
        calls.append(action)
        return {"workers": {"gpu1": {"updated": time.time(), "capabilities": ["source_archive"]}}}

    monkeypatch.setattr(inbox, "remote_rpc", rpc)
    monkeypatch.setattr(
        fleet, "_build_snapshot", lambda *a, **k: pytest.fail("snapshot before capability")
    )
    with pytest.raises(fleet.FleetError, match="does not advertise GPU UUID routing"):
        inbox.submit(
            argparse.Namespace(
                spec=path, state_dir=tmp_path, source_worktree=None, snapshot_transfer_seconds=None
            ),
            tmp_path,
        )
    assert calls == ["health"]
    assert not list(tmp_path.glob("20*.json"))


def legacy_eligible():
    raw = (Path(__file__).parent / "fixtures/legacy_eligible.py.txt").read_text()
    body = raw.split("\n", 1)[1]
    assert (
        hashlib.sha256(body.encode()).hexdigest()
        == "2caabfccb7e738a36f4c077397ab39ee1661aa1db2920cf2e0f403c8f44c6524"
    )
    namespace = {"Any": Any, "fleet": fleet, "staging": inbox.staging}
    exec(compile(raw, "<installed-5dbdc-eligible>", "exec"), namespace)
    return namespace["eligible"]


def test_real_legacy_worker_rejects_before_request_claims_or_models(tmp_path, monkeypatch):
    base = tmp_path / "fleet"
    directory = base / "inbox" / "20261113T000000Z-12345678"
    directory.mkdir(parents=True)
    (base / "workers/gpu1").mkdir(parents=True)
    spec = inbox.validate_spec(pinned_spec(tmp_path))
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "state.json", {"status": "pending"})
    monitor.write_json(directory / "ready.json", {})
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(inbox, "eligible", legacy_eligible())
    monkeypatch.setattr(inbox, "local_probe", lambda alias: host([gpu(OTHER), gpu(TARGET)]))
    monkeypatch.setattr(inbox, "get_process_start_time", lambda pid: 1)
    monkeypatch.setattr(
        inbox, "start_request", lambda *a, **k: pytest.fail("legacy started request/claims/models")
    )
    monkeypatch.setattr(fleet, "select_gpu", lambda *a, **k: pytest.fail("legacy selected a GPU"))
    inbox.worker_tick(base, "gpu1")
    assert monitor.read_json(directory / "state.json")["status"] == "pending"
    assert (
        "required executable unavailable"
        in monitor.read_json(base / "workers/gpu1/admission.json")["reason"]
    )
    assert not (directory / "owner").exists() and (not (directory / "run").exists())


def test_worker_bundle_keeps_private_marker(tmp_path):
    import io
    import tarfile

    name = "experiments/jobs/uuid-routing-bin/" + SENTINEL
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        entry = tarfile.TarInfo(name)
        entry.mode = 493
        entry.size = len(inbox.GPU_UUID_SENTINEL_BYTES)
        archive.addfile(entry, io.BytesIO(inbox.GPU_UUID_SENTINEL_BYTES))
    bundle, identities = inbox.worker_bundle(source)
    try:
        assert name in identities
        with tarfile.open(bundle) as archive:
            assert archive.getmember(name).mode == 493
    finally:
        bundle.unlink()
