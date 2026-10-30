"""Inbox jobs keep only source.tar on NFS; workers unpack it host-locally."""
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import time
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import fleet, inbox, monitor, staging
from tests.fleet.test_fleet_cpu_jobs import cpu_specification, host
from tests.fleet.test_fleet_inbox import specification

JOB = "20261030T000000Z-12345678"
WORK = b"""import json,os,pathlib,sys
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
p.joinpath('result.json').write_text(json.dumps({'frames':int(sys.argv[-1])}))
"""
VALIDATE = b"""import json,os,pathlib
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
value=json.loads(p.joinpath('result.json').read_text())
pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed':value['frames']==2,'checks':['frame count']}))
"""


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    root = tmp_path / "local" / "pointstream"
    root.parent.mkdir()
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(root))
    # CI's temporary directory is on the runner's root disk, which real hosts refuse.
    monkeypatch.setattr(staging, "system_disk", lambda path: False)
    return root


def snapshot_tar(path: Path) -> tuple[Path, dict[str, Any]]:
    """A snapshot like _build_snapshot's: HEAD, then a re-added edit, plus a deletion."""
    def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with tarfile.open(path, "w") as tar:
        add(tar, "work.py", b"stale")
        add(tar, "validate.py", VALIDATE)
        add(tar, "pkg/removed.py", b"deleted in the worktree")
        add(tar, "work.py", WORK)
    metadata = {"git_head": "a" * 40, "deleted_paths": ["pkg/removed.py"], "snapshot_sha256": inbox.file_digest(path)}
    return path, metadata


def legacy_identity(archive: Path, metadata: dict[str, Any], where: Path) -> str:
    """The identity the old NFS publication computed for the same archive."""
    where.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(where, filter="fully_trusted")
    for rel in metadata["deleted_paths"]:
        (where / rel).unlink()
    return inbox.source_identity(where)


def published_archive_job(tmp_path: Path, spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = tmp_path / "jobs" / "fleet"
    prepared = inbox.rpc(base, "prepare", {"job_id": JOB, "source_archive": True})
    directory = Path(prepared["directory"])
    assert prepared["source_archive"] is True and not (directory / "source").exists()
    _, metadata = snapshot_tar(directory / inbox.SOURCE_ARCHIVE)
    inbox.rpc(base, "publish", {"job_id": JOB, "spec": spec, "snapshot": metadata, "source_archive": True})
    return directory, metadata


def test_send_snapshot_without_extraction_keeps_only_the_verified_archive(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"archive")
    commands: list[list[str]] = []
    def remote(host, command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="a" * 64 + " source.tar", stderr="")
    monkeypatch.setattr(fleet.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stderr=b""))
    monkeypatch.setattr(fleet, "_ssh", remote)
    fleet._send_snapshot("gpu2", source, "/inbox/job/source", "a" * 64, timeout=90, extract=False)
    assert commands == [["sha256sum", "--", "/inbox/job/source.tar"]]


def test_archive_publication_keeps_no_tree_on_nfs_and_the_legacy_identity(tmp_path, local_root):
    directory, metadata = published_archive_job(tmp_path, inbox.validate_spec(specification(tmp_path)))
    ready = monitor.read_json(directory / "ready.json")
    archive = directory / inbox.SOURCE_ARCHIVE
    assert ready["source_archive"] == inbox.SOURCE_ARCHIVE
    assert ready["snapshot_sha256"] == metadata["snapshot_sha256"] and ready["git_head"] == "a" * 40
    assert ready["source_sha256"] == legacy_identity(archive, metadata, tmp_path / "legacy")
    assert not (directory / "source").exists()
    assert sorted(p.name for p in directory.iterdir()) == ["events", "ready.json", "source.tar", "spec.json", "state.json"]
    assert stat.S_IMODE(archive.stat().st_mode) == 0o444
    # The identity check left nothing behind on the management host.
    assert list((local_root / "sources").iterdir()) == []


def test_archive_publication_rejects_a_changed_archive(tmp_path, local_root):
    base = tmp_path / "jobs" / "fleet"
    directory = Path(inbox.rpc(base, "prepare", {"job_id": JOB, "source_archive": True})["directory"])
    _, metadata = snapshot_tar(directory / inbox.SOURCE_ARCHIVE)
    metadata["snapshot_sha256"] = "0" * 64
    spec = inbox.validate_spec(specification(tmp_path))
    with pytest.raises(fleet.FleetError, match="checksum"):
        inbox.rpc(base, "publish", {"job_id": JOB, "spec": spec, "snapshot": metadata, "source_archive": True})
    assert not (directory / "ready.json").exists()


def test_legacy_prepare_still_creates_the_source_tree(tmp_path):
    prepared = inbox.rpc(tmp_path / "jobs" / "fleet", "prepare", {"job_id": JOB})
    assert "source_archive" not in prepared and (Path(prepared["directory"]) / "source").is_dir()


def test_worker_unpacks_archive_on_local_disk_and_runs_from_there(tmp_path, local_root, monkeypatch):
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    directory, _ = published_archive_job(tmp_path, spec)
    launched: list[dict[str, Any]] = []
    real = inbox.subprocess.Popen
    def spawn(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "tar":
            return real(command, **kwargs)
        launched.append(kwargs)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(inbox.subprocess, "Popen", spawn)
    inbox.start_request(directory, spec, None, host(), tmp_path / "jobs" / "fleet")
    local = local_root / "sources" / JOB
    assert (local / "work.py").read_bytes() == WORK and not (local / "pkg" / "removed.py").exists()
    assert os.readlink(directory / "source") == str(local)
    assert monitor.read_json(directory / "run" / "request.json")["cwd"] == str(local)
    assert launched[0]["cwd"] == local and launched[0]["env"]["PYTHONPATH"] == str(local)
    assert monitor.read_json(directory / "state.json")["status"] == "running"


def test_worker_refuses_an_archive_changed_after_publication(tmp_path, local_root, monkeypatch):
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    directory, _ = published_archive_job(tmp_path, spec)
    archive = directory / inbox.SOURCE_ARCHIVE
    archive.chmod(0o644)
    snapshot_tar(archive)
    with tarfile.open(archive, "a") as tar:
        info = tarfile.TarInfo("extra.py")
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(fleet.FleetError, match="checksum"):
        inbox.start_request(directory, spec, None, host(), tmp_path / "jobs" / "fleet")
    assert not (directory / "run").exists()


def test_campaign_runs_an_archive_job_from_the_local_tree(tmp_path, local_root, monkeypatch):
    spec = inbox.validate_spec(specification(tmp_path))
    directory, _ = published_archive_job(tmp_path, spec)
    manifest = monitor.read_json(directory / "ready.json")
    inbox.local_source(directory, manifest)
    (directory / "run").mkdir()
    monkeypatch.setenv("PS_JOB_DIR", str(directory / "run"))
    assert inbox.campaign(directory) == 0
    assert json.loads((directory / "full" / "result.json").read_text())["frames"] == 8
    assert monitor.read_json(directory / "gate.json")["source_sha256"] == manifest["source_sha256"]


def test_terminal_job_releases_its_local_source_but_keeps_the_archive(tmp_path, local_root, monkeypatch):
    spec = inbox.validate_spec(cpu_specification(tmp_path))
    directory, _ = published_archive_job(tmp_path, spec)
    local = inbox.local_source(directory, monitor.read_json(directory / "ready.json"))
    inbox.transition(directory, "complete")
    monkeypatch.setattr(inbox, "local_probe", lambda _: pytest.fail("terminal job probed a host"))
    inbox.worker_tick(tmp_path / "jobs" / "fleet", "gpu1")
    assert not local.exists() and (directory / inbox.SOURCE_ARCHIVE).is_file()
    heartbeat = monitor.read_json(tmp_path / "jobs" / "fleet" / "workers" / "gpu1" / "heartbeat.json")
    assert "source_archive" in heartbeat["capabilities"]


def test_live_supervisor_keeps_its_local_source(tmp_path, local_root):
    directory = tmp_path / "inbox" / JOB
    local = local_root / "sources" / JOB
    local.mkdir(parents=True)
    monitor.write_json(directory / "run" / "supervisor.json", {"pid": os.getpid(), "proc_start_time": inbox.get_process_start_time(os.getpid())})
    inbox.release_local_source(directory, local_root)
    assert local.is_dir()


@pytest.mark.parametrize("capable, confirmed, expected", [
    (True, True, True), (True, False, False), (False, True, False),
])
def test_submit_uses_archive_mode_only_when_workers_and_release_support_it(tmp_path, monkeypatch, capable, confirmed, expected):
    from argparse import Namespace
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(specification(tmp_path)))
    snapshot = tmp_path / "snapshot.tar"
    snapshot.write_bytes(b"snapshot")
    monkeypatch.setattr(inbox, "load_config", lambda _: {"hosts": ["gpu1"]})
    monkeypatch.setattr(fleet, "_build_snapshot", lambda *a, **kw: (snapshot, "hash", {}))
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(fleet, "_send_snapshot", lambda *a, **kw: sent.append(kw))
    payloads: dict[str, dict[str, Any]] = {}
    def remote(config, action, payload):
        payloads[action] = payload
        if action == "health":
            heartbeat = {"updated": time.time(), **({"capabilities": ["source_archive"]} if capable else {})}
            return {"workers": {"gpu1": heartbeat}}
        if action == "prepare":
            return {"directory": "/data/jobs/fleet/inbox/" + payload["job_id"], **({"source_archive": True} if confirmed else {})}
        return {"status": "pending"}
    monkeypatch.setattr(inbox, "remote_rpc", remote)
    args = Namespace(state_dir=tmp_path / "state", spec=spec_path, include_change=[], include_untracked=[], chat_id="chat")
    result = inbox.submit(args, tmp_path)
    assert payloads["prepare"].get("source_archive", False) is capable
    assert payloads["publish"].get("source_archive", False) is expected
    assert sent[0].get("extract", True) is (not expected)
    assert result["source_archive"] is expected


def test_selftest_spec_is_valid_and_allows_a_cold_nfs_torch_import(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.setattr(inbox, "load_config", lambda _: {"hosts": ["gpu6"]})
    identity = tmp_path / "input.json"
    identity.write_text("{}")
    monkeypatch.setattr(inbox, "remote_rpc", lambda *a: {"path": str(identity), "sha256": inbox.file_digest(identity)})
    monkeypatch.setattr(inbox, "submit", lambda args, root: json.loads(args.spec.read_text()))
    spec = inbox.validate_spec(inbox.selftest_submit(Namespace(state_dir=tmp_path / "state", hosts=["gpu6"]), tmp_path))
    assert spec["smoke"]["seconds"] >= 3 * 61 - 3 and spec["full"]["seconds"] >= spec["smoke"]["seconds"]
    assert spec["deadline_epoch"] - time.time() > spec["budget_seconds"]
