"""Admission must fail closed and rank only demonstrably idle devices."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import fleet


def test_local_code_snapshot_keeps_data_root_run_paths():
    host = {"data_root": "/data/pointstream"}
    job = "20261001T172825Z-77156054"
    assert fleet.snapshot_directory(host, job, None) == f"/data/pointstream/jobs/fleet/snapshots/{job}"
    assert fleet.snapshot_directory(host, job, "/var/tmp/owned-code") == f"/var/tmp/owned-code/{job}"
    assert fleet._remote_path(host, "runs", job) == f"/data/pointstream/jobs/fleet/runs/{job}"


@pytest.mark.parametrize("path", ["relative", "/", "/var/tmp/../foreign"])
def test_local_snapshot_rejects_ambiguous_roots(path):
    with pytest.raises(fleet.FleetError):
        fleet.snapshot_directory({"data_root": "/data"}, "20261001T172825Z-77156054", path)


def device(
    name: str = "NVIDIA RTX 6000 Ada Generation",
    *,
    uuid: str = "GPU-a",
    used: int = 1,
    free: int = 48000,
    util: int = 0,
    pids: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "name": name,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "memory_total_mib": 49140,
        "utilization_pct": util,
        "compute_processes": pids or [],
    }


def host(alias: str, *gpus: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {
        "alias": alias,
        "host": alias + ".example",
        "available": True,
        "gpus": list(gpus),
        "cpu_headroom": 32,
        "python_available": True,
        "data_root_available": True,
        "tools": {"ffmpeg": "/opt/ffmpeg"},
        **overrides,
    }


def test_selection_prefers_compatible_gpu_and_rejects_busy_gpu_even_at_zero_utilization() -> None:
    a6000 = device("NVIDIA RTX A6000", uuid="GPU-a6000")
    occupied = device("NVIDIA RTX 6000 Ada Generation", uuid="GPU-busy", free=48000, pids=[{"pid": 7}])
    ada = device("NVIDIA RTX 6000 Ada Generation", uuid="GPU-ada")
    selected_host, selected_gpu = fleet.select_gpu(
        [host("gpu2", a6000), host("gpu5", occupied, ada)], required_memory_mib=12000, cpu_threads=8
    )
    assert selected_host["alias"] == "gpu5"
    assert selected_gpu["uuid"] == "GPU-ada"


def test_selection_honors_explicit_measured_hardware_preference() -> None:
    selected_host, selected_gpu = fleet.select_gpu(
        [
            host("gpu5", device("NVIDIA RTX 6000 Ada Generation", uuid="GPU-ada")),
            host("gpu2", device("NVIDIA RTX A6000", uuid="GPU-a6000")),
        ],
        required_memory_mib=12000,
        cpu_threads=8,
        prefer_gpu_names=("RTX A6000",),
    )
    assert selected_host["alias"] == "gpu2"
    assert selected_gpu["uuid"] == "GPU-a6000"


@pytest.mark.parametrize(
    "busy",
    [
        device(used=512),
        device(util=7),
        device(free=16000),
        device(pids=[{"pid": 12}]),
    ],
)
def test_selection_refuses_devices_that_are_not_clearly_idle(busy: dict[str, Any]) -> None:
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([host("gpu5", busy)], required_memory_mib=16000, cpu_threads=8)


def test_selection_reserves_an_empty_gpu_already_claimed_by_another_pointstream_job() -> None:
    claimed = {**device(), "pointstream_claimed": True}
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([host("gpu5", claimed)], required_memory_mib=1, cpu_threads=1)


def test_selection_refuses_failed_probes_missing_inputs_and_cpu_oversubscription() -> None:
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([{"alias": "gpu6", "available": False, "error": "timeout"}], required_memory_mib=1, cpu_threads=1)
    with pytest.raises(fleet.FleetError, match="No eligible GPU"):
        fleet.select_gpu([host("gpu5", device(), cpu_headroom=8)], required_memory_mib=1, cpu_threads=8)
    with pytest.raises(fleet.FleetError, match="missing commands"):
        fleet.select_gpu([host("gpu5", device())], required_memory_mib=1, cpu_threads=1, required_commands=("vvdecapp",))
    with pytest.raises(fleet.FleetError, match="absolute remote paths"):
        fleet.select_gpu([host("gpu5", device())], required_memory_mib=1, cpu_threads=1, required_paths=("relative/input",))


def test_required_command_preflight_checks_path_names_and_absolute_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = [
        host("gpu5", device(), python="/opt/pointstream/bin/python"),
        host("gpu6", device(uuid="GPU-b"), python="/opt/pointstream/bin/python"),
    ]
    seen: list[str] = []

    def check(alias: str, python: str, script: str) -> dict[str, Any]:
        seen.append(alias)
        assert python == "/opt/pointstream/bin/python"
        assert "ffmpeg" in script and "/opt/pointstream/bin/python" in script
        return {"commands": {"ffmpeg": alias == "gpu5", "/opt/pointstream/bin/python": True}}

    monkeypatch.setattr(fleet, "_remote_json", check)
    checked = fleet.inspect_required_commands(hosts, ("ffmpeg", "/opt/pointstream/bin/python"))
    assert set(seen) == {"gpu5", "gpu6"}
    assert checked[0]["required_commands_available"]["/opt/pointstream/bin/python"] is True
    selected, _ = fleet.select_gpu(
        checked,
        required_memory_mib=1,
        cpu_threads=1,
        required_commands=("ffmpeg", "/opt/pointstream/bin/python"),
    )
    assert selected["alias"] == "gpu5"


def test_required_command_preflight_fails_closed_on_malformed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = [host("gpu5", device(), python="/opt/pointstream/bin/python")]
    monkeypatch.setattr(fleet, "_remote_json", lambda *args, **kwargs: {"commands": {}})
    checked = fleet.inspect_required_commands(hosts, ("ffmpeg",))
    assert checked[0]["available"] is False
    with pytest.raises(fleet.FleetError, match="No eligible GPU.*incomplete required-command probe"):
        fleet.select_gpu(checked, required_memory_mib=1, cpu_threads=1, required_commands=("ffmpeg",))


@pytest.mark.parametrize(
    ("returncode", "stdout", "available"),
    [
        (255, "", False),
        (0, "not json", False),
        (0, json.dumps({"probe_error": "nvidia-smi failed"}), False),
        (0, json.dumps({"gpus": [{"uuid": "GPU-a"}]}), False),
    ],
)
def test_probe_fails_closed_for_unreachable_malformed_or_incomplete_hosts(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, available: bool
) -> None:
    monkeypatch.setattr(
        fleet,
        "_ssh",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout, stderr="ssh failed"),
    )
    result = fleet.probe_host("gpu5")
    assert result["available"] is available
    if not available:
        assert result["gpus"] == []


def test_snapshot_contains_only_selected_tracked_edits_and_explicit_untracked_files(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "fleet-test@example.invalid")
    git("config", "user.name", "Fleet test")
    (tmp_path / "tracked.txt").write_text("base")
    (tmp_path / "other.txt").write_text("other base")
    (tmp_path / "deleted.txt").write_text("remove")
    git("add", "tracked.txt", "other.txt", "deleted.txt")
    git("commit", "-qm", "base")
    (tmp_path / "tracked.txt").write_text("working copy")
    (tmp_path / "other.txt").write_text("unselected working copy")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "included.py").write_text("snapshot file")
    (tmp_path / "ignored.py").write_text("exclude")
    archive, digest, meta = fleet._build_snapshot(
        tmp_path,
        include_changes=("tracked.txt", "deleted.txt"),
        include_untracked=("included.py",),
    )
    try:
        assert len(digest) == 64
        assert meta["deleted_paths"] == ["deleted.txt"]
        assert meta["tracked_changes_included"] == ["deleted.txt", "tracked.txt"]
        assert meta["tracked_changes_excluded"] == ["other.txt"]
        assert meta["included_untracked_sha256"]["included.py"]
        assert meta["untracked_excluded_sample"] == ["ignored.py"]
        assert meta["untracked_excluded_count"] == 1
        files = tarfile_open(archive)
        assert files["tracked.txt"] == b"working copy"
        assert files["other.txt"] == b"other base"
        assert files["included.py"] == b"snapshot file"
        assert "ignored.py" not in files
        for deleted in meta["deleted_paths"]:
            files.pop(deleted, None)
        assert "deleted.txt" not in files
    finally:
        archive.unlink(missing_ok=True)

    base_archive, _, base_meta = fleet._build_snapshot(tmp_path)
    try:
        base_files = tarfile_open(base_archive)
        assert base_files["tracked.txt"] == b"base"
        assert base_files["deleted.txt"] == b"remove"
        assert base_meta["tracked_changes_included"] == []
        assert base_meta["tracked_changes_excluded"] == ["deleted.txt", "other.txt", "tracked.txt"]
    finally:
        base_archive.unlink(missing_ok=True)


def test_snapshot_transfer_and_extraction_allow_slow_shared_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "snapshot.tar"
    source.write_bytes(b"snapshot")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    ssh_calls: list[tuple[list[str], float]] = []
    upload_calls: list[dict[str, Any]] = []

    def fake_ssh(host: str, command: list[str], *, timeout: float = 30) -> SimpleNamespace:
        ssh_calls.append((command, timeout))
        stdout = f"{digest} archive.tar\n" if command[0] == "sha256sum" else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        upload_calls.append(kwargs)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(fleet, "_ssh", fake_ssh)
    monkeypatch.setattr(fleet.subprocess, "run", fake_run)

    fleet._send_snapshot("gpu6", source, "/data/snapshots/job", digest)

    assert upload_calls[0]["timeout"] == fleet.SNAPSHOT_TRANSFER_TIMEOUT_SECONDS
    extract_command, extract_timeout = ssh_calls[1]
    assert extract_command[0] == "tar"
    assert extract_timeout == fleet.SNAPSHOT_TRANSFER_TIMEOUT_SECONDS


def tarfile_open(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r") as tar:
        files: dict[str, bytes] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            files[member.name] = extracted.read()
        return files


def test_remote_launcher_python_is_well_formed_and_uses_the_existing_supervisor() -> None:
    import shlex

    command = fleet._launch_script(
        data_root="/data/pointstream",
        job_id="20260926T120000Z-12345678",
        source_dir="/data/pointstream/jobs/fleet/snapshots/id",
        run_dir="/data/pointstream/jobs/fleet/runs/id",
        remote_python="/env/bin/python",
        gpu_uuid="GPU-ada",
        cpu_threads=1,
        budget_hours=1,
        command=["/usr/bin/true"],
        requirements=[],
        require_paths=[],
        snapshot_sha256="abc",
        git_metadata={},
        selected_gpu={"uuid": "GPU-ada"},
        estimated_gpu_memory_mib=1024,
        idle_memory_mib=256,
        idle_utilization_pct=5,
    )
    tokens = shlex.split(command)
    compile(tokens[3], "remote fleet launcher", "exec")
    assert "experiments.jobs.monitor" in tokens[3]
    assert "--claim-gpu" in tokens[3]
    assert "monotonic()+60" in tokens[3]
    assert "within 60 seconds" in tokens[3]
