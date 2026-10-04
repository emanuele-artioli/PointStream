import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experiments.jobs import artifacts, fleet


@pytest.fixture
def recorded_job(tmp_path, monkeypatch):
    job = "20261004T000000Z-12345678"
    root = tmp_path / "jobs/fleet/runs" / job
    (root / "smoke").mkdir(parents=True)
    (root / "smoke/result.json").write_text('{"passed":false}')
    def transport(alias, argv, *, timeout):
        assert alias == "gpu5" and timeout == artifacts.READ_TIMEOUT
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    monkeypatch.setattr(fleet, "_ssh", transport)
    return {"job_id": job, "host_alias": "gpu5", "run_dir": str(root)}, root


def test_export_retains_source_and_verifies_transport(recorded_job, tmp_path):
    record, root = recorded_job
    before = (root / "smoke/result.json").read_bytes()
    target = tmp_path / "export"
    artifacts.export(record, ["smoke/result.json", "smoke/missing.json"], target)
    receipt = json.loads((target / "export-receipt.json").read_text())
    assert receipt["read_only"] and receipt["files"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["files"][1]["status"] == "missing"
    assert (target / "smoke/result.json").read_bytes() == before == (root / "smoke/result.json").read_bytes()
    assert (target / "smoke/result.json").stat().st_mode & 0o222 == 0
    with pytest.raises(fleet.FleetError, match="already exists"):
        artifacts.export(record, ["smoke/result.json"], target)


@pytest.mark.parametrize("path", ["../secret.json", "/etc/passwd", "source/secrets.json", "smoke/work.py"])
def test_arbitrary_files_are_not_exportable(path):
    with pytest.raises(fleet.FleetError):
        artifacts.validate_paths([path])


def test_symlink_escape_is_rejected(recorded_job, tmp_path):
    record, root = recorded_job
    secret = tmp_path / "outside.json"
    secret.write_text("private")
    (root / "smoke/escape.json").symlink_to(secret)
    with pytest.raises(fleet.FleetError, match="symlink escapes"):
        artifacts.export(record, ["smoke/escape.json"], tmp_path / "export")


def test_oversized_logs_have_no_complete_hash(recorded_job, tmp_path):
    record, root = recorded_job
    (root / "smoke/command.log").write_bytes(b"x" * (artifacts.MAX_BYTES + 5))
    target = tmp_path / "export"
    artifacts.export(record, ["smoke/command.log"], target)
    row = json.loads((target / "export-receipt.json").read_text())["files"][0]
    assert row["truncated"] is True and row["sha256"] is None
    assert (target / "smoke/command.log").stat().st_size == artifacts.MAX_BYTES


def test_read_timeout_publishes_no_partial_export(recorded_job, tmp_path, monkeypatch):
    record, root = recorded_job
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("read", artifacts.READ_TIMEOUT)
    monkeypatch.setattr(fleet, "_ssh", timeout)
    target = tmp_path / "export"
    with pytest.raises(subprocess.TimeoutExpired):
        artifacts.export(record, ["smoke/result.json"], target)
    assert not target.exists()
