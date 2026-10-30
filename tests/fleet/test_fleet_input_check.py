"""Publication and admission never stream large inputs; the campaign hashes every byte.

On 2026-10-30 the publish RPC hashed 8.2 GB over NFS at ~8 MB/s and died at the
client's 120 s SSH limit, and admission hashed 10.3 GB inside the worker scan,
freezing its heartbeat for up to 576 s. Both now run bounded checks; every byte
is still hashed before a stage's command runs.
"""
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import fleet, inbox, monitor, staging
from tests.fleet.test_fleet_cpu_jobs import cpu_specification, host
from tests.fleet.test_fleet_inbox import specification

JOB = "20261030T000000Z-87654321"
WORK = """import json,os,pathlib,sys
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
unet=pathlib.Path(sys.argv[sys.argv.index('--unet')+1]).read_bytes()
p.joinpath('result.json').write_text(json.dumps({'frames':int(sys.argv[-1]),'unet':len(unet)}))
"""
VALIDATE = """import json,os,pathlib
p=pathlib.Path(os.environ['PS_STAGE_DIR'])
value=json.loads(p.joinpath('result.json').read_text())
pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps({'passed':value['frames']==2,'checks':['frame count']}))
"""
# Staged through the verified cache, staged without the opt-in, or read directly from NFS.
MODES = ["staged-opt-in", "staged", "direct"]


def large_input_spec(tmp_path: Path, mode: str, make: Any = specification) -> tuple[dict[str, Any], Path]:
    """The base spec plus a 'large' weight: 4 KiB against a 1 KiB precheck limit."""
    spec = make(tmp_path)
    weight = tmp_path / "weights" / "unet.bin"
    weight.parent.mkdir()
    weight.write_bytes(b"w" * 4096)
    item = {"path": str(weight), "sha256": inbox.file_digest(weight)}
    spec["inputs"].append(item)
    if mode == "direct":
        spec["arguments"] = ["--unet", str(weight), *spec["arguments"]]
    else:
        spec["staged_inputs"] = [{"name": "unet", **item}]
        spec["arguments"] = ["--unet", "{staged:unet}", *spec["arguments"]]
        spec["verify_staged_inputs"] = mode == "staged-opt-in"
    return inbox.validate_spec(spec), weight


@pytest.fixture(autouse=True)
def small_limit(monkeypatch):
    monkeypatch.setattr(inbox, "PRECHECK_HASH_BYTES", 1024)


@pytest.fixture
def local(tmp_path, monkeypatch):
    root = tmp_path / "local"
    root.mkdir()
    monkeypatch.setattr(staging, "local_root", lambda: root)
    monkeypatch.setattr(staging, "available_bytes", lambda root: 10**12)
    return root


def forbid_reading(monkeypatch, weight: Path) -> None:
    """Fail if any hashing or staging path reads the large file (cold cache included)."""
    def guard(real):
        def call(path, *args):
            assert Path(path).resolve() != weight.resolve(), f"large input read: {path}"
            return real(path, *args)
        return call
    monkeypatch.setattr(inbox, "file_digest", guard(inbox.file_digest))
    monkeypatch.setattr(staging, "_hash_file", guard(staging._hash_file))
    monkeypatch.setattr(staging, "_copy_verified", guard(staging._copy_verified))


def publish(tmp_path: Path, spec: dict[str, Any]) -> Path:
    base = tmp_path / "jobs" / "fleet"
    directory = Path(inbox.rpc(base, "prepare", {"job_id": JOB})["directory"])
    (directory / "source" / "work.py").write_text(WORK)
    (directory / "source" / "validate.py").write_text(VALIDATE)
    inbox.rpc(base, "publish", {"job_id": JOB, "spec": spec, "snapshot": {"git_head": "a" * 40}})
    return directory


@pytest.mark.parametrize("mode", MODES)
def test_publish_reads_no_byte_of_a_large_input_and_records_the_deferral(tmp_path, monkeypatch, mode):
    spec, weight = large_input_spec(tmp_path, mode)
    forbid_reading(monkeypatch, weight)
    directory = publish(tmp_path, spec)
    check = monitor.read_json(directory / "ready.json")["input_check"]
    assert check["deferred"] == [str(weight)] and check["deferred_bytes"] == 4096
    assert check["hashed"] == [spec["inputs"][0]["path"]] and check["deferred_to"] == "campaign"


def test_publish_still_rejects_a_wrong_digest_within_the_limit(tmp_path):
    spec, _ = large_input_spec(tmp_path, "staged-opt-in")
    spec["inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(fleet.FleetError, match="input identity changed"):
        publish(tmp_path, spec)
    assert not (tmp_path / "jobs" / "fleet" / "inbox" / JOB / "ready.json").exists()


@pytest.mark.parametrize("field", ["inputs", "staged_inputs"])
def test_publish_rejects_a_missing_input_without_hashing(tmp_path, field):
    spec, weight = large_input_spec(tmp_path, "staged-opt-in")
    spec[field][-1] = {**spec[field][-1], "path": str(weight.with_name("missing.bin"))}
    with pytest.raises(fleet.FleetError, match="under the external data root"):
        publish(tmp_path, spec)


@pytest.mark.parametrize("mode", MODES)
def test_admission_reads_no_byte_of_a_large_input(tmp_path, monkeypatch, mode):
    spec, weight = large_input_spec(tmp_path, mode, cpu_specification)
    directory = publish(tmp_path, spec)
    forbid_reading(monkeypatch, weight)
    monkeypatch.setattr(inbox.subprocess, "Popen", lambda command, **kwargs: SimpleNamespace(pid=os.getpid()))
    inbox.start_request(directory, spec, None, host(), tmp_path / "jobs" / "fleet")
    assert monitor.read_json(directory / "state.json")["status"] == "running"


def run_campaign(directory: Path, monkeypatch) -> int:
    (directory / "run").mkdir()
    monkeypatch.setenv("PS_JOB_DIR", str(directory / "run"))
    return inbox.campaign(directory)


@pytest.mark.parametrize("mode", MODES)
def test_deferred_input_is_hashed_and_consumed_by_the_campaign(tmp_path, monkeypatch, local, mode):
    spec, weight = large_input_spec(tmp_path, mode)
    directory = publish(tmp_path, spec)
    assert run_campaign(directory, monkeypatch) == 0
    assert monitor.read_json(directory / "full" / "result.json")["unet"] == 4096
    staged = monitor.read_json(directory / "smoke" / "staging.json")["inputs"]
    assert [row["mode"] for row in staged] == ([] if mode == "direct" else ["local-copy"])


@pytest.mark.parametrize("mode", MODES)
def test_input_corrupted_after_publication_fails_before_the_component_runs(tmp_path, monkeypatch, local, mode):
    spec, weight = large_input_spec(tmp_path, mode)
    directory = publish(tmp_path, spec)
    weight.write_bytes(b"x" * 4096)  # same size: only a hash can tell
    assert run_campaign(directory, monkeypatch) == 1
    assert "identity changed" in monitor.read_json(directory / "campaign-error.json")["error"]
    assert not (directory / "smoke" / "execution.json").exists()
    assert not (directory / "smoke" / "result.json").exists()
    assert not list(local.glob("cache/*/unet.bin"))
