"""Host-local staging must never change which bytes a stage reads."""
import io
import json
from pathlib import Path
import tarfile

import pytest

from experiments.jobs import fleet, inbox, monitor, staging
from tests.experiments import test_fleet_inbox
from tests.experiments.test_fleet_inbox import specification

campaign = test_fleet_inbox.campaign


def item(path: Path, name="clips", **extra):
    return {"name": name, "path": str(path), "sha256": inbox.file_digest(path), **extra}


@pytest.fixture
def local(tmp_path, monkeypatch):
    root = tmp_path / "local" / "pointstream"
    root.parent.mkdir()
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(root))
    # Hosts' real free space varies; only the reserve tests constrain it.
    monkeypatch.setattr(staging, "free_bytes", lambda path: 2**50)
    return root


def archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return path


def test_first_use_copies_then_hits_the_verified_cache(tmp_path, local):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"frames" * 1000)
    first = staging.stage_input(item(source), staging.local_root())
    second = staging.stage_input(item(source), staging.local_root())
    assert (first["mode"], second["mode"]) == ("local-copy", "local-hit")
    assert Path(first["path"]).read_bytes() == source.read_bytes()
    assert Path(first["path"]).is_relative_to(local)


def test_corrupted_cache_entry_is_rebuilt_from_shared_source(tmp_path, local):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"canonical")
    cached = Path(staging.stage_input(item(source), staging.local_root())["path"])
    cached.chmod(0o644)
    cached.write_bytes(b"corrupted")
    again = staging.stage_input(item(source), staging.local_root())
    assert again["mode"] == "local-copy"
    assert Path(again["path"]).read_bytes() == b"canonical"


def test_changed_shared_bytes_never_reach_the_cache(tmp_path, local):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"declared")
    declared = item(source)
    source.write_bytes(b"changed")
    with pytest.raises(staging.StagingError):
        staging.stage_input(declared, staging.local_root())
    assert not list((local / "cache").rglob("clip.bin"))


def test_hosts_without_local_storage_use_verified_shared_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(tmp_path / "missing" / "pointstream"))
    assert staging.local_root() is None
    source = tmp_path / "clip.bin"
    source.write_bytes(b"x")
    record = staging.stage_input(item(source), None)
    assert (record["mode"], record["path"]) == ("shared", str(source))
    with pytest.raises(staging.StagingError):
        staging.stage_input(item(source, extract=True), None)


def test_reserve_protects_the_shared_local_disk(tmp_path, local, monkeypatch):
    source = tmp_path / "clip.bin"
    source.write_bytes(b"x")
    monkeypatch.setattr(staging, "free_bytes", lambda path: staging.RESERVE_BYTES)
    assert staging.stage_input(item(source), staging.local_root())["mode"] == "shared"


def test_extracted_tree_is_read_only_and_reused(tmp_path, local):
    bundle = archive(tmp_path / "frames.tar", {"clip/frame_000000.png": b"a", "clip/frame_000001.png": b"b"})
    first = staging.stage_input(item(bundle, extract=True), staging.local_root())
    tree = Path(first["path"])
    assert (tree / "clip" / "frame_000001.png").read_bytes() == b"b"
    with pytest.raises(PermissionError):
        (tree / "clip" / "frame_000000.png").write_bytes(b"edited")
    assert staging.stage_input(item(bundle, extract=True), staging.local_root())["path"] == str(tree)


def test_archive_members_cannot_escape_the_cache(tmp_path, local):
    bundle = archive(tmp_path / "evil.tar", {"../outside.txt": b"x"})
    with pytest.raises(tarfile.TarError):
        staging.stage_input(item(bundle, extract=True), staging.local_root())
    assert not (local / "cache" / "outside.txt").exists()
    assert not list((local / "cache").rglob(".partial-tree-*"))


def test_storage_requirement_controls_admission(tmp_path, local, monkeypatch):
    assert staging.admission_error({}) is None
    assert staging.admission_error({"local_storage_gib": 1}) is None
    monkeypatch.setattr(staging, "free_bytes", lambda path: staging.RESERVE_BYTES)
    assert "less than" in staging.admission_error({"local_storage_gib": 1})
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(tmp_path / "missing" / "pointstream"))
    assert "no writable" in staging.admission_error({"local_storage_gib": 1})


@pytest.mark.parametrize("mutation", [
    lambda s, p: s.update(staged_inputs=[item(p, extract=True)]),
    lambda s, p: s.update(staged_inputs=[item(p), item(p)]),
    lambda s, p: s.update(staged_inputs=[{**item(p), "path": "relative.bin"}]),
    lambda s, p: s.update(staged_inputs=[{**item(p), "mode": "local"}]),
    lambda s, p: s.update(arguments=[*s["arguments"], "{staged:unknown}"]),
    lambda s, p: s.update(local_storage_gib=-1),
])
def test_bad_staging_declarations_are_rejected(tmp_path, mutation):
    payload = tmp_path / "clip.bin"
    payload.write_bytes(b"x")
    spec = specification(tmp_path)
    mutation(spec, payload)
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec)


def stage_job(directory: Path, data: Path, work: str) -> None:
    payload = data / "clip.bin"
    payload.write_bytes(b"payload")
    (directory / "source" / "work.py").write_text(work)
    spec = specification(data)
    spec["staged_inputs"] = [item(payload)]
    spec["arguments"] = ["--clip", "{staged:clips}", "--frames", "{frames}"]
    spec = inbox.validate_spec(spec)
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec), "source_sha256": inbox.source_identity(directory / "source")})


def test_campaign_stages_inputs_and_publishes_scratch_once(campaign, local):
    stage_job(campaign, campaign.parents[3], '''import json,os,pathlib,sys
clip=pathlib.Path(sys.argv[2]).read_bytes()
scratch=pathlib.Path(os.environ['PS_SCRATCH_DIR'])
(scratch/'publish').mkdir()
(scratch/'publish'/'frames.txt').write_bytes(clip)
(scratch/'intermediate.bin').write_bytes(b'discard')
pathlib.Path(os.environ['PS_STAGE_DIR']).joinpath('result.json').write_text(json.dumps({'frames':int(sys.argv[-1])}))
''')
    assert inbox.campaign(campaign) == 0
    smoke = monitor.read_json(campaign / "smoke" / "staging.json")
    full = monitor.read_json(campaign / "full" / "staging.json")
    assert [s["inputs"][0]["mode"] for s in (smoke, full)] == ["local-copy", "local-hit"]
    assert not Path(full["scratch"]).exists()
    with tarfile.open(full["published"]["path"]) as bundle:
        assert bundle.extractfile("publish/frames.txt").read() == b"payload"
    assert full["published"]["sha256"] == inbox.file_digest(Path(full["published"]["path"]))
    dispatch = monitor.read_json(campaign / "full" / "dispatch.json")
    assert dispatch["command"][3] == full["inputs"][0]["path"]


def test_failed_stage_preserves_local_scratch(campaign, local):
    stage_job(campaign, campaign.parents[3], '''import os,pathlib,sys
pathlib.Path(os.environ['PS_SCRATCH_DIR']).joinpath('partial.bin').write_bytes(b'evidence')
sys.exit(3)
''')
    assert inbox.campaign(campaign) == 1
    record = monitor.read_json(campaign / "smoke" / "staging.json")
    assert (Path(record["scratch"]) / "partial.bin").read_bytes() == b"evidence"
    assert not (campaign / "full").exists()


def test_campaign_without_local_storage_keeps_scratch_with_outputs(campaign, monkeypatch, tmp_path):
    monkeypatch.setenv(staging.LOCAL_ROOT_ENV, str(tmp_path / "missing" / "pointstream"))
    stage_job(campaign, campaign.parents[3], '''import json,os,pathlib,sys
pathlib.Path(os.environ['PS_SCRATCH_DIR']).joinpath('kept.bin').write_bytes(b'x')
pathlib.Path(os.environ['PS_STAGE_DIR']).joinpath('result.json').write_text(json.dumps({'frames':int(sys.argv[-1])}))
''')
    assert inbox.campaign(campaign) == 0
    record = monitor.read_json(campaign / "smoke" / "staging.json")
    assert record["local_root"] is None and record["inputs"][0]["mode"] == "shared"
    assert (campaign / "smoke" / "scratch" / "kept.bin").exists()
    assert json.loads((campaign / "full" / "result.json").read_text())["frames"] == 8


def packed_environment(data: Path, prefix_line: str) -> dict:
    """A tar'd prefix whose bin/python reports its prefix and forwards real work."""
    import sys
    prefix = data / "env-src"
    (prefix / "bin").mkdir(parents=True)
    script = prefix / "bin" / "python"
    script.write_text(f'''#!/bin/sh
here="$(cd "$(dirname "$0")/.." && pwd -P)"
if [ "$1" = "-c" ] && [ "$2" = "import sys; print(sys.prefix)" ]; then echo {prefix_line}; exit 0; fi
echo "$0" >> "$PS_STAGE_DIR/interpreters.log"
exec {sys.executable} "$@"
''')
    script.chmod(0o755)
    bundle = data / "env.tar.gz"
    with tarfile.open(bundle, "w:gz") as handle:
        handle.add(prefix, arcname=".")
    return {"path": str(bundle), "sha256": inbox.file_digest(bundle)}


def environment_job(directory: Path, data: Path, prefix_line: str) -> None:
    spec = specification(data)
    spec.update(environment=packed_environment(data, prefix_line), local_storage_gib=1)
    spec = inbox.validate_spec(spec)
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec), "source_sha256": inbox.source_identity(directory / "source")})


def test_stages_and_validator_run_from_the_staged_environment(campaign, local):
    environment_job(campaign, campaign.parents[3], '"$here"')
    assert inbox.campaign(campaign) == 0
    for stage in ("smoke", "full"):
        record = monitor.read_json(campaign / stage / "staging.json")["environment"]
        assert Path(record["prefix"]).is_relative_to(local)
        assert monitor.read_json(campaign / stage / "dispatch.json")["command"][0] == record["python"]
    smoke_runs = (campaign / "smoke" / "interpreters.log").read_text().split()
    assert smoke_runs == [record["python"]] * 2  # workload, then validator
    assert json.loads((campaign / "full" / "result.json").read_text())["frames"] == 8


def test_environment_that_runs_from_elsewhere_is_refused(campaign, local):
    environment_job(campaign, campaign.parents[3], "/home/itec/emanuele/.conda/envs/pointstream")
    assert inbox.campaign(campaign) == 1
    assert "local prefix" in monitor.read_json(campaign / "campaign-error.json")["error"]
    assert not (campaign / "smoke" / "result.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(environment={"path": s["inputs"][0]["path"], "sha256": s["inputs"][0]["sha256"]}),
    lambda s: s.update(local_storage_gib=1, environment={**s["inputs"][0], "extract": True}),
])
def test_bad_environment_declarations_are_rejected(tmp_path, mutation):
    spec = specification(tmp_path)
    mutation(spec)
    with pytest.raises(fleet.FleetError):
        inbox.validate_spec(spec)
