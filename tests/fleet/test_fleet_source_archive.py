"""Inbox jobs keep only source.tar on NFS; workers unpack it host-locally."""
import io
import gzip
import json
import os
import shlex
import subprocess
import sys
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
    uploads = []
    def upload(argv, **kwargs):
        uploads.append((argv, kwargs['stdin'].read(), kwargs['timeout']))
        return SimpleNamespace(returncode=0, stderr=b"")
    monkeypatch.setattr(fleet.subprocess, "run", upload)
    monkeypatch.setattr(fleet, "_ssh", remote)
    fleet._send_snapshot("gpu2", source, "/inbox/job/source", "a" * 64, timeout=90, extract=False)
    argv, sent, timeout = uploads[0]
    assert argv[:2] == ["ssh", "-C"] and argv[-2] == "gpu2"
    receiver = shlex.split(argv[-1])
    assert receiver[:4] == ["bash", "--noprofile", "--norc", "-c"]
    assert "-x /usr/bin/gzip" in receiver[4] and "-x /usr/bin/dd" in receiver[4]
    assert "iflag=fullblock" in argv[-1] and "noclobber" in argv[-1]
    assert "pipefail" in argv[-1] and "/usr/bin/gzip -dc" in argv[-1]
    assert "bs=1M" in argv[-1] and "/inbox/job/source.tar" in argv[-1]
    assert gzip.decompress(sent) == b"archive" and 0 < timeout <= 90
    assert commands == [["sha256sum", "--", "/inbox/job/source.tar"]]


def test_snapshot_transfer_stops_when_shared_deadline_expires(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"archive")
    moments = iter((0.0, 1.0, 2.0, 3.0, 91.0))
    monkeypatch.setattr(fleet.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(fleet.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stderr=b""))
    monkeypatch.setattr(fleet, "_ssh", lambda *a, **kw: pytest.fail("checksum after expired deadline"))
    with pytest.raises(fleet.FleetError, match="time budget exhausted"):
        fleet._send_snapshot("gpu2", source, "/inbox/job/source", "a" * 64, timeout=90, extract=False)
    assert source.read_bytes() == b"archive"


def test_snapshot_pipeline_failure_is_not_accepted(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"archive")
    monkeypatch.setattr(fleet.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr=b"gzip failed"))
    monkeypatch.setattr(fleet, "_ssh", lambda *a, **kw: pytest.fail("checksum after pipeline failure"))
    with pytest.raises(fleet.FleetError, match="gzip failed"):
        fleet._send_snapshot("gpu2", source, "/inbox/job/source", "a" * 64, timeout=90, extract=False)
    assert source.read_bytes() == b"archive"


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


@pytest.mark.parametrize('chosen, expected', [(None, 'gpu1'), ('gpu2', 'gpu2')])
def test_submit_routes_only_snapshot_through_verified_host(tmp_path, monkeypatch, chosen, expected):
    from argparse import Namespace

    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(specification(tmp_path)))
    snapshot = tmp_path / 'snapshot.tar'
    snapshot.write_bytes(b'fixed archive')
    config = {'hosts': ['gpu1', 'gpu2'], 'base': '/data/jobs/fleet'}
    sent = []
    payloads = {}
    monkeypatch.setattr(inbox, 'load_config', lambda _: config)
    monkeypatch.setattr(inbox, 'doctor', lambda _: {'passed': True, 'base': config['base']})
    monkeypatch.setattr(fleet, '_build_snapshot', lambda *a, **kw: (snapshot, 'fixed-sha', {}))
    monkeypatch.setattr(fleet, '_send_snapshot', lambda *a, **kw: sent.append((a, kw)))
    def remote(_config, action, payload):
        payloads[action] = payload
        if action == 'health':
            return {'workers': {h: {'updated': time.time(), 'capabilities': ['source_archive']} for h in config['hosts']}}
        if action == 'prepare':
            return {'directory': '/data/jobs/fleet/inbox/' + payload['job_id'], 'source_archive': True}
        return {'status': 'pending'}
    monkeypatch.setattr(inbox, 'remote_rpc', remote)
    args = Namespace(state_dir=tmp_path / 'state', spec=spec_path, include_change=[], include_untracked=[],
                     chat_id='chat', snapshot_upload_host=chosen, snapshot_transfer_seconds=90)
    result = inbox.submit(args, tmp_path)
    assert sent[0][0][0] == expected
    assert sent[0][0][1:] == (snapshot, '/data/jobs/fleet/inbox/' + result['job_id'] + '/source', 'fixed-sha')
    assert sent[0][1]['extract'] is False and sent[0][1]['timeout'] == 90
    assert payloads['publish']['spec']['hosts'] == ['gpu1']
    assert result.get('snapshot_upload_host') == chosen


def test_upload_host_rejects_unconfigured_or_failed_visibility_before_prepare(tmp_path, monkeypatch):
    from argparse import Namespace

    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(specification(tmp_path)))
    config = {'hosts': ['gpu1', 'gpu2'], 'base': '/data/jobs/fleet'}
    monkeypatch.setattr(inbox, 'load_config', lambda _: config)
    monkeypatch.setattr(inbox, 'remote_rpc', lambda *a: pytest.fail('must not prepare or publish'))
    args = Namespace(state_dir=tmp_path / 'state', spec=spec_path, include_change=[], include_untracked=[],
                     chat_id='chat', snapshot_upload_host='gpu6')
    with pytest.raises(fleet.FleetError, match='verified fleet configuration'):
        inbox.submit(args, tmp_path)
    args.snapshot_upload_host = 'gpu2'
    for report in ({'passed': False, 'base': config['base']}, {'passed': True, 'base': '/other'}):
        monkeypatch.setattr(inbox, 'doctor', lambda _, r=report: r)
        with pytest.raises(fleet.FleetError, match='shared-visibility'):
            inbox.submit(args, tmp_path)


def test_upload_host_requires_fresh_heartbeat_before_prepare(tmp_path, monkeypatch):
    from argparse import Namespace

    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(specification(tmp_path)))
    config = {'hosts': ['gpu1', 'gpu2'], 'base': '/data/jobs/fleet'}
    monkeypatch.setattr(inbox, 'load_config', lambda _: config)
    monkeypatch.setattr(inbox, 'doctor', lambda _: {'passed': True, 'base': config['base']})
    def remote(_config, action, _payload):
        if action != 'health':
            pytest.fail('must not prepare or publish')
        return {'workers': {'gpu1': {'updated': time.time()}, 'gpu2': {'updated': 0}}}
    monkeypatch.setattr(inbox, 'remote_rpc', remote)
    args = Namespace(state_dir=tmp_path / 'state', spec=spec_path, include_change=[], include_untracked=[],
                     chat_id='chat', snapshot_upload_host='gpu2')
    with pytest.raises(fleet.FleetError, match='gpu2 has no fresh heartbeat'):
        inbox.submit(args, tmp_path)


def test_cache_copy_verifies_sha_and_never_overwrites(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    import sys

    base = tmp_path / 'fleet'
    base.mkdir()
    archive = base / 'cached.tar'
    archive.write_bytes(b'original snapshot bytes')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    def local_ssh(_host, argv, *, timeout):
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    monkeypatch.setattr(fleet, '_ssh', local_ssh)
    target = base / 'new' / 'source'
    target.parent.mkdir()
    fleet._copy_snapshot_from_cache('gpu5', sys.executable, str(base), str(archive), str(target), sha, timeout=30)
    saved = target.with_suffix('.tar')
    assert saved.read_bytes() == archive.read_bytes()
    assert saved.stat().st_ino != archive.stat().st_ino
    with pytest.raises(fleet.FleetError, match='cache copy failed'):
        fleet._copy_snapshot_from_cache('gpu5', sys.executable, str(base), str(archive), str(target), sha, timeout=30)
    assert saved.read_bytes() == b'original snapshot bytes'
    archive.write_bytes(b'truncated')
    with pytest.raises(fleet.FleetError, match='cache copy failed'):
        fleet._copy_snapshot_from_cache('gpu5', sys.executable, str(base), str(archive), str(base / 'other'), sha, timeout=30)
    assert not (base / 'other.tar').exists()
    foreign = tmp_path / 'outside.tar'
    foreign.write_bytes(b'original snapshot bytes')
    with pytest.raises(fleet.FleetError, match='within shared fleet'):
        fleet._copy_snapshot_from_cache('gpu5', sys.executable, str(base), str(foreign), str(base / 'other'), sha, timeout=30)
    link = base / 'link.tar'
    link.symlink_to(foreign)
    with pytest.raises(fleet.FleetError, match='cache copy failed'):
        fleet._copy_snapshot_from_cache('gpu5', sys.executable, str(base), str(link), str(base / 'other'), sha, timeout=30)


def test_cache_reuse_requires_explicit_verified_host(tmp_path, monkeypatch):
    from argparse import Namespace

    monkeypatch.setattr(inbox, 'load_config', lambda _: {'hosts': ['gpu1'], 'base': '/data/jobs/fleet'})
    args = Namespace(state_dir=tmp_path, snapshot_transfer_seconds=450,
                     snapshot_cache_archive='/data/jobs/fleet/cache.tar')
    with pytest.raises(fleet.FleetError, match='explicitly verified upload host'):
        inbox.submit(args, tmp_path)


def test_submit_cache_reuse_keeps_job_spec_and_passes_exact_snapshot_sha(tmp_path, monkeypatch):
    from argparse import Namespace

    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(specification(tmp_path)))
    snapshot = tmp_path / 'snapshot.tar'
    snapshot.write_bytes(b'fixed snapshot')
    config = {'hosts': ['gpu1', 'gpu2'], 'base': '/data/jobs/fleet', 'pythons': {'gpu2': '/usr/bin/python3'}}
    copies = []
    monkeypatch.setattr(inbox, 'load_config', lambda _: config)
    monkeypatch.setattr(inbox, 'doctor', lambda _: {'passed': True, 'base': config['base']})
    monkeypatch.setattr(fleet, '_build_snapshot', lambda *a, **kw: (snapshot, 'fixed-sha', {}))
    monkeypatch.setattr(fleet, '_send_snapshot', lambda *a, **kw: pytest.fail('unexpected stream upload'))
    monkeypatch.setattr(fleet, '_copy_snapshot_from_cache', lambda *a, **kw: copies.append((a, kw)))
    def remote(_config, action, payload):
        if action == 'health':
            return {'workers': {h: {'updated': time.time(), 'capabilities': ['source_archive']} for h in config['hosts']}}
        if action == 'prepare':
            return {'directory': '/data/jobs/fleet/inbox/' + payload['job_id'], 'source_archive': True}
        assert payload['spec']['hosts'] == ['gpu1']
        return {'status': 'pending'}
    monkeypatch.setattr(inbox, 'remote_rpc', remote)
    args = Namespace(state_dir=tmp_path / 'state', spec=spec_path, include_change=[], include_untracked=[],
                     chat_id='chat', snapshot_upload_host='gpu2', snapshot_transfer_seconds=450,
                     snapshot_cache_archive='/data/jobs/fleet/checks/cache.tar')
    result = inbox.submit(args, tmp_path)
    assert copies == [(('gpu2', '/usr/bin/python3', config['base'], args.snapshot_cache_archive,
                        '/data/jobs/fleet/inbox/' + result['job_id'] + '/source', 'fixed-sha'), {'timeout': 450})]
    assert result['snapshot_cache_archive'] == args.snapshot_cache_archive


@pytest.mark.parametrize('existing', [False, True])
def test_snapshot_receiver_quotes_target_and_preserves_existing_bytes(tmp_path, monkeypatch, existing):
    """Run the actual receiver shell with local transport-tool stand-ins."""
    source = tmp_path / 'source.tar'
    source.write_bytes(b'new archive bytes')
    # If shell quoting breaks, command substitution would create this marker.
    marker = tmp_path / 'injected'
    target = tmp_path / "source ' $(touch injected)"
    archive = target.with_suffix('.tar')
    if existing:
        archive.write_bytes(b'preserved evidence')
    tools = []
    for name, body in [('gzip', 'import gzip; sys.stdout.buffer.write(gzip.decompress(sys.stdin.buffer.read()))'),
                       ('dd', 'import shutil; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)')]:
        tool = tmp_path / name
        tool.write_text('#!' + sys.executable + '\nimport sys\n' + body + '\n')
        tool.chmod(0o755)
        tools.append(tool)
    real_run = subprocess.run
    def upload(argv, **kwargs):
        receiver = shlex.split(argv[-1])
        assert receiver[:4] == ['bash', '--noprofile', '--norc', '-c']
        script = receiver[4].replace('/usr/bin/gzip', str(tools[0])).replace('/usr/bin/dd', str(tools[1]))
        return real_run([*receiver[:4], script], cwd=tmp_path, **kwargs)
    monkeypatch.setattr(fleet.subprocess, 'run', upload)
    checks = []
    monkeypatch.setattr(fleet, '_ssh', lambda host, argv, **kw:
                        checks.append(argv) or SimpleNamespace(returncode=0, stdout='a' * 64 + ' archive'))
    if existing:
        with pytest.raises(fleet.FleetError, match='snapshot transfer failed'):
            fleet._send_snapshot('unused', source, str(target), 'a' * 64, extract=False)
        assert archive.read_bytes() == b'preserved evidence'
        assert not checks
    else:
        fleet._send_snapshot('unused', source, str(target), 'a' * 64, extract=False)
        assert archive.read_bytes() == source.read_bytes()
        assert checks == [['sha256sum', '--', str(archive)]]
    assert not marker.exists()


def test_snapshot_receiver_missing_absolute_tool_stops_before_writing(tmp_path, monkeypatch):
    source = tmp_path / 'source.tar'
    source.write_bytes(b'archive')
    target = tmp_path / 'target'
    real_run = subprocess.run
    def upload(argv, **kwargs):
        receiver = shlex.split(argv[-1])
        script = receiver[4].replace('/usr/bin/dd', str(tmp_path / 'missing-dd'))
        return real_run([*receiver[:4], script], cwd=tmp_path, **kwargs)
    monkeypatch.setattr(fleet.subprocess, 'run', upload)
    monkeypatch.setattr(fleet, '_ssh', lambda *a, **kw: pytest.fail('checksum after failed tool preflight'))
    with pytest.raises(fleet.FleetError, match='receiver tools unavailable'):
        fleet._send_snapshot('unused', source, str(target), 'a' * 64, extract=False)
    assert not target.with_suffix('.tar').exists()
