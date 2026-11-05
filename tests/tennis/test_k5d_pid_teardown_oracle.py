import json
import os
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import k5d_pid_teardown_oracle as oracle


def test_process_identity_requires_actual_namespace_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fake_read(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self) == '/proc/self/status':
            return 'NSpid:\t123\n'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', fake_read)
    with pytest.raises(ValueError, match='namespace PIDs'):
        oracle.process_identity()


def test_exclusive_receipt_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / 'result.json'
    oracle.write(path, {'passed': True})
    with pytest.raises(FileExistsError):
        oracle.write(path, {'passed': False})
    assert json.loads(path.read_text())['passed'] is True


def test_missing_proof_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    with pytest.raises(FileNotFoundError):
        oracle.validate()
    assert json.loads((tmp_path / 'validation.json').read_text())['passed'] is False


def test_gpu_or_wrong_platform_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle.sys, 'platform', 'darwin')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='CPU-only'):
        oracle.execute('unused')


def test_pid_absence_only_for_dead_or_reused_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A PID still assigned to the task, including a zombie, is not reported reaped.
    root = tmp_path / 'proc'
    root.mkdir()
    fakepid = root / str(os.getpid())
    fakepid.mkdir()
    (fakepid / 'stat').write_text('1 (synthetic) Z ' + '0 ' * 18 + '42 0')
    real_path = oracle.Path

    def path(value: str) -> Path:
        return root if value == '/proc' else real_path(value)

    monkeypatch.setattr(oracle, 'Path', path)
    assert not oracle.gone({'host_pid': os.getpid(), 'start_ticks': 42})
    assert oracle.gone({'host_pid': os.getpid(), 'start_ticks': 43})


def test_validator_rejects_natural_expiry_disguised_as_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import io
    import tarfile

    stage = tmp_path / 'smoke'
    stage.mkdir()
    record: dict[str, Any] = {'schema': 1, 'scope': oracle.SCOPE, 'stage': 'smoke',
              'source_sha256': oracle.sha(Path(oracle.__file__)), 'dispatch_source_sha256': 'source',
              'dispatch_snapshot_sha256': 'snapshot', 'environment_sha256': 'environment',
              'unshare_path': oracle.__file__, 'unshare_sha256': oracle.sha(Path(oracle.__file__)),
              'unshare_version': 'unshare from util-linux 2.37.2', 'cases': []}
    for case in oracle.CASES:
        record['cases'].append({'case': case, 'launcher_host_pid': 100,
                                'launcher_exit_code': {'normal': 0, 'timeout': -9, 'sigterm': -15, 'sigkill': -9}[case],
                                'outside_pid_namespace': 'outside',
                                'init': {'host_pid': 101, 'namespace_pid': 1, 'namespace_pids': [101, 1], 'start_ticks': 1, 'pid_namespace': 'pid:[2]', 'namespace_sid': 0},
                                'intermediate': {'host_pid': 102, 'namespace_pid': 2, 'namespace_pids': [102, 2], 'start_ticks': 1, 'pid_namespace': 'pid:[2]'},
                                'escaped': {'host_pid': 103, 'namespace_pid': 3, 'namespace_pids': [103, 3], 'start_ticks': 1, 'pid_namespace': 'pid:[2]',
                                            'namespace_sid': 2, 'namespace_pgid': 2},
                                'members_absent': True, 'signals_sent_only_to_launcher': True,
                                'elapsed_since_escape_seconds': 2.9})
    archive = stage / 'published.tar'
    with tarfile.open(archive, 'w') as tar:
        root = tarfile.TarInfo('publish')
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        content = json.dumps(record).encode()
        entry = tarfile.TarInfo('publish/oracle.json')
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}}))
    (stage / 'dispatch.json').write_text(json.dumps({'code': {'source_sha256': 'source', 'snapshot_sha256': 'snapshot'},
                                                        'staged_environment': {'sha256': 'environment'},
                                                        'environment': {'gpu': None}}))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    with pytest.raises(ValueError, match='cleanup evidence'):
        oracle.validate()
    assert json.loads((tmp_path / 'validation.json').read_text())['passed'] is False
