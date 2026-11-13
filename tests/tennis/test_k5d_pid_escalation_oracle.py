import json
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import k5d_pid_escalation_oracle as oracle
from experiments.tennis import k5d_pid_namespace_wrapper as wrapper


def test_wrapper_rejects_root_outside_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    with pytest.raises(ValueError, match='inside fleet scratch'):
        wrapper.checked_root(str(tmp_path))


def test_outer_execs_unshare_with_fixed_inner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / 'scratch'
    case = scratch / 'case'
    case.mkdir(parents=True)
    fake = tmp_path / 'unshare'
    fake.write_bytes(b'synthetic executable')
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(wrapper.sys, 'platform', 'linux')
    monkeypatch.setattr(wrapper.shutil, 'which', lambda _: str(fake))
    monkeypatch.setattr(wrapper.os, 'readlink', lambda _: 'pid:[1]')
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(wrapper.os, 'execv', lambda path, args: calls.append((path, args)))
    wrapper.outer(str(case), wrapper.sha(fake))
    assert calls[0][0] == str(fake)
    assert calls[0][1][1:6] == ['--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL']
    assert 'experiments.tennis.k5d_pid_namespace_wrapper' in calls[0][1]
    with pytest.raises(ValueError, match='identity'):
        wrapper.outer(str(case), '0' * 64)


def test_inner_rejects_host_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / 'scratch'
    case = scratch / 'case'
    case.mkdir(parents=True)
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setattr(wrapper.os, 'getpid', lambda: 99)
    original = Path.read_text

    def read(self: Path, *args: Any, **kwargs: Any) -> str:
        return 'NSpid:\t99\t99\n' if str(self) == '/proc/self/status' else original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(ValueError, match='PID 1'):
        wrapper.inner(str(case), 'pid:[1]')


def test_prior_cases_and_failure_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the persistence policy with CPU-only synthetic local records.
    stage = tmp_path / 'stage'
    stage.mkdir()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (stage / 'dispatch.json').write_text(json.dumps({'code': {'source_sha256': 's', 'snapshot_sha256': 't'},
         'staged_environment': {'sha256': 'e'}, 'environment': {'gpu': None}}))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(oracle.sys, 'platform', 'linux')
    monkeypatch.setattr(oracle.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(oracle, 'sha', lambda _: 'same')
    monkeypatch.setattr(oracle.subprocess, 'run', lambda *a, **kw: type('R', (), {'stdout':
        'unshare from util-linux 2.37 --user --map-root-user --pid --fork --kill-child'})())

    def run_case(root: Path, case: str, unshare: str) -> dict[str, Any]:
        (root / case).mkdir()
        if case == 'timeout':
            (root / case / 'launcher.json').write_text('{"host_pid":123}')
            raise ValueError('synthetic failure')
        return {'case': case}

    monkeypatch.setattr(oracle, 'one_case', run_case)
    with pytest.raises(ValueError, match='synthetic failure'):
        oracle.execute('same', 'same')
    assert json.loads((scratch / 'publish/case-normal.json').read_text()) == {'case': 'normal'}
    result = json.loads((scratch / 'publish/case-timeout-failure.json').read_text())
    assert result['partial']['launcher.json']['host_pid'] == 123


def test_namespace_identity_rejects_wrong_host_pid() -> None:
    valid = {'host_pid': 100, 'namespace_pid': 1, 'namespace_pids': [100, 1],
             'start_ticks': 55, 'pid_namespace': 'pid:[123]'}
    assert oracle.valid_identity(valid)
    valid['host_pid'] = 101
    assert not oracle.valid_identity(valid)


def test_missing_partial_validation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / 'stage'
    stage.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(tmp_path / 'validation.json'))
    with pytest.raises(FileNotFoundError):
        oracle.validate()
    assert json.loads((tmp_path / 'validation.json').read_text())['passed'] is False
