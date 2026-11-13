"""Local guards only; these tests do not establish a Linux lifecycle proof."""
from typing import Any, cast
import signal

import pytest

from experiments.tennis import k5d_pgid_oracle as oracle, k5d_scoped_namespace as namespace


def test_surrogate_group_refuses_shared_or_exited_group(monkeypatch: Any) -> None:
    class Child:
        pid = 23
        def poll(self) -> None:
            return None
    child = cast(Any, Child())
    monkeypatch.setattr(oracle.os, 'getpgid', lambda _: 23)
    monkeypatch.setattr(oracle.os, 'getpgrp', lambda: 23)
    with pytest.raises(ValueError, match='surrogate group'):
        oracle._group(child)
    monkeypatch.setattr(oracle.os, 'getpgrp', lambda: 12)
    assert oracle._group(child) == 23
    monkeypatch.setattr(oracle.os, 'getpgid', lambda _: 24)
    with pytest.raises(ValueError, match='surrogate group'):
        oracle._group(child)


def test_lifecycle_worker_is_cpu_only(monkeypatch: Any) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert namespace.worker_command(['ada-pgid-cpu', '--wire', 'packet'])[2:] == [
        'experiments.tennis.k5d_pgid_oracle', 'run', '--wire', 'packet']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='fixed'):
        namespace.worker_command(['ada-pgid-cpu'])


def test_surrogate_cleanup_kills_only_owned_group_before_wait(monkeypatch: Any) -> None:
    events: list[tuple[Any, ...]] = []
    class Child:
        pid = 23
        def poll(self) -> None:
            return None
        def wait(self, timeout: int) -> None:
            events.append(('wait', timeout))
    monkeypatch.setattr(oracle.os, 'getpgid', lambda _: 23)
    monkeypatch.setattr(oracle.os, 'getpgrp', lambda: 12)
    monkeypatch.setattr(oracle.os, 'killpg', lambda pid, sig: events.append(('killpg', pid, sig)))
    monkeypatch.setattr(oracle, '_finish_group', lambda *args: events.append(('finish', *args)))
    oracle._cleanup_group(cast(Any, Child()), 100, {100: '7'}, 2)
    assert events == [('killpg', 23, signal.SIGKILL), ('wait', 10), ('finish', 100, {100: '7'}, 2)]


def test_gone_uses_birth_to_detect_pid_reuse(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle.Path, 'read_text', lambda _: '17 (x) ' + ' '.join(['0'] * 19 + ['new']))
    assert oracle._gone({'pid': 17, 'birth': 'old'})
    assert not oracle._gone({'pid': 17, 'birth': 'new'})


def test_unknown_proc_access_is_not_absence(monkeypatch: Any) -> None:
    def denied(_: Any) -> None:
        raise PermissionError('proc denied')
    monkeypatch.setattr(oracle.Path, 'read_text', denied)
    with pytest.raises(PermissionError):
        oracle._gone({'pid': 17, 'birth': 'old'})


def test_zombie_stat_is_not_absent(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle.Path, 'read_text', lambda _: '17 (x) Z ' + ' '.join(['0'] * 18 + ['birth']))
    assert not oracle._gone({'pid': 17, 'birth': 'birth'})


def test_unknown_residual_is_never_signalled(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle, '_members', lambda _: {91: 'new'})
    monkeypatch.setattr(oracle, 'Pidfd', lambda: pytest.fail('must not signal unknown process'))
    with pytest.raises(RuntimeError, match='unknown residual'):
        oracle._finish_group(90, {91: 'old'}, 2)


def test_exited_leader_still_checks_group(monkeypatch: Any) -> None:
    events: list[str] = []
    class Child:
        def poll(self) -> int:
            return 1
        def wait(self, timeout: int) -> None:
            events.append('wait')
    monkeypatch.setattr(oracle.os, 'killpg', lambda *_: pytest.fail('must not signal stale group'))
    monkeypatch.setattr(oracle, '_finish_group', lambda *_: events.append('finish'))
    oracle._cleanup_group(cast(Any, Child()), 100, {}, 2)
    assert events == ['wait', 'finish']


def test_emergency_cleanup_cannot_pass_as_natural_exit(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle, '_finish_group', lambda *_: True)
    with pytest.raises(ValueError, match='emergency cleanup'):
        oracle._require_natural_group_exit(100, {}, 2)


def test_all_captured_births_must_disappear_even_after_group_change(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle, '_gone', lambda item: item['pid'] != 18)
    assert not oracle._captured_gone({'17': 'first', '18': 'second'})
    assert oracle._captured_gone({'17': 'first'})


@pytest.mark.parametrize('case', ['residual_cleanup', 'launcher_kill'])
def test_oracle_pidfd_kill_reaches_backend_with_strict_integer_signal(
        monkeypatch: Any, tmp_path: Any, case: str) -> None:
    """Exercise both callers through the real strict pidfd send validation."""
    from types import SimpleNamespace

    calls: list[tuple[int, int]] = []
    adapter = object.__new__(oracle.Pidfd)
    adapter.python_backend = True
    monkeypatch.setattr(adapter, 'open', lambda pid: 99)
    monkeypatch.setattr(oracle, 'Pidfd', lambda: adapter)
    monkeypatch.setattr(oracle.os, 'close', lambda fd: None)

    class BackendReached(Exception):
        pass

    def send(fd: int, signum: int) -> None:
        calls.append((fd, signum))
        if case == 'launcher_kill':
            raise BackendReached

    monkeypatch.setattr(signal, 'pidfd_send_signal', send, raising=False)
    launcher = {'pid': 21, 'birth': 'launcher', 'host_pgid': 20, 'namespace_pids': [21, 2]}
    worker = {'pid': 22, 'birth': 'worker', 'host_pgid': 20, 'host_ppid': 21}
    leader = {'pid': 20, 'birth': 'leader', 'host_pgid': 20}
    identities = {20: leader, 21: launcher, 22: worker,
                  50: {'host_ppid': 51, 'namespace_pids': [50, 1]}}
    monkeypatch.setattr(oracle.isolated, '_proc_identity', lambda pid: identities[pid])

    if case == 'residual_cleanup':
        inventories = iter([{21: 'launcher'}, {}])
        monkeypatch.setattr(oracle, '_members', lambda group: next(inventories))
        times = iter([0, 0, 3, 3])
        monkeypatch.setattr(oracle.time, 'monotonic', lambda: next(times))
        monkeypatch.setattr(oracle.time, 'sleep', lambda _: None)
        assert oracle._finish_group(20, {21: 'launcher'}, 2)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        monkeypatch.setattr(oracle.os, 'sched_getaffinity', lambda _: {0}, raising=False)
        monkeypatch.setattr(oracle.os, 'readlink', lambda _: '50')
        monkeypatch.setattr(oracle.os, 'getppid', lambda: 1)
        monkeypatch.setattr(oracle.Path, 'read_bytes', lambda _: b'experiments.tennis.k5d_scoped_namespace\0inner')
        monkeypatch.setattr(oracle.witness, 'parser', lambda: SimpleNamespace(
            parse_args=lambda _: SimpleNamespace(unshare_sha256='pinned')))
        unshare = tmp_path / 'unshare'
        unshare.touch()
        monkeypatch.setattr(oracle.shutil, 'which', lambda _: str(unshare))
        monkeypatch.setattr(oracle.oracle, 'sha', lambda _: 'pinned')
        stage = tmp_path / 'job' / 'smoke'
        stage.mkdir(parents=True)
        (stage / 'dispatch.json').write_text('{}')
        scratch = tmp_path / 'scratch'
        scratch.mkdir()
        monkeypatch.setenv('PS_STAGE_DIR', str(stage))
        monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
        monkeypatch.setattr(oracle.cp, 'execution_identity', lambda: {})
        child = SimpleNamespace(pid=20, stdin=None, wait=lambda **kwargs: None)
        monkeypatch.setattr(oracle.subprocess, 'Popen', lambda *args, **kwargs: child)
        translations = iter([{'pid': 10, 'host_pgid': 10}, leader])
        monkeypatch.setattr(oracle.isolated, '_translate_child', lambda *args: next(translations))
        monkeypatch.setattr(oracle, '_group', lambda _: 20)
        monkeypatch.setattr(oracle, '_wait_marker', lambda *args: dict(worker, launcher_observer=launcher))
        monkeypatch.setattr(oracle, '_members', lambda _: {20: 'leader', 21: 'launcher', 22: 'worker'})
        monkeypatch.setattr(oracle, '_cleanup_group', lambda *args: None)
        monkeypatch.setattr(oracle.os, 'killpg', lambda *args: None)
        monkeypatch.setattr(oracle, 'CASES', ('launcher_kill',))
        with pytest.raises(BackendReached):
            oracle.run([])

    assert calls == [(99, int(signal.SIGKILL))]
    assert type(calls[0][1]) is int
