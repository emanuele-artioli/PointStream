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
    oracle._cleanup_group(cast(Any, Child()))
    assert events == [('killpg', 23, signal.SIGKILL), ('wait', 10)]


def test_gone_uses_birth_to_detect_pid_reuse(monkeypatch: Any) -> None:
    monkeypatch.setattr(oracle.isolated, '_proc_identity', lambda _: {'birth': 'new'})
    assert oracle._gone({'pid': 17, 'birth': 'old'})
    assert not oracle._gone({'pid': 17, 'birth': 'new'})
