import json

import pytest

from experiments.tennis import k5d_native_pid_oracle as oracle


def test_validator_fails_closed_without_publication(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / 'smoke'
    stage.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    target = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_VALIDATION_PATH', str(target))
    with pytest.raises(FileNotFoundError):
        oracle.validate()
    assert json.loads(target.read_text())['passed'] is False


def test_cpu_proof_rejects_visible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_native_cpu_workload as workload
    from pathlib import Path

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'):
        workload.run(Path('/unused'), 'a' * 64, Path('/unused'), {})


def test_libc_pidfd_exports_keep_pidfd_signal_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_pidfd

    calls: list[tuple] = []
    def opened(pid, flags):
        calls.append(('open', pid, flags))
        return 7
    def sent(fd, sig, info, flags):
        calls.append(('send', fd, sig, info, flags))
        return 0
    class Function:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *args):
            return self.fn(*args)
    class Libc:
        gnu_get_libc_version = Function(lambda: b'2.39')
        pidfd_open = Function(opened)
        pidfd_send_signal = Function(sent)
    monkeypatch.setattr(k5d_pidfd.sys, 'platform', 'linux')
    monkeypatch.setattr(k5d_pidfd.ctypes, 'CDLL', lambda *_args, **_kw: Libc())
    monkeypatch.setattr(k5d_pidfd.os, 'pidfd_open', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.signal, 'pidfd_send_signal', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.os, 'close', lambda fd: calls.append(('close', fd)))
    adapter = k5d_pidfd.Pidfd()
    proof = adapter.self_check()
    assert proof['backend'] == 'libc-exports' and proof['own_pid_signal_zero'] is True
    assert calls == [('open', k5d_pidfd.os.getpid(), 0), ('send', 7, 0, None, 0), ('close', 7)]
    adapter.send(7, 19)
    assert calls[-1] == ('send', 7, 19, None, 0)


def test_missing_libc_pidfd_exports_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_pidfd

    class Function:
        def __call__(self):
            return b'2.35'
    class Libc:
        gnu_get_libc_version = Function()
    monkeypatch.setattr(k5d_pidfd.sys, 'platform', 'linux')
    monkeypatch.setattr(k5d_pidfd.ctypes, 'CDLL', lambda *_args, **_kw: Libc())
    monkeypatch.setattr(k5d_pidfd.os, 'pidfd_open', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.signal, 'pidfd_send_signal', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.platform, 'machine', lambda: 'not-x86_64')
    with pytest.raises(RuntimeError, match='unsupported Linux pidfd syscall ABI'):
        k5d_pidfd.Pidfd()


def test_x86_64_pidfd_syscall_backend_uses_typed_arguments_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    from experiments.tennis import k5d_pidfd

    calls: list[tuple] = []
    class Function:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *args):
            return self.fn(*args)
    def call(*args):
        calls.append(args)
        return 7 if args[0].value == 434 else 0
    class Libc:
        gnu_get_libc_version = Function(lambda: b'2.35')
        syscall = Function(call)
    monkeypatch.setattr(k5d_pidfd.sys, 'platform', 'linux')
    monkeypatch.setattr(k5d_pidfd.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(k5d_pidfd.ctypes, 'CDLL', lambda *_args, **_kw: Libc())
    monkeypatch.setattr(k5d_pidfd.os, 'pidfd_open', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.signal, 'pidfd_send_signal', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.os, 'close', lambda _fd: None)
    adapter = k5d_pidfd.Pidfd()
    proof = adapter.self_check()
    assert proof['backend'] == 'linux-x86_64-syscall'
    assert [x.value for x in calls[0]] == [434, k5d_pidfd.os.getpid(), 0]
    assert [x.value for x in calls[1]] == [424, 7, 0, None, 0]
    assert isinstance(calls[0][1], ctypes.c_int) and isinstance(calls[1][3], ctypes.c_void_p)
    setattr(adapter, 'syscall_fn', Function(lambda *_args: -1))
    with pytest.raises(OSError, match='pidfd_open failed'):
        adapter.self_check()


def test_x86_64_pidfd_syscall_rejects_wrong_word_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_pidfd

    class Function:
        def __call__(self):
            return b'2.35'
    class Libc:
        gnu_get_libc_version = Function()
    monkeypatch.setattr(k5d_pidfd.sys, 'platform', 'linux')
    monkeypatch.setattr(k5d_pidfd.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(k5d_pidfd.ctypes, 'CDLL', lambda *_args, **_kw: Libc())
    monkeypatch.setattr(k5d_pidfd.ctypes, 'sizeof', lambda _: 4)
    monkeypatch.setattr(k5d_pidfd.os, 'pidfd_open', None, raising=False)
    monkeypatch.setattr(k5d_pidfd.signal, 'pidfd_send_signal', None, raising=False)
    with pytest.raises(RuntimeError, match='unsupported Linux pidfd syscall ABI'):
        k5d_pidfd.Pidfd()
