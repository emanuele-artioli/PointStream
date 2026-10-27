"""PID ownership survives lazy proc inode recreation and legacy receipts."""
from experiments.jobs import claims


def test_birth_time_uses_kernel_ticks_even_with_parentheses_in_name(tmp_path, monkeypatch):
    process = tmp_path / 'proc' / '42'
    process.mkdir(parents=True)
    (process / 'stat').write_text('42 (worker (archive)) ' + ' '.join(['S'] + ['0'] * 18 + ['400']))
    (tmp_path / 'proc' / 'stat').write_text('cpu 1 2 3\nbtime 1000\n')
    monkeypatch.setattr(claims, 'Path', lambda path: tmp_path / path.lstrip('/'))
    monkeypatch.setattr(claims.os, 'sysconf', lambda name: 100)
    assert claims.get_process_start_time(42) == 1004


def test_legacy_late_inode_time_protects_live_pid_but_reuse_does_not(monkeypatch):
    monkeypatch.setattr(claims.os, 'kill', lambda *args: None)
    monkeypatch.setattr(claims, 'get_process_start_time', lambda pid: 1000)
    assert claims.is_pid_alive(99999999, 1000)
    assert claims.is_pid_alive(99999999, 1100)
    assert not claims.is_pid_alive(99999999, 900)
