import json

import pytest

from experiments.tennis import ball_confirmation as physical
from experiments.tennis import ball_confirmation_entry as entry


def configure(monkeypatch, *, allowed=None, threads=1, allowance='4'):
    original = set(allowed if allowed is not None else [2, 4, 6, 8, 10])
    state = {'cpus': original, 'sets': []}
    monkeypatch.setenv('PS_CPU_ALLOWANCE', allowance)
    monkeypatch.setattr(entry, 'task_count', lambda: threads)
    monkeypatch.setattr(entry, 'process_start_ticks', lambda: 12345)
    monkeypatch.setattr(entry.os, 'sched_getaffinity', lambda _: state['cpus'], raising=False)
    def set_cpus(_pid, cpus):
        state['sets'].append(list(cpus))
        state['cpus'] = set(cpus)
    monkeypatch.setattr(entry.os, 'sched_setaffinity', set_cpus, raising=False)
    return state


def test_bootstrap_binds_before_import_and_writes_audit(tmp_path, monkeypatch):
    state = configure(monkeypatch)
    receipt = entry.bind('command', tmp_path)
    assert state['sets'] == [[2, 4, 6, 8]]
    assert json.loads((tmp_path / 'command-cpu-binding.json').read_text()) == receipt
    assert physical.valid_cpu_binding(receipt, [2, 4, 6, 8], role='command')
    with pytest.raises(FileExistsError):
        entry.bind('command', tmp_path)


@pytest.mark.parametrize('allowance,threads,allowed', [
    ('3', 1, [0, 1, 2, 3]), ('4', 2, [0, 1, 2, 3]), ('4', 1, [0, 1, 2]),
])
def test_bootstrap_rejects_unsafe_prior_state(tmp_path, monkeypatch, allowance, threads, allowed):
    state = configure(monkeypatch, allowance=allowance, threads=threads, allowed=allowed)
    with pytest.raises(ValueError):
        entry.bind('command', tmp_path)
    assert not state['sets']
    assert not list(tmp_path.iterdir())


def test_bootstrap_fails_if_affinity_does_not_take(tmp_path, monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(entry.os, 'sched_setaffinity', lambda *_: None, raising=False)
    with pytest.raises(ValueError, match='did not take'):
        entry.bind('validator', tmp_path)
    assert not list(tmp_path.iterdir())


def test_binding_requires_exact_parent_process_identity(tmp_path, monkeypatch):
    configure(monkeypatch, allowed=[0, 1, 2, 3])
    binding = entry.bind('command', tmp_path)
    parent = {'host_pid': binding['pid'], 'host_pgid': binding['pgid'], 'start_ticks': binding['start_ticks']}
    assert physical.valid_cpu_binding(binding, [0, 1, 2, 3], role='command', parent=parent)
    assert not physical.valid_cpu_binding(binding, [0, 1, 2, 3], role='command',
                                          parent={**parent, 'start_ticks': 99})
    assert not physical.valid_cpu_binding(binding, [0, 1, 2, 3], role='validator')


def test_stage_context_fails_closed_when_binding_missing(tmp_path, monkeypatch):
    configure(monkeypatch, allowed=[0, 1, 2, 3])
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'staged_environment': {'sha256': physical.RUNTIME_SHA}}))
    with pytest.raises(FileNotFoundError):
        physical.stage_context()


def test_preflight_uses_exact_bootstrap_without_importing_physical(tmp_path, monkeypatch):
    import sys

    configure(monkeypatch)
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['ball_confirmation_entry', '--preflight-only'])
    def no_run(*_args, **_kwargs):
        pytest.fail('preflight must not run the scientific module')
    monkeypatch.setattr(entry.runpy, 'run_module', no_run)
    entry.main()
    assert json.loads((tmp_path / 'command-cpu-binding.json').read_text())['selected'] == [2, 4, 6, 8]


def test_stage_context_rejects_foreign_process_binding(tmp_path, monkeypatch):
    configure(monkeypatch, allowed=[0, 1, 2, 3])
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'staged_environment': {'sha256': physical.RUNTIME_SHA}}))
    binding = entry.bind('command', tmp_path)
    (tmp_path / 'command-cpu-binding.json').write_text(json.dumps({**binding, 'pid': binding['pid'] + 1}))
    with pytest.raises(ValueError, match='affinity'):
        physical.stage_context()
