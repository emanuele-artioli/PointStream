import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tennis import w8_cuda_factorial as factorial
from experiments.tennis import w8_receiver_bootstrap as bootstrap


@pytest.mark.parametrize('condition', ['n0l0', 'n1l0', 'n0l1', 'n1l1'])
@pytest.mark.parametrize('fault', ['none', 'preinitialized', 'thread', 'fd'])
def test_all_cells_use_same_warmup_and_conditional_controls(tmp_path, monkeypatch, capsys, condition, fault):
    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    protected = tmp_path / 'protected'
    protected.write_bytes(b'never read')
    namespace = condition[1] == '1'
    landlock = condition[3] == '1'
    c = {'allow_paths': [str(allowed)], 'denied_paths': {'x': [str(protected)]},
         'parent_user_namespace': 'u0', 'parent_network_namespace': 'n0',
         'environment': {'CUDA_VISIBLE_DEVICES': 'GPU-test'}}
    monkeypatch.setattr(bootstrap, 'load_config', lambda *_: c)
    calls = []
    monkeypatch.setattr(bootstrap, '_restrict', lambda roots: calls.append(('restrict', roots)) or 4)
    monkeypatch.setattr(bootstrap, '_check_denials', lambda paths: calls.append(('denials', paths)) or {str(protected): 13})
    monkeypatch.setattr(factorial.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(factorial.os, 'readlink', lambda path: ('u1' if namespace else 'u0') if path.endswith('/user') else ('n1' if namespace else 'n0'))
    monkeypatch.setattr(factorial.os, 'listdir', lambda path: (['1', '2'] if fault == 'thread' else ['1']) if path.endswith('/task') else (['0', '1', '2', '3'] if fault == 'fd' else ['0', '1', '2']))
    monkeypatch.setattr(factorial.os, 'environ', {})
    monkeypatch.setattr(factorial.resource, 'setrlimit', lambda *_: None)
    monkeypatch.setattr(factorial.sys, 'addaudithook', lambda _: None)
    import socket
    monkeypatch.setattr(socket, 'if_nameindex', lambda: [(1, 'lo')])
    original_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda p: True if str(p) == '/proc/self/fd/3' else original_exists(p))
    fake_cuda = SimpleNamespace(is_initialized=lambda: fault == 'preinitialized', device_count=lambda: 0)
    fake_torch = SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
    fake_input = SimpleNamespace(synthetic_torch_bootstrap=lambda _: calls.append(('warmup', None)) or 'same')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'experiments.tennis.w8_input_job', fake_input)
    source = str(Path(bootstrap.__file__).resolve().parents[2])
    status = factorial.run_worker(argparse.Namespace(source=source, config='unused', config_sha256='sha', condition=condition))
    result = json.loads(capsys.readouterr().out)
    assert status == 1
    if fault != 'none':
        assert result['phase'] == 'bootstrap' and 'isolation' not in result
        assert not any(x[0] == 'restrict' for x in calls)
        return
    assert result['phase'] == 'cuda_probe'
    assert result['isolation']['cpu_before'] == result['isolation']['cpu_after'] == 'same'
    assert [x[0] for x in calls].count('warmup') == 2
    assert ('restrict' in [x[0] for x in calls]) is landlock
    assert ('denials' in [x[0] for x in calls]) is landlock
    assert result['isolation']['handled_mask'] == (13 if landlock else 0)
    assert protected.read_bytes() == b'never read'


def test_cuda_failure_is_not_a_successful_generation_gate(tmp_path):
    assert factorial.PORTABLE_PROFILE != factorial.PROFILE
    assert factorial.canonical_gpu_uuid('ef96fc4f-134e-4c01-290e-9a6c8bbef553') == factorial.canonical_gpu_uuid('GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553')


@pytest.mark.parametrize('condition', ['n0l0', 'n1l0', 'n0l1', 'n1l1'])
def test_fresh_process_command_only_changes_namespace_factor(condition):
    cmd = factorial.cell_command('/runtime/bin/python', '/worker.py', condition, Path('/config'), 'sha', Path('/source'))
    if condition[1] == '1':
        assert cmd[:4] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
        cmd = cmd[4:]
    assert cmd == ['/runtime/bin/python', '/worker.py', 'worker', '--condition', condition,
                   '--config', '/config', '--config-sha256', 'sha', '--source', '/source']


def test_rejects_unknown_condition():
    with pytest.raises(ValueError):
        factorial.cell_command('python', 'script', 'n2l1', Path('/config'), 'sha', Path('/source'))


def test_actual_dispatch_requires_runtime_identity_and_exclusivity():
    import copy

    fixture = Path(__file__).with_name('w8_portable_actual_dispatch_fixture.json')
    dispatch = json.loads(fixture.read_text())
    python = Path(dispatch['staged_environment']['python'])
    assert factorial.check_dispatch(dispatch, python)['sha256'] == factorial.RUNTIME_SHA
    altered = copy.deepcopy(dispatch)
    altered['gpu_sharing']['mode'] = 'shared'
    with pytest.raises(ValueError, match='identity'):
        factorial.check_dispatch(altered, python)
    altered = copy.deepcopy(dispatch)
    altered['staged_environment']['sha256'] = 'bad'
    with pytest.raises(ValueError, match='identity'):
        factorial.check_dispatch(altered, python)


def test_spec_has_bounded_four_cell_scope():
    from experiments.jobs.inbox import validate_spec
    spec = factorial.make_spec('2026-11-05T03:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 300
    assert spec['smoke']['seconds'] == 260
    assert spec['full']['seconds'] == 1
    assert spec['gpu_models'] == ['RTX 6000 Ada']
    assert spec['hosts'] == ['gpu5', 'gpu6']
