import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tennis import w8_cuda_strace as trace
from experiments.tennis import w8_receiver_bootstrap as bootstrap


@pytest.mark.parametrize('cell', ['baseline', 'nnp', 'landlock'])
@pytest.mark.parametrize('initial_nnp', [0, 1])
def test_controls_apply_only_selected_policy_and_reject_existing_nnp(tmp_path, monkeypatch, capsys, cell, initial_nnp):
    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    protected = tmp_path / 'protected'
    protected.write_bytes(b'protected body')
    c = {'allow_paths': [str(allowed)], 'denied_paths': {'x': [str(protected)]},
         'parent_user_namespace': 'u0', 'parent_network_namespace': 'n0',
         'environment': {'CUDA_VISIBLE_DEVICES': 'GPU-test'}}
    monkeypatch.setattr(bootstrap, 'load_config', lambda *_: c)
    state = {'nnp': initial_nnp}
    calls = []
    def prctl(op, value, *_):
        if op == 39:
            return state['nnp']
        if op == 38:
            state['nnp'] = value
            return 0
        raise AssertionError(op)
    monkeypatch.setattr(trace.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(prctl=prctl))
    def restrict(paths):
        calls.append('restrict')
        state['nnp'] = 1
        return 4
    monkeypatch.setattr(bootstrap, '_restrict', restrict)
    monkeypatch.setattr(bootstrap, '_check_denials', lambda paths: calls.append('denials') or {str(protected): 13})
    monkeypatch.setattr(trace.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(trace.os, 'readlink', lambda path: 'u0' if path.endswith('/user') else 'n0')
    monkeypatch.setattr(trace.os, 'listdir', lambda path: ['1'] if path.endswith('/task') else ['0', '1', '2'])
    monkeypatch.setattr(trace.os, 'environ', {})
    monkeypatch.setattr(trace.resource, 'setrlimit', lambda *_: None)
    monkeypatch.setattr(trace.sys, 'addaudithook', lambda _: None)
    fake_cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 0)
    fake_torch = SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
    fake_input = SimpleNamespace(synthetic_torch_bootstrap=lambda _: calls.append('warmup') or 'same')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'experiments.tennis.w8_input_job', fake_input)
    source = str(Path(bootstrap.__file__).resolve().parents[2])
    status = trace.run_worker(argparse.Namespace(source=source, config='unused', config_sha256='sha', condition=cell))
    result = json.loads(capsys.readouterr().out)
    assert status == 1
    if initial_nnp:
        assert result['phase'] == 'bootstrap' and 'NoNewPrivs' in result['error']['message']
        assert 'restrict' not in calls
    else:
        assert result['phase'] == 'cuda_probe'
        assert result['isolation']['nnp_before'] == 0
        assert result['isolation']['nnp_after'] == (0 if cell == 'baseline' else 1)
        assert ('restrict' in calls) == (cell == 'landlock')
        assert ('denials' in calls) == (cell == 'landlock')
        assert calls.count('warmup') == 2
    assert protected.read_bytes() == b'protected body'


def test_identical_trace_instrumentation_and_denial_summary(tmp_path):
    commands = [trace.cell_command('/python', '/worker', cell, tmp_path / 'config', 'sha', Path('/source')) for cell in ('baseline', 'nnp', 'landlock')]
    assert all(c[:4] == ['/usr/bin/strace', '-f', '-s', '512'] for c in commands)
    assert all('/usr/bin/unshare' not in c for c in commands)
    sample = '123 openat(AT_FDCWD, "/protected", O_RDONLY) = -1 EACCES (Permission denied)\n123 ioctl(3, 0, 0) = -1 EPERM (Operation not permitted)\n'
    assert trace.denied_candidates(sample, ['/protected']) == ['123 ioctl(3, 0, 0) = -1 EPERM (Operation not permitted)']


def test_trace_spec_and_real_dispatch_fixture():
    from experiments.jobs.inbox import validate_spec
    fixture = Path(__file__).with_name('w8_portable_actual_dispatch_fixture.json')
    dispatch = json.loads(fixture.read_text())
    assert trace.check_dispatch(dispatch, Path(dispatch['staged_environment']['python']))['sha256'] == trace.RUNTIME_SHA
    spec = trace.make_spec('2026-11-05T03:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 150
    assert spec['smoke']['seconds'] == 120 and spec['full']['seconds'] == 1
    assert 'strace' in spec['required_commands'] and spec['hosts'] == ['gpu5', 'gpu6']
