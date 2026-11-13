import argparse
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tennis import w8_cuda_system_leaves as pilot
from experiments.tennis import w8_receiver_bootstrap as bootstrap


@pytest.mark.parametrize('cell', ['portable', 'leaves', 'strict'])
def test_three_namespace_controls_preserve_warmup_and_selected_policy(tmp_path, monkeypatch, capsys, cell):
    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    protected = tmp_path / 'protected'
    protected.write_bytes(b'protected body')
    c = {'allow_paths': [str(allowed), str(output)], 'output': str(output),
         'denied_paths': {'x': [str(protected)]}, 'parent_user_namespace': 'u0',
         'parent_network_namespace': 'n0', 'environment': {'CUDA_VISIBLE_DEVICES': 'GPU-test'}}
    monkeypatch.setattr(bootstrap, 'load_config', lambda *_: c)
    calls = []
    monkeypatch.setattr(bootstrap, '_restrict', lambda paths: calls.append(('read', len(paths))) or 4)
    monkeypatch.setattr(bootstrap, '_check_denials', lambda paths: {str(protected): 13})
    leaves = [Path('/approved')]
    monkeypatch.setattr(pilot, 'approved_leaves', lambda denied, loader: (leaves, [{'name': 'fixture'}]))
    monkeypatch.setattr(pilot, 'restrict', lambda paths, dest: (calls.append(('strict', len(paths))) or (4, 32767)))
    monkeypatch.setattr(pilot, 'mutation_probes', lambda canary, dest: {'write': 13})
    monkeypatch.setattr(pilot.os, 'sched_getaffinity', lambda _: {0}, raising=False)
    monkeypatch.setattr(pilot.os, 'readlink', lambda path: 'u1' if path.endswith('/user') else 'n1')
    monkeypatch.setattr(pilot.os, 'listdir', lambda path: ['1'] if path.endswith('/task') else ['0', '1', '2'])
    monkeypatch.setattr(pilot.os, 'environ', {})
    monkeypatch.setattr(pilot.resource, 'setrlimit', lambda *_: None)
    monkeypatch.setattr(pilot.sys, 'addaudithook', lambda _: None)
    import socket
    monkeypatch.setattr(socket, 'if_nameindex', lambda: [(1, 'lo')])
    fake_cuda = SimpleNamespace(is_initialized=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None))
    monkeypatch.setitem(sys.modules, 'experiments.tennis.w8_input_job', SimpleNamespace(synthetic_torch_bootstrap=lambda _: 'same'))
    source = str(Path(bootstrap.__file__).resolve().parents[2])
    loader_file = tmp_path / 'loader-identity.json'
    loader_file.write_text(json.dumps({'device': 1, 'inode': 2, 'uid': 0, 'mode': stat.S_IFREG | 0o644}))
    status = pilot.run_worker(argparse.Namespace(source=source, config=str(tmp_path / 'config'), config_sha256='sha', condition=cell, loader_sha256=pilot.digest(loader_file)))
    result = json.loads(capsys.readouterr().out)
    assert status == 1 and result['phase'] == 'cuda_probe'
    assert calls == [({'portable': 'read', 'leaves': 'read', 'strict': 'strict'}[cell], 2 if cell == 'portable' else 3)]
    assert result['isolation']['denials'] == {str(protected): 13}
    assert result['isolation']['output_write_ok'] == (True if cell == 'strict' else None)
    assert protected.read_bytes() == b'protected body'


def test_leaf_receipt_is_fixed_and_self_resolves_to_a_child_pid():
    rows = []
    for name in pilot.APPROVED_LEAVES:
        resolved = name.replace('/proc/self/', '/proc/1234/')
        rows.append({'name': name, 'resolved': resolved, 'device': 1, 'inode': 2,
                     'mode': stat.S_IFREG | 0o644, 'uid': 0})
    loader = {'device': 1, 'inode': 2, 'uid': 0, 'mode': stat.S_IFREG | 0o644}
    pilot.validate_leaf_receipt(rows, loader)
    rows[6]['resolved'] = '/proc/self/maps'
    with pytest.raises(ValueError, match='child-resolved'):
        pilot.validate_leaf_receipt(rows, loader)
    rows[6]['resolved'] = '/proc/1234/maps'
    rows[0]['mode'] |= stat.S_IWOTH
    with pytest.raises(ValueError, match='unsafe'):
        pilot.validate_leaf_receipt(rows, loader)


def test_mutation_abi_and_output_only_write(tmp_path):
    with pytest.raises(RuntimeError, match='ABI >=3'):
        pilot.handled_rights(2)
    assert pilot.handled_rights(3) & pilot.TRUNCATE
    source = tmp_path / 'source'
    output = tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    assert not pilot.allowed_rights(source, output, 3) & pilot.WRITE_FILE
    assert pilot.allowed_rights(output, output, 3) & pilot.WRITE_FILE


def test_spec_and_fresh_namespace_commands():
    from experiments.jobs.inbox import validate_spec
    spec = pilot.make_spec('2026-11-05T03:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 120
    assert spec['smoke']['seconds'] == 100 and spec['hosts'] == ['gpu5', 'gpu6']
    for name in ('portable', 'leaves', 'strict'):
        cmd = pilot.cell_command('/python', '/script', name, Path('/config'), 'hash', Path('/source'), 'loader-sha')
        assert cmd[:4] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
        assert cmd[cmd.index('--condition') + 1] == name
    fixture = Path(__file__).with_name('w8_portable_actual_dispatch_fixture.json')
    dispatch = json.loads(fixture.read_text())
    assert pilot.check_dispatch(dispatch, Path(dispatch['staged_environment']['python']))['sha256'] == pilot.RUNTIME_SHA


def test_self_leaves_join_exact_namespace_worker_identity():
    worker = {'host_pid': 1234, 'host_pgid': 987, 'start_ticks': 42, 'campaign_pgid': 987}
    identity = {key: worker[key] for key in ('host_pid', 'host_pgid', 'start_ticks')}
    rows = [{'name': f'/proc/self/{name}', 'resolved': f'/proc/1234/{name}', 'self_process': dict(identity)}
            for name in ('maps', 'status')]
    pilot.validate_self_leaf_identity(rows, worker)
    for key in ('host_pid', 'host_pgid', 'start_ticks'):
        changed = dict(worker, **{key: worker[key] + 1})
        with pytest.raises(ValueError, match='namespace worker'):
            pilot.validate_self_leaf_identity(rows, changed)
    rows[0]['resolved'] = '/proc/1235/maps'
    with pytest.raises(ValueError, match='namespace worker'):
        pilot.validate_self_leaf_identity(rows, worker)


def test_system_leaf_cannot_alias_protected_inode(tmp_path, monkeypatch):
    leaf = tmp_path / 'public'
    leaf.write_text('fixture')
    monkeypatch.setattr(pilot, 'APPROVED_LEAVES', (str(leaf),))
    monkeypatch.setattr(pilot, 'self_process_identity', lambda: {'host_pid': 1, 'host_pgid': 1, 'start_ticks': 1})
    with pytest.raises(ValueError, match='aliases protected'):
        pilot.approved_leaves([leaf], {})
