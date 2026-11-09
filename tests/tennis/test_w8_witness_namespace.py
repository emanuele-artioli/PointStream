import json
import sys

import pytest

from experiments.tennis import w8_witness_namespace as namespace


def test_fixed_targets_only():
    target = 'experiments.tennis.w8_strict_receiver_worker'
    assert namespace.command(target, ['--config', 'x']) == [sys.executable, '-m', target, '--config', 'x']
    with pytest.raises(ValueError):
        namespace.command('os', [])


def test_group_identity_reads_actual_process(tmp_path):
    if sys.platform != 'linux':
        pytest.skip('Linux procfs required')
    import os
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'process_group_id': os.getpgrp()}))
    result = namespace.identity(path)
    assert result['host_pid'] == os.getpid()
    assert result['host_pgid'] == result['campaign_pgid'] == os.getpgrp()
    assert result['start_ticks'] > 0
    path.write_text(json.dumps({'process_group_id': -1}))
    with pytest.raises(ValueError):
        namespace.identity(path)


def test_rejects_non_init(tmp_path):
    with pytest.raises(ValueError):
        namespace.supervise('pid:[1]', 1, tmp_path / 'receipt', 'os', [])


def test_receipt_rejects_wrong_group_and_failed_cleanup(tmp_path):
    target = 'experiments.tennis.w8_strict_receiver_worker'
    receipt = {'schema': 1, 'reason': 'complete', 'status': 0, 'target': target,
               'outside_namespace': 'pid:[10]', 'worker_namespace_pid': 2,
               'worker_host_identity': {'host_pid': 101, 'host_pgid': 99, 'campaign_pgid': 99, 'start_ticks': 5},
               'namespace_init': {'host_pid': 100, 'start_ticks': 4, 'host_pgid': 99,
                                  'namespace_pid': 1, 'pid_namespace': 'pid:[11]'},
               'reaped': 1, 'complete_descendant_wait': True, 'elapsed_seconds': 1.0, 'children_cpu_seconds': .1}
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt))
    check = dict(expected_pgid=99, expected_outside='pid:[10]', expected_target=target)
    assert namespace.validate_receipt(path, **check)['status'] == 0
    for key, value in [('complete_descendant_wait', False), ('reason', 'timeout'), ('worker_namespace_pid', 3)]:
        bad = {**receipt, key: value}
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            namespace.validate_receipt(path, **check)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        namespace.validate_receipt(path, **{**check, 'expected_pgid': 98})


def test_cpu_proof_target_is_fixed_and_not_a_candidate(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0}, raising=False)
    output = tmp_path / 'proof'
    output.mkdir()
    args = ['worker', 'normal', '--output', str(output)]
    assert namespace.command(namespace.PROOF_TARGET, args)[3:] == args
    with pytest.raises(ValueError):
        namespace.command(namespace.PROOF_TARGET, ['worker', 'arbitrary', '--output', str(output)])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-123')
    with pytest.raises(ValueError):
        namespace.command(namespace.PROOF_TARGET, args)
    with pytest.raises(ValueError):
        namespace.command(namespace.PROOF_TARGET, ['worker', 'normal', '--output', str(tmp_path)])


def test_independent_waited_cpu_counter():
    from experiments.tennis.w8_witness_namespace_proof import check_waited_cpu

    check_waited_cpu(1.1, .1, .9)
    with pytest.raises(ValueError):
        check_waited_cpu(1.4, .1, .9)
    with pytest.raises(ValueError):
        check_waited_cpu(float('nan'), .1, .9)


def test_proof_worker_rejects_unisolated_execution(tmp_path, monkeypatch):
    from experiments.tennis import w8_witness_namespace_proof as proof

    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-not-hidden')
    output = tmp_path / 'case'
    output.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        proof.worker('normal', output)


def test_native_target_requires_fixed_cpu_config_and_scratch(tmp_path, monkeypatch):
    import hashlib
    import os

    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0}, raising=False)
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('PYTHONPATH', str(namespace.Path(namespace.__file__).resolve().parents[2]))
    config = tmp_path / 'native.json'
    config.write_text('{}')
    args = ['--phase', 'launch', '--config', str(config), '--config-sha256',
            hashlib.sha256(config.read_bytes()).hexdigest(), '--parent-cpu', '2', '--worker-cpu', '3',
            '--child-cpu', '4', '--wall-seconds', '5', '--control', str(tmp_path / 'control')]
    assert namespace.command(namespace.NATIVE_TARGET, args)[3:] == args
    with pytest.raises(ValueError):
        namespace.command(namespace.NATIVE_TARGET, [*args[:-1], str(tmp_path.parent / 'escape')])
    with pytest.raises(ValueError):
        namespace.command(namespace.NATIVE_TARGET, [*args[:5], '0' * 64, *args[6:]])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-not-hidden')
    with pytest.raises(ValueError):
        namespace.command(namespace.NATIVE_TARGET, args)
    assert namespace.NATIVE_TARGET not in namespace.TARGETS


def test_kill_remaining_only_accepts_missing_process(monkeypatch):
    from experiments.tennis import w8_witness_namespace as namespace
    def missing(_pid, _sig):
        raise ProcessLookupError()
    monkeypatch.setattr(namespace.os, 'kill', missing)
    namespace.kill_remaining()
    def denied(_pid, _sig):
        raise PermissionError()
    monkeypatch.setattr(namespace.os, 'kill', denied)
    with pytest.raises(PermissionError):
        namespace.kill_remaining()


def test_packet_target_is_fixed_and_cuda_hidden(monkeypatch):
    target = namespace.PACKET_TARGET
    args = ['--config', 'a', '--config-sha256', 'b', '--loader', 'c', '--loader-sha256', 'd',
            '--canary', 'e', '--source-root', 'f', '--witness-mode', 'packet']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert namespace.command(target, args) == [sys.executable, '-m', target, *args]
    for bad in (args[:-2], [*args, '--extra', 'x'], [*args[:-1], 'other']):
        with pytest.raises(ValueError, match='fixed CPU witness packet'):
            namespace.command(target, bad)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-not-hidden')
    with pytest.raises(ValueError, match='fixed CPU witness packet'):
        namespace.command(target, args)


def test_term_wait_escalates_only_when_needed_and_fails_on_unkillable():
    import subprocess
    from experiments.tennis import w8_witness_namespace_proof as proof

    class Launcher:
        def __init__(self, timeouts):
            self.timeouts = iter(timeouts)
            self.killed = 0
        def wait(self, timeout):
            outcome = next(self.timeouts)
            if outcome == 'timeout':
                raise subprocess.TimeoutExpired('launcher', timeout)
            return outcome
        def kill(self):
            self.killed += 1
    clean = Launcher([-15])
    assert proof.wait_after_term(clean) == (-15, False) and clean.killed == 0
    escalation = Launcher(['timeout', -9])
    assert proof.wait_after_term(escalation) == (-9, True) and escalation.killed == 1
    stuck = Launcher(['timeout', 'timeout'])
    with pytest.raises(subprocess.TimeoutExpired):
        proof.wait_after_term(stuck)
    assert stuck.killed == 1


def test_detached_group_absence_rejects_live_or_inaccessible(monkeypatch):
    from experiments.tennis import w8_witness_namespace_proof as proof
    monkeypatch.setattr(proof.os, 'killpg', lambda *_: None)
    assert proof.group_absent(123) is False
    def inaccessible(*_):
        raise PermissionError()
    monkeypatch.setattr(proof.os, 'killpg', inaccessible)
    assert proof.group_absent(123) is False
    def missing(*_):
        raise ProcessLookupError()
    monkeypatch.setattr(proof.os, 'killpg', missing)
    assert proof.group_absent(123) is True
