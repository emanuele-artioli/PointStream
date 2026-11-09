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
               'namespace_init': {'host_pid': 100, 'namespace_pid': 1, 'pid_namespace': 'pid:[11]'},
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
