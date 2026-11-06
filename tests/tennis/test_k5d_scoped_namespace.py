import pytest

from experiments.tennis import k5d_scoped_namespace as namespace


def test_constructs_fixed_worker_and_rejects_other_entrypoints() -> None:
    cmd = namespace.worker_command(['prepare', '--clip', 'tracknet/game9/Clip5'])
    assert cmd[1:5] == ['-m', namespace.TARGET, '--accounted-child', 'prepare']
    for args in ([], ['shell'], ['score', '--accounted-child'], ['prepare', '']):
        with pytest.raises(ValueError):
            namespace.worker_command(args)


def test_refuses_host_process() -> None:
    with pytest.raises(ValueError, match='namespace PID 1'):
        namespace.supervise(['prepare'], 1, 'pid:[1]')


def test_rejects_same_or_unmapped_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(namespace.sys, 'platform', 'linux')
    monkeypatch.setattr(namespace.os, 'getpid', lambda: 1)
    monkeypatch.setattr(namespace.os, 'readlink', lambda _: 'pid:[123]')
    monkeypatch.setattr(namespace.Path, 'read_text', lambda _: 'NSpid:\t1\n')
    with pytest.raises(ValueError, match='distinct nested'):
        namespace.supervise(['prepare'], 1, 'pid:[123]')
    with pytest.raises(ValueError, match='distinct nested'):
        namespace.supervise(['prepare'], 1, 'pid:[456]')


def test_launcher_rejects_non_fleet_entrypoint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    stage = tmp_path / 'smoke'
    stage.mkdir()
    (tmp_path / 'spec.json').write_text(json.dumps({'entrypoint': ['-m', 'wrong'], 'scale': {}, 'arguments': []}))
    (stage / 'staging.json').write_text('{"inputs": []}')
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    with pytest.raises(ValueError, match='fleet stage'):
        namespace.outer('0' * 64, 10, ['prepare'])


def test_native_proof_is_fixed_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    cmd = namespace.worker_command(['native-proof', '--archive', '/staged/source'])
    assert cmd[1:3] == ['-m', 'experiments.tennis.k5d_native_cpu_workload']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='fixed prepare'):
        namespace.worker_command(['native-proof', '--archive', '/staged/source'])


def test_gpu_parity_target_is_fixed_and_requires_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    cmd = namespace.worker_command(['parity16', '--frames', '16'])
    assert cmd[1:4] == ['-m', 'experiments.tennis.k5d_real_parity', 'run']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError):
        namespace.worker_command(['parity16', '--frames', '16'])
