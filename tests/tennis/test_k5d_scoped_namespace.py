from typing import Any

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


def test_outer_accepts_real_canonical_fleet_argv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import json
    import sys

    fake = tmp_path / 'unshare'
    fake.write_bytes(b'fixed executable')
    sha = hashlib.sha256(fake.read_bytes()).hexdigest()
    stage = tmp_path / 'smoke'
    stage.mkdir()
    arguments = ['outer', '--unshare-sha', sha, '--timeout', '{timeout}', '--', 'prepare', '--frames', '{frames}']
    spec: dict[str, Any] = {'entrypoint': ['-m', 'experiments.tennis.k5d_scoped_namespace'], 'arguments': arguments,
            'scale': {'timeout': {'smoke': 10, 'full': 10}, 'frames': {'smoke': 16, 'full': 16}},
            'smoke': {'seconds': 20}}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text('{"inputs": []}')
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setattr(sys, 'argv', ['module', 'outer', '--unshare-sha', sha, '--timeout', '10', '--', 'prepare', '--frames', '16'])
    monkeypatch.setattr('shutil.which', lambda _: str(fake))
    monkeypatch.setattr(namespace.os, 'readlink', lambda _: 'pid:[123]')
    calls = []
    monkeypatch.setattr(namespace.os, 'execv', lambda path, args: calls.append((path, args)))
    namespace.outer(sha, 10, ['prepare', '--frames', '16'])
    assert calls[0][0] == str(fake)
    assert calls[0][1][1:6] == ['--user', '--map-root-user', '--pid', '--fork', '--kill-child=SIGKILL']
    spec['arguments'][0] = 'inner'
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='launcher arguments'):
        namespace.outer(sha, 10, ['prepare', '--frames', '16'])


@pytest.mark.skipif(__import__('sys').platform != 'linux', reason='host procfs required')
def test_real_child_group_identity_and_rejection(tmp_path) -> None:
    import json
    import os
    import subprocess
    import sys

    status = tmp_path / 'status.json'
    status.write_text(json.dumps({'process_group_id': os.getpgrp()}))
    program = ('import json,sys; from pathlib import Path; '
               'from experiments.tennis.k5d_scoped_namespace import host_group_identity; '
               'print(json.dumps(host_group_identity(Path(sys.argv[1]))))')
    child = subprocess.run([sys.executable, '-c', program, str(status)], capture_output=True, text=True, check=True)
    identity = json.loads(child.stdout)
    assert identity['host_pgid'] == os.getpgrp() == identity['campaign_pgid']
    assert identity['host_pid'] > 0 and identity['start_ticks'] > 0
    rejected = subprocess.run([sys.executable, '-c', program, str(status)], capture_output=True, text=True,
                              start_new_session=True)
    assert rejected.returncode != 0 and 'worker host process group differs' in rejected.stderr


def test_real_process_inherits_group_unless_detached() -> None:
    import os
    import subprocess
    import sys

    command = [sys.executable, '-c', 'import os; print(os.getpid(), os.getpgrp())']
    inherited = subprocess.check_output(command, text=True).split()
    detached = subprocess.check_output(command, text=True, start_new_session=True).split()
    assert int(inherited[1]) == os.getpgrp()
    assert detached[0] == detached[1] and int(detached[1]) != os.getpgrp()
