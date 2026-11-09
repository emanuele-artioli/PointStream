import json

import pytest

from experiments.tennis import k5d_real_parity as parity


def test_validator_requires_published_real_parity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / 'smoke'
    stage.mkdir()
    target = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_VALIDATION_PATH', str(target))
    with pytest.raises(FileNotFoundError):
        parity.validate()
    assert json.loads(target.read_text())['passed'] is False


def test_run_rejects_absent_cpu_qualification(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setattr(parity.cp, 'execution_identity', lambda: {})
    cpu = tmp_path / 'cpu.json'
    cpu.write_text('{}')
    args = argparse.Namespace(frames=16, native_cpu_receipt=str(cpu), native_cpu_receipt_sha256='0' * 64)
    with pytest.raises(ValueError, match='audited native CPU proof'):
        parity.run(args)


def test_parity_spec_requires_actual_cpu_and_reference_assets() -> None:
    with pytest.raises(ValueError, match='audited CPU receipt'):
        parity.spec({}, assets={}, deadline='later', unshare_sha256='0' * 64, identity={})


def test_parity_spec_stays_within_smoke_limit_and_uses_actual_cpu_receipt(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.jobs import inbox

    cpu = tmp_path / 'cpu.json'
    cpu.write_text(json.dumps({'audited': True, 'passed': True, 'scope': 'native CPU PID containment and accounting', 'gpu_used': False,
                               'unshare_sha256': 'a' * 64, 'namespace_wrapper_sha256': 'b' * 64,
                               'consumer_execution_identity': {}}))
    assets = {'fidelity_gate': {'local': str(cpu), 'remote': '/fidelity', 'sha256': 'c' * 64},
              'accepted16': {'local': str(cpu), 'remote': '/accepted', 'sha256': parity.cf.D8_SMOKE_SHA},
              'native_cpu_receipt': {'local': str(cpu), 'remote': '/cpu', 'sha256': 'd' * 64}}
    monkeypatch.setattr(parity.cf, 'asset', lambda name, *_args, **_kwargs: {'name': name})
    monkeypatch.setattr(parity.cf, 'common_spec', lambda *_: {'scale': {}})
    monkeypatch.setattr(parity.oracle, 'sha', lambda *_: 'b' * 64)
    monkeypatch.setattr(inbox, 'validate_spec', lambda *_: None)
    result = parity.spec({'arguments': []}, assets=assets, deadline='later', unshare_sha256='a' * 64, identity={})
    assert result['smoke']['seconds'] == 600
    assert result['scale']['namespace_timeout']['smoke'] == 580
    assert result['gpu_sharing'] == {'mode': 'exclusive'}
    assert result['scale']['proof_mode'] == {'smoke': 'parity16', 'full': 'parity-noop'}
    assert result['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace']


def test_parity_builder_passes_real_fleet_validator_and_outer_cli(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import sys
    from pathlib import Path
    from datetime import datetime, timedelta, timezone
    from experiments.jobs import inbox
    from experiments.tennis import k5d_scoped_namespace as namespace

    wrapper_sha = hashlib.sha256(Path(namespace.__file__).read_bytes()).hexdigest()
    cpu = tmp_path / 'cpu.json'
    cpu.write_text(json.dumps({'audited': True, 'passed': True, 'scope': 'native CPU PID containment and accounting',
                               'gpu_used': False, 'unshare_sha256': 'a' * 64,
                               'namespace_wrapper_sha256': wrapper_sha, 'consumer_execution_identity': {}}))
    remote = '/home/itec/emanuele/Datasets/pointstream-data/test/'
    assets = {'fidelity_gate': {'local': str(cpu), 'remote': remote + 'fidelity.json', 'sha256': 'c' * 64},
              'accepted16': {'local': str(cpu), 'remote': remote + 'accepted.tar', 'sha256': parity.cf.D8_SMOKE_SHA},
              'native_cpu_receipt': {'local': str(cpu), 'remote': remote + 'cpu.json', 'sha256': 'd' * 64}}
    monkeypatch.setattr(parity.cf, 'asset', lambda name, path, digest, _local, *, extract:
                        {'name': name, 'path': path, 'sha256': digest, 'extract': extract})
    base = json.loads(parity.cf.BASE.read_text())
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    result = parity.spec(base, assets=assets, deadline=deadline, unshare_sha256='a' * 64, identity={})
    normalized = inbox.validate_spec(result)
    shared = parity.spec(base, assets=assets, deadline=deadline, unshare_sha256='a' * 64,
                         identity={}, shared_numerical=True)
    shared_normalized = inbox.validate_spec(shared)
    assert shared_normalized['gpu_sharing']['mode'] == 'shared'
    assert shared_normalized['gpu_memory_mib'] == 16384
    assert shared_normalized['contention']['resume_attempts'] == 0
    assert '--shared-numerical-only' in shared['arguments']
    assert '--shared-numerical-only' not in result['arguments']
    legacy = {**result, 'entrypoint': [*result['entrypoint'], 'outer'], 'arguments': result['arguments'][1:]}
    with pytest.raises(inbox.fleet.FleetError, match='entrypoint must be'):
        inbox.validate_spec(legacy)
    assert normalized['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace']
    assert result['arguments'][0] == 'outer'
    staged = [{'name': item['name'], 'path': '/staged/' + item['name']} for item in result['staged_inputs']]
    command = inbox.phase_command(result, 'smoke', staged, '/runtime/python')
    assert command[:4] == ['/runtime/python', '-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']
    captured = []
    monkeypatch.setattr(namespace, 'outer', lambda sha, timeout, args: captured.append((sha, timeout, args)))
    monkeypatch.setattr(sys, 'argv', ['module', *command[3:]])
    namespace.main()
    assert captured[0][0] == 'a' * 64 and captured[0][1] == 580
    assert captured[0][2][0] == 'parity16'
