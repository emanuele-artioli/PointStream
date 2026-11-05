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
    assert result['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']
