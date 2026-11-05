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
    cpu = tmp_path / 'cpu.json'
    cpu.write_text('{}')
    args = argparse.Namespace(frames=16, native_cpu_receipt=str(cpu), native_cpu_receipt_sha256='0' * 64)
    with pytest.raises(ValueError, match='audited native CPU proof'):
        parity.run(args)


def test_parity_spec_requires_actual_cpu_and_reference_assets() -> None:
    with pytest.raises(ValueError, match='audited CPU receipt'):
        parity.spec({}, assets={}, deadline='later', unshare_sha256='0' * 64)
