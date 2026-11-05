import json

import pytest

from experiments.tennis import k5d_native_pid_oracle as oracle


def test_validator_fails_closed_without_publication(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = tmp_path / 'smoke'
    stage.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    target = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_VALIDATION_PATH', str(target))
    with pytest.raises(FileNotFoundError):
        oracle.validate()
    assert json.loads(target.read_text())['passed'] is False


def test_cpu_proof_rejects_visible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_native_cpu_workload as workload
    from pathlib import Path

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'):
        workload.run(Path('/unused'), 'a' * 64, Path('/unused'), {})
