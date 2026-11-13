import json
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import k5d_cpu_accounting_oracle as oracle


def test_worker_uses_small_bounded_cpu_and_unique_receipts(tmp_path: Path) -> None:
    oracle.work(tmp_path, 'direct')
    row = json.loads((tmp_path / 'direct.json').read_text())
    assert row['pid'] > 0 and row['self_cpu_seconds'] > 0
    with pytest.raises(FileExistsError):
        oracle.work(tmp_path, 'direct')


def test_rejects_non_linux_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle.sys, 'platform', 'darwin')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='CPU-only'):
        oracle.execute()


def test_phase_and_source_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle.sys, 'argv', ['oracle', 'run', '--phase', 'smoke'])
    monkeypatch.setenv('PS_STAGE', 'smoke')
    with pytest.raises(SystemExit):
        oracle.main()


def test_validator_emits_failure_when_publication_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    output = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_VALIDATION_PATH', str(output))
    with pytest.raises(FileNotFoundError):
        oracle.validate()
    result: dict[str, Any] = json.loads(output.read_text())
    assert result['passed'] is False and result['checks']['synthetic_accounting'] is False
