"""The 44-tick gate and post-run 240-tick audit have different clocks."""
import math
from pathlib import Path

import pytest

from experiments.tennis.common_receiver_job import actual_full_cost, generation_forecast, validation_workspace


def test_measured_44_tick_cost_caps_the_full_run() -> None:
    result = generation_forecast({'worker_wall_seconds': 250}, 175, 2000, 1800)
    assert result['forecast_seconds'] == pytest.approx(1.25 * (75 + 175 * 240 / 44) + 90)
    with pytest.raises(ValueError, match='forecast'):
        generation_forecast({'worker_wall_seconds': 250}, 175, 900, 1800)


def test_full_receipt_cost_uses_observed_wall_only() -> None:
    assert actual_full_cost({'worker_wall_seconds': 1052}, 1800)['actual_full_worker_seconds'] == 1052
    for invalid in (0, math.nan, math.inf, 1801):
        with pytest.raises(ValueError):
            actual_full_cost({'worker_wall_seconds': invalid}, 1800)


def test_validator_recreates_released_fleet_scratch(tmp_path, monkeypatch):
    scratch = tmp_path / "retired-stage" / "scratch"
    monkeypatch.setenv("PS_SCRATCH_DIR", str(scratch))
    with validation_workspace() as folder:
        extracted = Path(folder)
        assert extracted.parent == scratch and extracted.is_dir()
        (extracted / "receipt.json").write_text("{}")
    assert scratch.is_dir() and not extracted.exists()
