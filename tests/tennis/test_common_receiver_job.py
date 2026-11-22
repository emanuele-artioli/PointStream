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


def test_receiver_accepts_only_units_bound_to_the_paid_carrier(tmp_path, monkeypatch):
    import argparse
    import hashlib
    from experiments.tennis import common_receiver_job as job

    carrier = tmp_path / 'carrier.pscp'
    carrier.write_bytes(b'fixture')
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setattr(job.packet, '_embedded_profile', lambda _: {'order': ['source_player']})
    monkeypatch.setattr(job.packet, '_profile', lambda _: None)
    args = argparse.Namespace(frames=44, weight=list(range(8)), deny=list(range(4)), full_cap=600,
                              carrier=carrier, carrier_sha256=hashlib.sha256(b'fixture').hexdigest(),
                              unit='source_player')
    job.validate_args(args)
    args.unit = 'unbound_player'
    with pytest.raises(ValueError, match='absent from the paid carrier'):
        job.validate_args(args)


def test_25fps_profile_uses_stricter_requested_margin():
    result = generation_forecast({'worker_wall_seconds': 250}, 175, 2500, 2400, 1.5)
    assert result['forecast_seconds'] == pytest.approx(1.5 * (75 + 175 * 240 / 44) + 90)
    with pytest.raises(ValueError, match='margin'):
        generation_forecast({'worker_wall_seconds': 250}, 175, 2500, 2400, 1.0)
