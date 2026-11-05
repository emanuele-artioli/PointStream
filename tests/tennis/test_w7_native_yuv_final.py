import json
from pathlib import Path

import pytest

from experiments.jobs.inbox import validate_spec
from experiments.tennis import w7_native_yuv_final as final


def test_final_spec_is_gpu5_only_and_retains_reallocated_envelope():
    spec = json.loads(Path('docs/experiments/w7-inputs/mkv-native-yuv-final-spec-20261105.json').read_text())
    validate_spec(spec)
    assert spec['hosts'] == ['gpu5'] and spec['device'] == 'cpu'
    assert (spec['budget_seconds'], spec['external_audit_reserve_seconds']) == (760, 90)
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (40, 620, 45)
    assert len(spec['staged_inputs']) == 18


def test_final_forecast_uses_worse_historical_or_current_observation():
    historic = 4.8400312890298665
    assert final.final_forecast(1.3, historic, 30, 700) == pytest.approx(1.5 * 75 * historic + 30)
    assert final.final_forecast(5.0, historic, 30, 700) == pytest.approx(1.5 * 75 * 5 + 30)
    with pytest.raises(ValueError, match='cannot fit'):
        final.final_forecast(6, historic, 30, 700)
    with pytest.raises(ValueError, match='cannot fit'):
        final.final_forecast(1.3, historic, 200, 700)
    with pytest.raises(ValueError, match='invalid'):
        final.final_forecast(float('nan'), historic, 30, 700)
