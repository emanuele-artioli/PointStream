import json
from pathlib import Path

import pytest

from experiments.tennis import k5d_controls as controls
from experiments.tennis import k5d_controls_confirmation as confirmation


def test_spec_binds_real_audit_and_188_calibration():
    from experiments.jobs.inbox import validate_spec
    audit = Path('docs/experiments/k5d-inputs/controls-3ec159d8-independent-audit-20261105.json')
    spec = confirmation.make_spec('/home/itec/emanuele/Datasets/pointstream-data/analysis/controls/audit.json', audit, '2026-11-05T04:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 550
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 188}
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (100, 400, 30)
    entries = {row['name']: row for row in spec['staged_inputs']}
    for name, sha in {**confirmation.PILOT_SHA, **confirmation.CALIBRATION_SHA}.items():
        assert entries[name]['sha256'] == sha
        assert {'path': entries[name]['path'], 'sha256': sha} in spec['inputs']
    with pytest.raises(ValueError, match='data-root'):
        confirmation.make_spec('/tmp/audit.json', audit, spec['deadline'])


def test_rejects_forged_audit_and_forecast_respects_slowdown(tmp_path):
    original = Path('docs/experiments/k5d-inputs/controls-3ec159d8-independent-audit-20261105.json')
    audit = json.loads(original.read_text())
    s16, s48 = audit['stages']['smoke']['command_seconds'], audit['stages']['full']['command_seconds']
    assert controls.full188_forecast(s16, s48, s16, reserve=40) == pytest.approx(375.105, abs=.01)
    assert controls.full188_forecast(s16, s48, 1.1 * s16, reserve=40) > 400
    fake = tmp_path / 'audit.json'
    fake.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='SHA'):
        confirmation.make_spec('/home/itec/emanuele/Datasets/pointstream-data/analysis/audit.json', fake, '2026-11-05T04:00:00+00:00')
