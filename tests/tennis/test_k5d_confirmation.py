import json

import pytest

from experiments.tennis import k5d_confirmation as confirmation
from experiments.tennis import k5d_cost_pilot as cost
from experiments.tennis import k5d_encoder as encoder


def test_forecast_requires_current_smoke_and_campaign_room(tmp_path):
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 205.768}))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 100., 'phase_started': 110., 'budget_end': 2400.}}))
    result = confirmation.forecast(tmp_path, 400.)
    assert 1649 < result['full_estimate_seconds'] < 1651
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 230.}))
    with pytest.raises(ValueError, match='forecast'):
        confirmation.forecast(tmp_path, 400.)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 205.768}))
    with pytest.raises(ValueError, match='forecast'):
        confirmation.forecast(tmp_path, 800.)


def test_spec_requires_real_confirmation_receipt(tmp_path):
    from experiments.jobs.inbox import validate_spec

    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': confirmation.PURPOSE,
            'clip': cost.CLIP, 'max_frames': 188, 'archive_sha256': confirmation.fleet.EXPECTED['archive'],
            'calibration_job_id': cost.ACCEPTED_JOB, 'evidence_sha256': 'a' * 64,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'full_cost']}
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(gate))
    digest = encoder.oracle.sha(receipt)
    base = json.loads((confirmation.fleet.ROOT / 'docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json').read_text())
    gate_path = '/home/itec/emanuele/Datasets/pointstream-data/tennis/confirmation.json'
    spec = confirmation.make_spec(base, gate_path, digest, receipt, '2026-11-05T02:00:00+00:00')
    assert validate_spec(spec, now=1793830000)['budget_seconds'] == 2400
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 188}
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (500, 1750, 60)
    assert {e['name'] for e in spec['staged_inputs']} >= {'accepted16', 'cost16', 'fidelity_gate'}
    with pytest.raises(ValueError, match='SHA256'):
        confirmation.make_spec(base, gate_path, '0' * 64, receipt, spec['deadline'])
    gate['purpose'] = '48-frame-cost-pilot'
    receipt.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match='purpose'):
        confirmation.make_spec(base, gate_path, encoder.oracle.sha(receipt), receipt, spec['deadline'])
