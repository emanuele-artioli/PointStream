import copy
import json
from pathlib import Path

import pytest

from experiments.tennis import w8_strict_host_cost_job as job


def draft():
    path = Path('docs/experiments/w8-inputs/receiver-20261101/w8-paired-cost-800-draft-20261105.json')
    return json.loads(path.read_text())


def make_stage(tmp_path, spec):
    stage = tmp_path / 'smoke'
    stage.mkdir()
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    rows = []
    for item in spec['staged_inputs']:
        target = tmp_path / item['name']
        target.touch()
        rows.append({**item, 'path': str(target)})
    (stage / 'staging.json').write_text(json.dumps({'inputs': rows}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [0]}))
    expected = {item['name']: item['sha256'] for item in rows}
    return stage, lambda p: expected[p.name]


def test_profile_accepts_exact_inputs_and_default_stays_bounded(tmp_path):
    spec = draft()
    stage, digest = make_stage(tmp_path, spec)
    paths, _ = job.stage_inputs(stage, digest)
    assert len(paths) == 15
    legacy = copy.deepcopy(spec)
    legacy.pop('w8_paired_cost_profile')
    (tmp_path / 'spec.json').write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='bounded budget'):
        job.stage_inputs(stage, digest)


@pytest.mark.parametrize('change', ['unknown', 'full30', 'generator', 'validator', 'cpu', 'missing', 'extra', 'tamper'])
def test_reject_invalid_profile_or_unbound_inputs(tmp_path, change):
    spec = draft()
    if change == 'unknown':
        spec['w8_paired_cost_profile'] = 'unknown'
    elif change == 'full30':
        spec['scale']['steps']['full'] = 30
    elif change == 'generator':
        spec['arguments'][spec['arguments'].index('--generator-cpu') + 1] = '300'
    elif change == 'validator':
        spec['validator_seconds'] = 59
    elif change == 'cpu':
        spec['receiver_cpu_budget_seconds'] = 999
    elif change == 'missing':
        spec['staged_inputs'] = [x for x in spec['staged_inputs'] if x['name'] != 'identity']
    elif change == 'extra':
        spec['staged_inputs'].append({'name': 'extra', 'sha256': '0' * 64, 'path': '/extra'})
    elif change == 'tamper':
        next(x for x in spec['staged_inputs'] if x['name'] == 'w1')['sha256'] = '0' * 64
    stage, digest = make_stage(tmp_path, spec)
    with pytest.raises(ValueError):
        job.stage_inputs(stage, digest)
