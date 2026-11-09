from datetime import datetime, timedelta, timezone

import pytest

from experiments.jobs.inbox import phase_command, validate_spec
from experiments.tennis import w8_witness_cpu_proof_job as proof


def test_real_fleet_spec_is_cpu_only_and_dispatches_fixed_phases():
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    spec = proof.make_spec(deadline, '/home/itec/emanuele/Datasets/pointstream-data/test/inventory.json')
    validate_spec(spec)
    assert spec['device'] == 'cpu' and spec['gpu_models'] == [] and spec['gpu_memory_mib'] == 0
    assert spec['budget_seconds'] == 600 and spec['cpu_threads'] == 1
    assert phase_command(spec, 'smoke', python='/python') == ['/python', '-m',
        'experiments.tennis.w8_witness_cpu_proof_job', 'run']
    assert phase_command(spec, 'full', python='/python')[-1] == 'noop'
    assert {x['name'] for x in spec['staged_inputs']} == set(proof.packet.NAMES) | {'native_inventory'}


def test_inventory_must_be_registered_absolute_path():
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    for name in ('/tmp/inventory.json', 'relative', '/home/itec/emanuele/Datasets/pointstream-data/../x'):
        with pytest.raises(ValueError):
            proof.make_spec(deadline, name)


def test_lifecycle_rejects_missing_case_and_wrong_group():
    group = 17
    init = {'pid': 30, 'ppid': 20, 'pgid': group, 'start_ticks': 1}
    worker = {'pid': 31, 'ppid': 30, 'pgid': group, 'start_ticks': 2}
    detached = {'pid': 32, 'ppid': 31, 'pgid': 32, 'start_ticks': 3}
    normal_receipt = {'reason': 'complete', 'status': 0, 'complete_descendant_wait': True,
        'worker_host_identity': {'host_pid': 31, 'start_ticks': 2},
        'namespace_init': {'host_pid': 30, 'start_ticks': 1},
        'init_cpu_seconds': .1, 'children_cpu_seconds': .3}
    cases = []
    for name in ('normal', 'timeout', 'term', 'kill'):
        cases.append({'case': name, 'launcher_rc': 0 if name == 'normal' else -15 if name == 'term' else -9,
            'outer_waited_cpu_seconds': .4, 'init': init, 'worker': worker, 'detached': detached,
            'all_absent': True, 'receipt': normal_receipt if name == 'normal' else
                {'reason': 'timeout'} if name == 'timeout' else None})
    record = {'scope': 'synthetic CPU namespace lifecycle only', 'campaign_pgid': group, 'cases': cases}
    proof.check_lifecycle(record, group)
    with pytest.raises(ValueError):
        proof.check_lifecycle({**record, 'cases': cases[:-1]}, group)
    with pytest.raises(ValueError):
        proof.check_lifecycle(record, group + 1)
