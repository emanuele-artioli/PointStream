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


def test_actual_fleet_gate_accepts_cpu_proof_validation_shape(tmp_path, monkeypatch):
    import json
    from experiments.jobs import inbox, monitor

    root = tmp_path / 'jobs' / 'fleet' / 'inbox' / '20261109T000000Z-12345678'
    source = root / 'source'
    source.mkdir(parents=True)
    (root / 'run').mkdir()
    (source / 'work.py').write_text('pass\n')
    repo = str(proof.Path(proof.__file__).resolve().parents[2])
    (source / 'validate.py').write_text(
        'import json,os,pathlib,sys\n'
        f'sys.path.insert(0,{repo!r})\n'
        'from experiments.tennis.w8_witness_cpu_proof_job import validation_receipt\n'
        "pathlib.Path(os.environ['PS_VALIDATION_PATH']).write_text(json.dumps(validation_receipt()))\n")
    asset = tmp_path / 'input.json'
    asset.write_text('{}')
    spec = {'schema': 1, 'hosts': ['gpu1'], 'gpu_models': [], 'gpu_memory_mib': 100, 'cpu_threads': 1,
            'entrypoint': ['work.py'], 'arguments': ['--phase', '{phase}'],
            'scale': {'phase': {'smoke': 'smoke', 'full': 'full'}},
            'inputs': [{'path': str(asset), 'sha256': inbox.file_digest(asset)}],
            'smoke': {'seconds': 10, 'representative_basis': 'fleet validator contract'},
            'full': {'seconds': 10}, 'budget_seconds': 60,
            'deadline': (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
            'validator': ['{python}', 'validate.py'], 'validator_seconds': 10, 'required_commands': []}
    normalized = inbox.validate_spec(spec)
    monitor.write_json(root / 'spec.json', normalized)
    monitor.write_json(root / 'ready.json', {'spec_sha256': inbox.digest(normalized),
                                             'source_sha256': inbox.source_identity(source)})
    monkeypatch.setenv('PS_JOB_DIR', str(root / 'run'))
    assert inbox.campaign(root) == 0
    gate = json.loads((root / 'gate.json').read_text())
    assert gate['passed'] is True and gate['validation']['checks']['native_yuv_and_teardown'] is True
    assert gate['validation']['quality_claim'] is False


def test_protected_receipt_never_claims_fresh_content_verification():
    historical = {'protected_inputs': {}}
    receipt = {'schema': 1, 'scope': 'witness protected metadata only',
               'historical_receipt_sha256': proof.packet.IDENTITY_SHA, 'content_verified_now': True}
    with pytest.raises(ValueError, match='stat-only'):
        proof.check_protected_metadata(receipt, historical)
    receipt['content_verified_now'] = False
    receipt['historical_receipt_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='stat-only'):
        proof.check_protected_metadata(receipt, historical)


def test_protected_metadata_requires_historical_pin_and_current_stat_check(monkeypatch):
    from experiments.tennis import w8_protected_metadata

    observed = []
    monkeypatch.setattr(w8_protected_metadata, 'validate',
        lambda receipt, historical=None, *, check_current=False: observed.append((receipt, historical, check_current)))
    historical = {'protected_inputs': {}}
    receipt = {'schema': 1, 'scope': 'witness protected metadata only', 'content_verified_now': False,
               'historical_receipt_sha256': proof.packet.IDENTITY_SHA}
    proof.check_protected_metadata(receipt, historical)
    assert observed == [(receipt, historical, True)]


def test_retained_packet_cpu_receipt_joins_pinned_main_and_rejects_tampering(monkeypatch):
    from experiments.tennis import w8_protected_metadata
    calls = []
    def read(path, sha):
        calls.append((str(path), sha))
        return {'receipts': {'main': {'torch': 'pinned-torch'}}}
    monkeypatch.setattr(w8_protected_metadata, 'read_historical', read)
    spec = {'staged_inputs': [{'name': 'preflight', 'path': '/public/bundle.json', 'sha256': proof.packet.PREFLIGHT_SHA}]}
    valid = {'scope': 'CPU packet decoder only', 'cuda_visible_devices': '', 'cuda_initialized_after_decode': False,
             'torch_version': 'pinned-torch', 'pre_serialization': {'state_tensors': 480, 'restored_tensors': 480, 'unit_arrays': 2}}
    proof.check_retained_packet_cpu(spec, valid)
    assert calls == [('/public/bundle.json', proof.packet.PREFLIGHT_SHA)]
    bad = dict(valid, pre_serialization=dict(valid['pre_serialization'], restored_tensors=479))
    with pytest.raises(ValueError, match='pre-serialization'):
        proof.check_retained_packet_cpu(spec, bad)
    with pytest.raises(ValueError, match='pre-serialization'):
        proof.check_retained_packet_cpu(spec, dict(valid, torch_version='other'))
    with pytest.raises(ValueError, match='preflight'):
        proof.check_retained_packet_cpu({'staged_inputs': []}, valid)
