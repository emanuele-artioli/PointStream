import pytest

from experiments.tennis import k5d_ada_scale_adapter as adapter


def test_producer_rejects_legacy_receipt_without_actual_denial_verifier(tmp_path, monkeypatch):
    (tmp_path / 'checkpoint.json').write_text('{}')
    (tmp_path / 'pilot.json').write_text('{}')
    marker = {'producer_job': '20261112T000000Z-test', 'producer_stage': 'smoke', 'frames': 16,
              'gpu': {'name': adapter.GPU_NAME, 'uuid': 'GPU-test'}}
    monkeypatch.setattr(adapter.cp, 'verify', lambda *a, **kw: None)
    monkeypatch.setattr(adapter.receiver, 'verify', lambda *a, **kw: marker)
    with pytest.raises(ValueError, match='denial verifier unavailable'):
        adapter.producer(tmp_path, {}, job=marker['producer_job'], stage='smoke', frames=16, config_sha256='a' * 64, config_path=tmp_path / 'config')
    with pytest.raises(ValueError, match='producer job'):
        adapter.producer(tmp_path, {}, job='20261112T000000Z-other', stage='smoke', frames=16, config_sha256='a' * 64, config_path=tmp_path / 'config')
    with pytest.raises(ValueError, match='paired 16/48'):
        adapter.producer(tmp_path, {}, job=marker['producer_job'], stage='full', frames=188, config_sha256='a' * 64, config_path=tmp_path / 'config')


def test_score16_uses_pinned_ada_archive_and_exact_comparison(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def unpack(path, digest):
        assert path == tmp_path / 'archive' and digest == adapter.REFERENCE_SHA256
        yield tmp_path / 'ada'

    monkeypatch.setattr(adapter.cost, 'unpack', unpack)
    monkeypatch.setattr(adapter, 'reference', lambda root: ({}, {}))
    monkeypatch.setattr(adapter.fresh, 'compare', lambda *args: {'paths': args})
    result = adapter.score16(tmp_path / 'archive', tmp_path / 'scored', tmp_path / 'accepted')
    assert result['paths'] == (tmp_path / 'ada/scored/assembled', tmp_path / 'scored', tmp_path / 'accepted')


def test_prepare16_accepts_exact_six_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import copy
    import json
    from contextlib import contextmanager

    points = [{'codec': codec, 'point': point, 'native_sha256': 'n', 'native_bytes': 2,
               'wire_sha256': 'w', 'wire_bytes': 3, 'encoder_decoded_sha256': 'd',
               'encoder_rendered_sha256': 'r', 'baseline': {'native_bytes': 2, 'native_sha256': 'n',
               'zero_ball_wire_bytes': 4, 'zero_ball_wire_sha256': 'z'}, 'streak': {'guard_off': False}}
              for codec, point in adapter.enc.POINTS]
    old = {'frames': 16, 'conversion': {'yuv_sha256': 'y'}, 'points': points}
    source = {'rows': [1], 'scores': {'x': 1}}
    (tmp_path / 'checkpoint.json').write_text('{}')
    (tmp_path / 'source-qualification.json').write_text(json.dumps(source))
    (tmp_path / 'pilot.json').write_text(json.dumps(old))

    @contextmanager
    def unpack(*_):
        yield tmp_path

    monkeypatch.setattr(adapter.cost, 'unpack', unpack)
    monkeypatch.setattr(adapter.cost, 'compare_contract', lambda *a, **k: None)
    monkeypatch.setattr(adapter.cp, 'verify', lambda *a, **k: None)
    monkeypatch.setattr(adapter, 'reference', lambda _: (old, source))
    adapter.prepare16(tmp_path / 'ref', tmp_path, {})
    changed = copy.deepcopy(old)
    changed['points'][2]['wire_bytes'] += 1
    (tmp_path / 'pilot.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='wire_bytes'):
        adapter.prepare16(tmp_path / 'ref', tmp_path, {})


def test_spec_builder_limits_stages_hosts_and_requires_producer_jobs(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone
    from experiments.jobs import inbox
    from experiments.tennis import k5d_ada_scale_worker as worker

    keys = ('bridge', 'receiver_isolation_config', 'native_cpu_receipt', 'denial_witness',
            'fidelity_gate', 'accepted16', 'ada_reference', 'prepare16', 'prepare48')
    assets = {name: {'local': str(tmp_path / name),
                     'remote': '/home/itec/emanuele/Datasets/pointstream-data/test/' + name,
                     'sha256': 'a' * 64} for name in keys}
    assets['accepted16']['sha256'] = worker.cf.D8_SMOKE_SHA
    assets['ada_reference']['sha256'] = adapter.REFERENCE_SHA256
    monkeypatch.setattr(worker, 'admission', lambda *_: None)
    monkeypatch.setattr(worker.cf, 'asset', lambda name, remote, digest, local, *, extract:
                        {'name': name, 'path': remote, 'sha256': digest, 'extract': extract})
    base = json.loads(worker.cf.BASE.read_text())
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    with pytest.raises(ValueError, match='producer jobs'):
        worker.spec(base, action='score', assets=assets, deadline=deadline, unshare_sha256='a' * 64, identity={})
    result = worker.spec(base, action='score', assets=assets, deadline=deadline, unshare_sha256='a' * 64,
                         identity={}, producer16_job='20261112T000001Z-a', producer48_job='20261112T000001Z-a')
    inbox.validate_spec(result)
    assert result['hosts'] == ['gpu6'] and result['gpu_sharing'] == {'mode': 'exclusive'}
    assert result['scale']['proof_mode'] == {'smoke': 'ada-scale-score', 'full': 'ada-scale-score'}
    assert result['budget_seconds'] == 900 and result['validator_seconds'] == 30
    assert result['smoke']['seconds'] == 300 and result['full']['seconds'] == 600


def test_producer_verifies_every_isolation_receipt_against_actual_context(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    job = '20261112T000000Z-a'
    (tmp_path / 'checkpoint.json').write_text('{}')
    points = [{'codec': c, 'point': n} for c, n in adapter.enc.POINTS]
    (tmp_path / 'pilot.json').write_text(json.dumps({'points': points}))
    for point in points:
        path = tmp_path / f"{point['codec']}-{point['point']}" / 'receiver-evidence'
        path.mkdir(parents=True)
        (path / 'receiver.json').write_text(json.dumps({'key': [point['codec'], point['point']]}))
    marker = {'producer_job': job, 'producer_stage': 'smoke', 'frames': 16,
              'gpu': {'name': adapter.GPU_NAME, 'uuid': 'GPU-test'}}
    monkeypatch.setattr(adapter.cp, 'verify', lambda *a, **kw: None)
    monkeypatch.setattr(adapter.receiver, 'verify', lambda *a, **kw: marker)
    seen = []
    def verify(receipt, *, expected_context, expected_config_sha256):
        seen.append((receipt['key'], expected_context, expected_config_sha256))
        return {'profile_sha256': adapter.hashlib.sha256(json.dumps({'runtime_files': {}}, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'runtime_inventory_sha256': adapter.hashlib.sha256(b'{}').hexdigest(), 'actual_policy_sha256': 'f' * 64}
    monkeypatch.setitem(sys.modules, 'experiments.tennis.k5d_isolated_receiver', SimpleNamespace(verify_receipt=verify))
    config_path = tmp_path / 'config'
    config_path.write_text(json.dumps({'profiles': {'gpu6': {'runtime_files': {}}}}))
    config_sha = adapter.oracle.sha(config_path)
    identity = {'source_sha256': 's'}
    adapter.producer(tmp_path, identity, job=job, stage='smoke', frames=16, config_sha256=config_sha, config_path=config_path)
    assert len(seen) == 6 and all(x[1] == {'job_id': job, 'stage': 'smoke', 'execution_identity': identity,
                                           'gpu': {**marker['gpu'], 'capability': [8, 9]}} for x in seen)
    assert all(x[2] == config_sha for x in seen)


def test_admission_requires_actual_root_approved_witness_and_matching_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from argparse import Namespace
    from pathlib import Path
    from types import SimpleNamespace
    from experiments.tennis import k5d_ada_scale_worker as worker, k5d_scoped_namespace as namespace

    identity = {'source_sha256': 's', 'snapshot_sha256': 'n', 'environment_sha256': 'e'}
    context = {'job_id': '20261112T000001Z-witness', 'stage': 'smoke', 'execution_identity': identity,
               'gpu': {'uuid': 'GPU-test', 'name': adapter.GPU_NAME, 'capability': [8, 9]}}
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    config_path, config_sha = write('config', {'schema': 1, 'execution_identity': identity, 'profiles': {'gpu6': {'runtime_files': {}}}})
    bridge_path, bridge_sha = write('bridge', {'approved': True, 'consumer_execution_identity': identity,
        'calibration_execution_identity': adapter.REFERENCE_IDENTITY, 'consumer_snapshot_sha256': 'n'})
    cpu_path, cpu_sha = write('cpu', {'schema': 1, 'audited': True, 'passed': True, 'gpu_used': False,
        'scope': 'native CPU PID containment and accounting', 'consumer_execution_identity': identity,
        'namespace_wrapper_sha256': hashlib.sha256(Path(namespace.__file__).read_bytes()).hexdigest(),
        'unshare_sha256': 'a' * 64, 'job_id': '20261112T000001Z-cpu', 'all_owned_pids_absent': True,
        'independent_cpu_match': True, 'cases': ['normal', 'timeout', 'sigterm', 'sigkill']})
    witness = {'approved': True, 'actual_filesystem_denial_passed': True, 'host': 'gpu6', 'codecs': ['dcvc', 'svt'],
        'profile_sha256': worker.canonical_sha({'runtime_files': {}}), 'actual_policy_sha256': 'f' * 64,
        'runtime_inventory_sha256': worker.canonical_sha({}), 'consumer_execution_identity': identity,
        'config_sha256': config_sha, 'producer_context': context, 'receipt': {'proof': True}}
    witness_path, witness_sha = write('witness', witness)
    calls = []
    def verify(receipt, *, expected_context, expected_config_sha256):
        calls.append((receipt, expected_context, expected_config_sha256))
        if receipt.get('proof') is not True:
            raise ValueError('legacy packet-only receipt')
        return {'landlock_abi': 3, 'handled_mask': 42,
                'profile_sha256': worker.canonical_sha({'runtime_files': {}}), 'actual_policy_sha256': 'f' * 64,
                'runtime_inventory_sha256': worker.canonical_sha({})}
    monkeypatch.setitem(sys.modules, 'experiments.tennis.k5d_isolated_receiver', SimpleNamespace(verify_receipt=verify))
    args = Namespace(bridge=bridge_path, bridge_sha256=bridge_sha, receiver_isolation_config=config_path,
        receiver_isolation_config_sha256=config_sha, native_cpu_receipt=cpu_path, native_cpu_receipt_sha256=cpu_sha,
        denial_witness=witness_path, denial_witness_sha256=witness_sha, unshare_sha256='a' * 64)
    worker.admission(args, identity)
    assert calls == [(witness['receipt'], context, config_sha)]
    witness['actual_filesystem_denial_passed'] = False
    args.denial_witness, args.denial_witness_sha256 = write('witness', witness)
    with pytest.raises(ValueError, match='denial witness'):
        worker.admission(args, identity)


def test_namespace_uses_fixed_ada_worker_and_refuses_cpu_dispatch(monkeypatch):
    from experiments.tennis import k5d_scoped_namespace as namespace
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-example')
    assert namespace.worker_command(['ada-scale-prepare', '--frames', '16'])[2:5] == [
        'experiments.tennis.k5d_ada_scale_worker', 'prepare', '--frames']
    assert namespace.worker_command(['ada-scale-score', '--frames', '48'])[2:5] == [
        'experiments.tennis.k5d_ada_scale_worker', 'score', '--frames']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError):
        namespace.worker_command(['ada-scale-score'])
