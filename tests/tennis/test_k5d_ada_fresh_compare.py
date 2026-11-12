import copy
import json

import pytest

from experiments.tennis import k5d_ada_fresh_compare as fresh


def test_repeat_checks_serialized_bytes_and_both_source_contracts(tmp_path, monkeypatch):
    point = {'native_bytes': 1, 'wire_bytes': 2,
             'baseline': {'native_bytes': 1, 'zero_ball_wire_bytes': 3},
             'receiver': {'wire_sha256': 'w', 'wire_bytes': 2, 'native_sha256': 'n',
                          'native_bytes': 1, 'decoded_yuv_sha256': 'd', 'rendered_rgb_sha256': 'r'}}
    report = {'frames': 16, 'conversion': {'yuv_sha256': 'y'}, 'points': [point]}
    roots = [tmp_path / name for name in ('a', 'b')]
    for root in roots:
        root.mkdir()
        (root / 'pilot.json').write_text(json.dumps(report))
        (root / 'source-qualification.json').write_text('{}')
    calls = []
    monkeypatch.setattr(fresh.cf, 'reference16', lambda _: (report, {}))
    monkeypatch.setattr(fresh.cost, 'compare_contract', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(fresh.cp, 'compare_equivalence', lambda *_: {'passed': True})
    assert fresh.compare(*roots, tmp_path)['passed']
    assert calls == [{'exact': False, 'capability': (8, 9)}] * 2
    changed = copy.deepcopy(report)
    changed['points'][0]['receiver']['wire_bytes'] += 1
    (roots[1] / 'pilot.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='receiver differs'):
        fresh.compare(*roots, tmp_path)


def test_cost_gate_enforces_complete_a_and_remaining_time(tmp_path, monkeypatch):
    import time
    from experiments.tennis import k5d_ada_fresh as worker

    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 200, 'timing_contaminated': False, 'contention': [], 'paused_seconds': 0}))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 1, 'phase_started': 10, 'budget_end': 800}}))
    receipt = {'reason': 'complete', 'status': 0, 'complete_descendant_wait': True,
               'namespace_init': {'namespace_pid': 1}, 'children_cpu_seconds': 195, 'init_cpu_seconds': 1}
    (tmp_path / 'command.log').write_text('K5D_NAMESPACE_CANDIDATE ' + json.dumps(receipt) + '\n')
    monkeypatch.setattr(time, 'monotonic', lambda: 250)
    assert worker.cost_gate(tmp_path)['b_forecast_seconds'] == 350
    monkeypatch.setattr(time, 'monotonic', lambda: 400)
    with pytest.raises(ValueError, match='remaining campaign'):
        worker.cost_gate(tmp_path)
    monkeypatch.setattr(time, 'monotonic', lambda: 250)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 301, 'timing_contaminated': False, 'contention': [], 'paused_seconds': 0}))
    with pytest.raises(ValueError, match='complete A cost'):
        worker.cost_gate(tmp_path)


def test_namespace_allows_only_fixed_fresh_worker(monkeypatch):
    from experiments.tennis import k5d_scoped_namespace as ns
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    assert ns.worker_command(['ada-fresh16', '--frames', '16'])[2:5] == [
        'experiments.tennis.k5d_ada_fresh', 'run', '--frames']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError):
        ns.worker_command(['ada-fresh16'])


def test_worker_rejects_missing_or_shared_ada_before_work(tmp_path, monkeypatch):
    from argparse import Namespace
    from experiments.tennis import k5d_ada_fresh as worker
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setattr(worker.cp, 'execution_identity', lambda: {})
    args = Namespace(frames=16, gpu_profile=None)
    with pytest.raises(ValueError, match='exclusive Ada'):
        worker.run(args)
    args = Namespace(frames=16, gpu_profile=worker.ADA_PROFILE, first_pair_bridge='b',
                     first_pair_bridge_sha256='s', calibration_source='c', consumer_source='d',
                     shared_numerical_only=True)
    with pytest.raises(ValueError, match='exclusive Ada'):
        worker.run(args)


def test_fresh_spec_uses_real_fleet_validator_and_separate_bridge(tmp_path, monkeypatch):
    import hashlib
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from experiments.jobs import inbox
    from experiments.tennis import k5d_ada_fresh as worker, k5d_ada_fresh_source_bridge as bridge
    from experiments.tennis import k5d_scoped_namespace as namespace

    cpu = tmp_path / 'cpu.json'
    cpu.write_text(json.dumps({'audited': True, 'passed': True, 'scope': 'native CPU PID containment and accounting',
                               'gpu_used': False, 'unshare_sha256': 'a' * 64,
                               'namespace_wrapper_sha256': hashlib.sha256(Path(namespace.__file__).read_bytes()).hexdigest(),
                               'consumer_execution_identity': {}}))
    remote = '/home/itec/emanuele/Datasets/pointstream-data/test/'
    assets = {name: {'local': str(cpu), 'remote': remote + name, 'sha256': 'c' * 64}
              for name in ('fidelity_gate', 'accepted16', 'native_cpu_receipt', 'first_pair_bridge',
                           'calibration_source', 'consumer_source')}
    assets['accepted16']['sha256'] = worker.cf.D8_SMOKE_SHA
    monkeypatch.setattr(bridge, 'verify_signed', lambda *_: None)
    monkeypatch.setattr(worker.cf, 'asset', lambda name, path, digest, _local, *, extract:
                        {'name': name, 'path': path, 'sha256': digest, 'extract': extract})
    base = json.loads(worker.cf.BASE.read_text())
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    spec = worker.spec(base, assets=assets, deadline=deadline, unshare_sha256='a' * 64, identity={})
    inbox.validate_spec(spec)
    assert spec['gpu_models'] == ['RTX 6000 Ada'] and spec['gpu_sharing'] == {'mode': 'exclusive'}
    assert spec['scale']['proof_mode'] == {'smoke': 'ada-fresh16', 'full': 'ada-fresh16'}
    assert spec['validator'][-1] == 'validate' and spec['validator'][-2] == 'experiments.tennis.k5d_ada_fresh'
    assert spec['budget_seconds'] == 1200 and spec['single_cpu_campaign'] is True
