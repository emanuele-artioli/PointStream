from typing import Any
from pathlib import Path
import os

import pytest

from experiments.tennis import k5d_control_study as control


def receipt() -> dict:
    costs = {'prepare': {'seconds16': 168.21834516525269, 'seconds48': 234.93895959854126,
                         'margin': 1.5, 'reserve_seconds': 20},
             'score': {'seconds16': 110.028817653656, 'seconds48': 201.41556024551392,
                       'margin': 1.5, 'reserve_seconds': 20}}
    programme = {kind: {'readiness_seconds': 20, 'stage_staging_seconds': 1, 'validator_seconds': 1,
                        'audit_seconds': 1, 'failed_start_reserve_seconds': 1,
                        'ball_remaining_seconds': 100000, 'global_remaining_seconds': 100000}
                 for kind in ('gpu', 'cpu')}
    return {'budget_reconciled': True, 'cost': costs, 'programme': programme,
            'calibration_archives': control.CALIBRATION_ARCHIVES, 'calibration_audit_sha256': control.CALIBRATION_AUDIT_SHA,
            'score_timing_sha256': control.CALIBRATION_TIMING,
            'calibration_execution_sha256': {k: v[0] for k, v in control.CALIBRATION_EXECUTIONS.items()}}


def test_real_calibration_projection_fails_current_ball_allocation() -> None:
    data = receipt()
    result = control.programme_projection(data)
    assert result['campaigns'] == 60 and result['gpu_smoke_inferences'] == 0
    assert result['full_frames_per_phase'] == 5675 and result['max_full_seconds'] < 2670
    assert result['gpu']['incremental_seconds'] > 51022
    data['programme']['gpu']['ball_remaining_seconds'] = 39860
    data['programme']['cpu']['ball_remaining_seconds'] = 44340
    result = control.programme_projection(data)
    assert result['gpu']['fits'] is False and result['cpu']['fits'] is False


def test_understated_or_unbound_calibration_is_rejected() -> None:
    data = receipt()
    data['cost']['score']['seconds48'] -= 1
    with pytest.raises(ValueError, match='understates'):
        control.programme_projection(data)
    data = receipt()
    data['calibration_execution_sha256']['score16'] = '0' * 64
    with pytest.raises(ValueError, match='execution differs'):
        control.programme_projection(data)
    data = receipt()
    data['programme']['gpu']['readiness_seconds'] = 10
    with pytest.raises(ValueError, match='readiness'):
        control.programme_projection(data)


def test_budget_cannot_overstate_remaining_or_ignore_active_reservations() -> None:
    ledger = {'gpu_total_seconds': 1000, 'cpu_total_seconds': 1200,
              'workstreams': {'ball': {'gpu_limit_seconds': 800, 'cpu_limit_seconds': 900}},
              'charges': [{'workstream': 'ball', 'gpu_seconds': 100, 'cpu_seconds': 200}]}
    attestation = {'reviewed': True, 'ledger_sha256': 'a' * 64, 'active_reservations_reconciled': True,
                   'gpu': {'ball_charged_seconds': 100, 'global_charged_seconds': 100, 'ball_limit_seconds': 800,
                           'spent_and_reserved_seconds': 150, 'global_remaining_seconds': 850},
                   'cpu': {'ball_charged_seconds': 200, 'global_charged_seconds': 200, 'ball_limit_seconds': 900,
                           'spent_and_reserved_seconds': 250, 'global_remaining_seconds': 900}}
    result = receipt()
    result.update(budget_ledger_sha256='a' * 64, budget_attestation_sha256='b' * 64)
    result['programme']['gpu'].update(ball_remaining_seconds=650, global_remaining_seconds=850)
    result['programme']['cpu'].update(ball_remaining_seconds=650, global_remaining_seconds=900)
    control.check_budget(ledger, attestation, 'a' * 64, 'b' * 64, result)
    result['programme']['gpu']['ball_remaining_seconds'] = 651
    with pytest.raises(ValueError, match='reconciled ledger'):
        control.check_budget(ledger, attestation, 'a' * 64, 'b' * 64, result)


def test_cpu_fit_is_never_inferred_from_wall_and_nan_margin_rejected() -> None:
    data = receipt()
    data['programme']['cpu']['ball_remaining_seconds'] = 1e9
    data['programme']['cpu']['global_remaining_seconds'] = 1e9
    result = control.programme_projection(data)
    assert result['cpu']['fits'] is False
    assert result['cpu']['incremental_seconds'] is None
    data['cost']['score']['margin'] = float('nan')
    with pytest.raises(ValueError, match='calibration bounds'):
        control.programme_projection(data)


def test_cpu_sample_alone_cannot_authorize_unverified_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.tennis import k5d_study_wave as wave

    data = receipt()
    data['consumer_execution_identity'] = {}
    data['cpu_calibration'] = {'unverified': 'test'}
    monkeypatch.setattr(wave, 'verified_cpu_calibration', lambda *_: {
        phase: {'margin': 1.5, 'fixed_seconds': 1, 'marginal_seconds_per_frame': 0.1}
        for phase in ('prepare', 'score')})
    result = control.programme_projection(data)
    assert result['cpu']['incremental_seconds'] is not None
    assert result['cpu']['fits'] is False


def test_confirmation_caps_are_separate_from_full_study() -> None:
    data = receipt()
    clips = ['tracknet/game9/Clip5', 'tracknet/game10/Clip12']
    gpu = dict(zip((f'{c}:{p}' for c in clips for p in ('prepare', 'score')),
                   (1200, 1400, 2100, 2700), strict=True))
    cpu = {name: 2000 for name in gpu}
    gate = {'allowed_clips': clips, 'confirmation_caps': {
        'gpu': {'jobs': gpu, 'external': 200, 'total': 7600},
        'cpu': {'jobs': cpu, 'external': 200, 'total': 8200}}}
    control.confirmation_budget(gate, data)
    assert control.programme_projection(data)['cpu']['fits'] is False
    gpu[next(iter(gpu))] = 1
    with pytest.raises(ValueError, match='below numerical forecast'):
        control.confirmation_budget(gate, data)
    gpu[next(iter(gpu))] = 1200
    gate['allowed_clips'] = list(reversed(clips))
    with pytest.raises(ValueError, match='registered clips'):
        control.confirmation_budget(gate, data)


def test_confirmation_requires_both_audited_receipts_without_full30_fit(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.tennis import k5d_full_study as study

    data = receipt()
    data.update(schema=1, passed=True, selection_sha256=control.fleet.SOURCE_SHA,
                calibration_execution_identity={'old': 'old'}, consumer_execution_identity={'new': 'new'},
                bridge_sha256='b' * 64, gpu_model='RTX A6000', gpu_capability=[8, 6],
                calibration_jobs={'prepare': '20261105T025241Z-fe451f9e', 'score': '20261105T032433Z-1423f92a'},
                stage_caps_fit=True)
    clips = ['tracknet/game9/Clip5', 'tracknet/game10/Clip12']
    jobs = [f'{c}:{p}' for c in clips for p in ('prepare', 'score')]
    qualification = {'audited': True, 'cpu_native_pid_passed': True, 'gpu_16_parity_passed': True,
                     'consumer_execution_identity': {'new': 'new'}, 'gpu_model': 'RTX A6000',
                     'gpu_capability': [8, 6], 'namespace_wrapper_sha256': 'w' * 64,
                     'native_cpu_receipt_sha256': 'a' * 64, 'gpu_parity_receipt_sha256': 'c' * 64,
                     'unshare_sha256': 'd' * 64}
    gate = {'schema': 1, 'approved': True, 'purpose': 'full-retrospective-benchmark-cpu-readiness',
            'selection_sha256': control.fleet.SOURCE_SHA, 'consumer_execution_identity': {'new': 'new'},
            'cost_receipt_sha256': 'e' * 64, 'bridge_sha256': 'b' * 64, 'gpu_models': ['RTX A6000'],
            'gpu_capability': [8, 6], 'fidelity_only_for_unqualified': True, 'allowed_clips': clips,
            'release': 'cost-confirmation', 'confirmation_qualification': qualification,
            'confirmation_caps': {'gpu': {'jobs': dict(zip(jobs, (1200, 1400, 2100, 2700), strict=True)),
                                          'external': 200, 'total': 7600},
                                  'cpu': {'jobs': dict.fromkeys(jobs, 2000), 'external': 200, 'total': 8200}}}
    bridge = {'approved': True, 'calibration_execution_identity': {'old': 'old'},
              'consumer_execution_identity': {'new': 'new'}, 'calibration_snapshot_sha256': 'x',
              'consumer_snapshot_sha256': 'y', 'changed_paths': [], 'unchanged_files': 1}
    cpu = {'schema': 1, 'audited': True, 'passed': True, 'scope': 'native CPU PID containment and accounting', 'gpu_used': False,
           'namespace_wrapper_sha256': 'w' * 64, 'unshare_sha256': 'd' * 64,
           'consumer_execution_identity': {'new': 'new'}, 'job_id': '20261105T000000Z-12345678',
           'cases': ['normal', 'timeout', 'sigterm', 'sigkill'], 'all_owned_pids_absent': True,
           'independent_cpu_match': True}
    parity = {'schema': 1, 'audited': True, 'passed': True, 'scope': 'actual 16-frame GPU native and score parity',
              'consumer_execution_identity': {'new': 'new'}, 'job_id': '20261105T000000Z-abcdef12',
              'gpu_model': 'RTX A6000', 'gpu_capability': [8, 6], 'all_six_exact': True, 'namespace_wrapper_sha256': 'w' * 64,
              'unshare_sha256': 'd' * 64}
    objects = {'gate': gate, 'cost': data, 'bridge': bridge, 'ledger': {}, 'budget': {}, 'cpu': cpu, 'parity': parity}
    monkeypatch.setattr(study, 'checked_json', lambda path, _: objects[path.name])
    monkeypatch.setattr(study, 'selection', lambda: {clips[0]: {'label_rows': 292}, clips[1]: {'label_rows': 586}})
    monkeypatch.setattr(control.bridge, 'bridge', lambda *_: bridge)
    monkeypatch.setattr(control.evidence, 'historical', lambda *_: None)
    monkeypatch.setattr(control, 'check_budget', lambda *_: None)
    monkeypatch.setattr(control, 'programme_projection', lambda *_, **_kwargs: {'gpu': {'fits': False}, 'cpu': {'fits': False}})
    monkeypatch.setattr(control.oracle, 'sha', lambda *_: 'w' * 64)
    args = (tmp_path / 'gate', 'g' * 64, tmp_path / 'cost', 'e' * 64, tmp_path / 'bridge', 'b' * 64,
            tmp_path / 'old', tmp_path / 'new', tmp_path / 'ledger', 'l' * 64, tmp_path / 'budget', 't' * 64,
            {'new': 'new'}, clips[0], 'prepare')
    with pytest.raises(ValueError, match='qualification receipts'):
        control.approved(*args)
    paths = {'native_cpu': tmp_path / 'cpu', 'gpu_parity': tmp_path / 'parity'}
    _, frames = control.approved(*args, paths)
    assert frames == 292
    parity['all_six_exact'] = False
    with pytest.raises(ValueError, match='qualification receipt'):
        control.approved(*args, paths)
    parity['all_six_exact'] = True
    gate['release'] = control.PAIR_RELEASE
    gate['allowed_clips'] = [clips[0]]
    gate['future_study_fit_proven'] = False
    gate['confirmation_caps'] = {resource: {'jobs': {f'{clips[0]}:prepare': 1450, f'{clips[0]}:score': 1700},
                                            'external': {'prepare': 100, 'score': 100}, 'total': 3350}
                                 for resource in ('gpu', 'cpu')}
    bridge['bridge_type'] = 'parent8-resource-v1'
    monkeypatch.setattr(control, 'parent8_bridge_record', lambda *_: {k: v for k, v in bridge.items() if k != 'approved'})
    parity.update(exclusive=True, timing_contaminated=False, source_bridge_sha256='b' * 64)
    _, frames = control.approved(*args, paths)
    assert frames == 292
    parity['timing_contaminated'] = True
    with pytest.raises(ValueError, match='clean exclusive'):
        control.approved(*args, paths)
    gate['release'] = 'full-study'
    data['budget_fits'] = {'gpu': True, 'cpu': True}
    with pytest.raises(ValueError, match='compact certificate'):
        control.approved(*args, paths)


def test_confirmation_spec_uses_namespace_and_enforces_job_caps(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.jobs import inbox

    clip = 'tracknet/game9/Clip5'
    gate = {'release': 'cost-confirmation', 'confirmation_qualification': {'unshare_sha256': 'a' * 64},
            'confirmation_caps': {resource: {'jobs': {f'{clip}:prepare': 500}}
                                  for resource in ('gpu', 'cpu')}}
    names = ('full_gate', 'cost_receipt', 'bridge', 'calibration_source', 'consumer_source', 'fidelity_gate',
             'budget_ledger', 'budget_attestation', 'native_cpu_receipt', 'gpu_parity_receipt')
    assets = {name: {'local': str(tmp_path / name), 'remote': '/remote/' + name, 'sha256': 'a' * 64} for name in names}
    monkeypatch.setattr(control.study, 'checked_json', lambda *_: gate)
    monkeypatch.setattr(control, 'approved', lambda *_: ({'cost': {'prepare': {'seconds16': 1}}}, 292))
    monkeypatch.setattr(control.study, 'forecast', lambda *_: 1)
    monkeypatch.setattr(control.cf, 'asset', lambda name, *_args, **_kwargs: {'name': name})
    monkeypatch.setattr(control.cf, 'common_spec', lambda *_: {'scale': {}})
    monkeypatch.setattr(inbox, 'validate_spec', lambda *_: None)
    base: dict = {'arguments': [], 'device': 'gpu', 'cpu_threads': 1, 'single_cpu_campaign': True,
                  'gpu_sharing': {'mode': 'exclusive'}, 'contention': {'policy': 'stop', 'resume_attempts': 0}}
    result = control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                          deadline='later', readiness_seconds=20, full_seconds=100, campaign_seconds=150)
    assert result['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace']
    assert result['arguments'][:5] == ['outer', '--unshare-sha', 'a' * 64, '--timeout', '{namespace_timeout}']
    assert '--native-cpu-receipt' in result['arguments'] and '--gpu-parity-receipt' in result['arguments']
    with pytest.raises(ValueError, match='job cap'):
        control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                     deadline='later', readiness_seconds=20, full_seconds=490, campaign_seconds=550)


def test_qualified_projection_excludes_verified_complete_stages_and_rejects_understatement(monkeypatch):
    from experiments.tennis import k5d_study_wave as wave

    data = receipt()
    data['consumer_execution_identity'] = {'source': 'a'}
    rows = control.study.selection()
    inventory = {clip: {phase: {'state': 'pending'} for phase in ('prepare', 'score')} for clip in rows}
    samples = {'prepare': [], 'score': []}
    for index, clip in enumerate(wave.CONFIRMATIONS):
        for phase in ('prepare', 'score'):
            entry = {'state': 'complete', 'job_id': f'20261105T00000{index}Z-{phase[0] * 8}',
                     'receipt_sha256': 'a' * 64, 'publications': {}, 'independent_audit_sha256': 'b' * 64}
            inventory[clip][phase] = entry
            samples[phase].append({'clip': clip, 'entry': entry.copy()})
    data['cpu_calibration'] = samples
    monkeypatch.setattr(wave, 'verified_cpu_calibration', lambda *_: {
        phase: {'margin': 1.5, 'fixed_seconds': 1, 'marginal_seconds_per_frame': 0.1}
        for phase in ('prepare', 'score')})
    noncommand = {'groups': {'whole': {'gpu': 0.5, 'cpu': 0.5}},
                  'command_covered': ['isolation'], 'failure_reserve': 'one-bounded-campaign'}
    for value in data['programme'].values():
        value.update(noncommand_groups={'whole': 1}, failed_start_reserve_seconds=3000)
    result = control.programme_projection(data, inventory=inventory, noncommand=noncommand)
    assert result['campaigns'] == 56 and result['cpu']['fits'] is True
    data['programme']['cpu']['failed_start_reserve_seconds'] = 1
    with pytest.raises(ValueError, match='failure reserve'):
        control.programme_projection(data, inventory=inventory, noncommand=noncommand)
    data['programme']['cpu']['failed_start_reserve_seconds'] = 3000
    data['programme']['cpu']['noncommand_groups']['whole'] = 0.1
    with pytest.raises(ValueError, match='understates'):
        control.programme_projection(data, inventory=inventory, noncommand=noncommand)
    data['programme']['cpu']['noncommand_groups']['whole'] = 1
    inventory[wave.CONFIRMATIONS[0]]['prepare']['state'] = 'unknown'
    with pytest.raises(ValueError, match='unknown or unfinished'):
        control.programme_projection(data, inventory=inventory, noncommand=noncommand)


def test_full_study_spec_routes_through_qualified_namespace(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from experiments.jobs import inbox
    from experiments.tennis import k5d_full_admission as admission

    clip = 'tracknet/game9/Clip5'
    gate = {'release': 'full-study', 'confirmation_qualification': {'unshare_sha256': 'a' * 64},
            'certificate_validation_reserve_seconds': 2, 'namespace_bootstrap_reserve_seconds': 5,
            'certificate_sha256': 'b' * 64}
    names = ('full_gate', 'cost_receipt', 'fidelity_gate', 'budget_ledger', 'budget_attestation',
             'native_cpu_receipt', 'gpu_parity_receipt', 'certificate', 'certificate_microbench', 'certificate_microbench_audit')
    assets = {name: {'local': str(tmp_path / name), 'remote': '/remote/' + name, 'sha256': 'a' * 64} for name in names}
    monkeypatch.setattr(control.study, 'checked_json', lambda path, *_: (
        {'pair_cost_bound_sha256': 'a' * 64} if path.name == 'fidelity_gate' else gate))
    monkeypatch.setattr(control.benchmark.encoder, 'fidelity_gate', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(control, 'approved', lambda *_: ({'cost': {'prepare': {'seconds16': 1}},
        'programme': {r: {'failed_start_reserve_seconds': 500, 'noncommand_groups': {'whole': 5}}
                      for r in ('gpu', 'cpu')}}, 292))
    monkeypatch.setattr(control.study, 'forecast', lambda *_: 1)
    monkeypatch.setattr(control.cf, 'asset', lambda name, *_args, **_kwargs: {'name': name})
    monkeypatch.setattr(control.cf, 'common_spec', lambda *_: {'scale': {}})
    monkeypatch.setattr(inbox, 'validate_spec', lambda *_: None)
    future = datetime.now(timezone.utc)
    deadline = (future + timedelta(minutes=20)).isoformat()
    monkeypatch.setattr(admission, 'read_certificate', lambda *_: {'valid_until': (future + timedelta(minutes=30)).isoformat()})
    base = {'arguments': [], 'device': 'gpu', 'cpu_threads': 1, 'single_cpu_campaign': True,
            'gpu_sharing': {'mode': 'exclusive'}, 'contention': {'policy': 'stop', 'resume_attempts': 0}}
    result = control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                          deadline=deadline, readiness_seconds=20, full_seconds=100, campaign_seconds=150)
    assert result['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace']
    for option in ('--native-cpu-receipt', '--gpu-parity-receipt', '--certificate', '--certificate-microbench', '--certificate-microbench-audit'):
        assert option in result['arguments']
    monkeypatch.setattr(control.study, 'forecast', lambda *_: 88)
    with pytest.raises(ValueError, match='actual namespace timeout'):
        control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                     deadline=deadline, readiness_seconds=20, full_seconds=100, campaign_seconds=150)
    with pytest.raises(ValueError, match='invalid readiness/full campaign bounds'):
        control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                     deadline=deadline, readiness_seconds=20, full_seconds=100, campaign_seconds=2701)


def test_budget_reconciles_active_reservations_and_parent_charges_once():
    ledger = {'gpu_total_seconds': 1000, 'cpu_total_seconds': 1200,
              'workstreams': {'ball': {'gpu_limit_seconds': 800, 'cpu_limit_seconds': 900}},
              'charges': [{'phase': 'partial', 'parent_phase': 'active', 'workstream': 'ball',
                           'gpu_seconds': 100, 'cpu_seconds': 200}],
              'reservation_amount_basis': 'remaining_unspent',
              'reservations': [{'phase': 'active', 'workstream': 'ball', 'status': 'running',
                                'gpu_seconds': 200, 'cpu_seconds': 200}]}
    attestation = {'reviewed': True, 'ledger_sha256': 'a' * 64, 'active_reservations_reconciled': True,
                   'active_reservations': {'active': {'workstream': 'ball', 'gpu': 200, 'cpu': 200}},
                   'gpu': {'ball_charged_seconds': 100, 'global_charged_seconds': 100, 'ball_limit_seconds': 800,
                           'spent_and_reserved_seconds': 300, 'global_remaining_seconds': 700},
                   'cpu': {'ball_charged_seconds': 200, 'global_charged_seconds': 200, 'ball_limit_seconds': 900,
                           'spent_and_reserved_seconds': 400, 'global_remaining_seconds': 800}}
    data = receipt()
    data.update(budget_ledger_sha256='a' * 64, budget_attestation_sha256='b' * 64)
    data['programme']['gpu'].update(ball_remaining_seconds=500, global_remaining_seconds=700)
    data['programme']['cpu'].update(ball_remaining_seconds=500, global_remaining_seconds=800)
    control.check_budget(ledger, attestation, 'a' * 64, 'b' * 64, data)
    attestation['active_reservations']['active']['gpu'] = 0
    with pytest.raises(ValueError, match='active reservation attestation'):
        control.check_budget(ledger, attestation, 'a' * 64, 'b' * 64, data)


def test_main_accounted_child_passes_normalized_arguments_to_worker(monkeypatch):
    import sys

    args = ['prepare', '--frames', '292']
    monkeypatch.setattr(sys, 'argv', ['k5d_control_study', '--accounted-child', *args])
    monkeypatch.setattr('argparse.ArgumentParser.parse_args', lambda _self: type('Args', (), {
        'action': 'prepare', 'frames': 292, 'checkpoint': None, 'checkpoint_sha256': None})())
    called = []
    def actual_worker(_args):
        control.validate_namespace_child_argv(sys.argv[1:], args)
        called.append(True)
    monkeypatch.setattr(control, 'run', actual_worker)
    control.main()
    assert called == [True]
    with pytest.raises(ValueError, match='arguments differ'):
        control.validate_namespace_child_argv(['prepare', '--frames', '586'], args)


def test_namespace_parent_must_be_exact_init_module_and_arguments():
    import sys

    expected = [sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'outer',
                '--timeout', '60', '--', 'prepare', '--frames', '292']
    tail = ['prepare', '--frames', '292']
    parent = [sys.executable, '-m', 'experiments.tennis.k5d_scoped_namespace', 'inner',
              '--outside', 'pid:[111]', '--timeout', '60.0', '--', *tail]
    control.validate_namespace_parent(parent, [['4321', '1']], 'pid:[222]', 'pid:[222]', expected, tail)
    fake = parent.copy()
    fake[2] = 'experiments.tennis.k5d_other_init'
    with pytest.raises(ValueError, match='namespace init'):
        control.validate_namespace_parent(fake, [['4321', '1']], 'pid:[222]', 'pid:[222]', expected, tail)
    with pytest.raises(ValueError, match='namespace init'):
        control.validate_namespace_parent(parent, [['4321', '1']], 'pid:[222]', 'pid:[333]', expected, tail)


def test_effective_namespace_timeout_includes_observed_readiness():
    control.check_full_timeout(80, 6, 90)
    with pytest.raises(ValueError, match='namespace timeout'):
        control.check_full_timeout(82, 6, 90)
    with pytest.raises(ValueError, match='namespace timeout'):
        control.check_full_timeout(80, float('nan'), 90)


def test_compact_certificate_pins_target_qualification_and_recomputes_budget(tmp_path, monkeypatch):
    import hashlib
    import json
    from datetime import datetime, timedelta, timezone
    from experiments.tennis import k5d_full_admission as admission

    identity = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'environment_sha256': 'c' * 64}
    selected = {'tracknet/game9/Clip5': {'label_rows': 292}, 'tracknet/game10/Clip12': {'label_rows': 586},
                'tracknet/game8/Clip1': {'label_rows': 100}}
    monkeypatch.setattr(admission.study, 'selection', lambda: selected)
    forecast = {'gpu': {'fits': True, 'incremental_seconds': 10}, 'cpu': {'fits': True, 'incremental_seconds': 10}}
    monkeypatch.setattr(control, 'programme_projection', lambda *_args, **_kwargs: forecast)
    native, parity = tmp_path / 'native.json', tmp_path / 'parity.json'
    native.write_text('{}')
    parity.write_text(json.dumps({'namespace_bootstrap_seconds': 1, 'namespace_bootstrap_audited': True}))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    q = {'native_cpu_receipt_sha256': sha(native), 'gpu_parity_receipt_sha256': sha(parity)}
    def entry(job):
        return {'job_id': job, 'receipt_sha256': 'e' * 64, 'independent_audit_sha256': 'f' * 64,
                'publications': {'smoke': {'sha256': '1' * 64}, 'full': {'sha256': '2' * 64}}}
    completed = {c: {'prepare': entry('20261105T000000Z-11111111'),
                     'score': entry('20261105T000001Z-22222222')} for c in list(selected)[:2]}
    now = datetime.now(timezone.utc)
    receipt = dict.fromkeys(admission.COST_CORE_KEYS, 'immutable')
    cert = {'schema': 1, 'scope': 'server-side full raw admission audit; root must review exact SHA',
            'selection_sha256': admission.fleet.SOURCE_SHA, 'execution_identity': identity,
            'cost_sha256': '3' * 64, 'cost_core_sha256': admission.cost_core_sha256(receipt),
            'ledger_sha256': '4' * 64, 'attestation_sha256': '5' * 64, 'qualification': q,
            'targets': {'tracknet/game8/Clip1': {'prepare': {'state': 'pending', 'frames': 100}}},
            'pending': [['tracknet/game8/Clip1', 'prepare'], ['tracknet/game8/Clip1', 'score']],
            'blocked_targets': [], 'completed': completed, 'cpu_models': {}, 'noncommand': {}, 'forecast': forecast,
            'authored_at': now.isoformat(), 'valid_until': (now + timedelta(hours=2)).isoformat()}
    gate = {'schema': 1, 'approved': True, 'release': 'full-study', 'purpose': 'full-retrospective-benchmark-cpu-readiness',
            'selection_sha256': admission.fleet.SOURCE_SHA, 'consumer_execution_identity': identity,
            'gpu_models': ['RTX A6000'], 'gpu_capability': [8, 6], 'fidelity_only_for_unqualified': True,
            'cost_receipt_sha256': '3' * 64, 'confirmation_qualification': q,
            'allowed_clips': ['tracknet/game8/Clip1'], 'namespace_bootstrap_reserve_seconds': 5}
    paths = {'native_cpu': native, 'gpu_parity': parity}
    admission.check_certificate(cert, gate, identity, 'tracknet/game8/Clip1', 'prepare', '3' * 64, '4' * 64, '5' * 64, paths, receipt)
    with pytest.raises(ValueError, match='target is not certified pending'):
        admission.check_certificate(cert, gate, identity, 'tracknet/game8/Clip1', 'score', '3' * 64, '4' * 64, '5' * 64, paths, receipt)
    cert['forecast']['cpu']['fits'] = False
    with pytest.raises(ValueError, match='budget differs'):
        admission.check_certificate(cert, gate, identity, 'tracknet/game8/Clip1', 'prepare', '3' * 64, '4' * 64, '5' * 64, paths, receipt)


def test_certificate_cost_core_allows_budget_changes_but_not_science_changes():
    from experiments.tennis import k5d_full_admission as admission

    a = dict.fromkeys(admission.COST_CORE_KEYS, 'same')
    a.update(budget_ledger_sha256='a' * 64, programme={'cpu': 1})
    b = {**a, 'budget_ledger_sha256': 'b' * 64, 'programme': {'cpu': 2}}
    assert admission.cost_core_sha256(a) == admission.cost_core_sha256(b)
    b['cost'] = 'changed'
    assert admission.cost_core_sha256(a) != admission.cost_core_sha256(b)


def test_persistent_target_claim_prevents_duplicate(tmp_path, monkeypatch):
    from experiments.tennis import k5d_full_admission as admission

    monkeypatch.setattr(admission, 'DATA_ROOT', tmp_path)
    cert = {'execution_identity': {'source_sha256': 'a' * 64}}
    admission.target_claim(cert, 'b' * 64, 'clip', 'prepare', 'job1')
    admission.target_claim(cert, 'b' * 64, 'clip', 'prepare', 'job1')
    with pytest.raises(ValueError, match='already claimed'):
        admission.target_claim(cert, 'b' * 64, 'clip', 'prepare', 'job2')


def test_full_score_spec_uses_exact_remote_certificate_producer_without_local_archive(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from experiments.jobs import inbox
    from experiments.tennis import k5d_full_admission as admission

    clip = 'tracknet/game8/Clip1'
    remote = '/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261105T000000Z-11111111/full/published.tar'
    producer = {'remote': remote, 'archive_sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64}
    future = datetime.now(timezone.utc)
    gate = {'release': 'full-study', 'confirmation_qualification': {'unshare_sha256': 'a' * 64},
            'certificate_validation_reserve_seconds': 2, 'namespace_bootstrap_reserve_seconds': 5,
            'certificate_sha256': 'c' * 64}
    names = ('full_gate', 'cost_receipt', 'fidelity_gate', 'budget_ledger', 'budget_attestation',
             'native_cpu_receipt', 'gpu_parity_receipt', 'certificate', 'certificate_microbench', 'certificate_microbench_audit')
    assets = {name: {'local': str(tmp_path / name), 'remote': '/remote/' + name, 'sha256': 'a' * 64} for name in names}
    assets['checkpoint'] = {'remote': remote, 'sha256': 'a' * 64, 'archive_sha256': 'a' * 64,
                            'checkpoint_sha256': 'b' * 64}
    monkeypatch.setattr(control.study, 'checked_json', lambda *_: gate)
    monkeypatch.setattr(control, 'approved', lambda *_: ({'cost': {'score': {'seconds16': 1}},
        'programme': {r: {'failed_start_reserve_seconds': 500, 'noncommand_groups': {'whole': 5}}
                      for r in ('gpu', 'cpu')}}, 100))
    monkeypatch.setattr(control.study, 'forecast', lambda *_: 1)
    monkeypatch.setattr(control.cf, 'asset', lambda name, *_args, **_kwargs: {'name': name})
    monkeypatch.setattr(control.cf, 'common_spec', lambda _base, extra, _deadline: {'scale': {}, 'extra': extra})
    monkeypatch.setattr(inbox, 'validate_spec', lambda *_: None)
    monkeypatch.setattr(admission, 'read_certificate', lambda *_: {'valid_until': (future + timedelta(minutes=30)).isoformat(),
        'targets': {clip: {'score': {'producer': producer}}}})
    base = {'arguments': [], 'device': 'gpu', 'cpu_threads': 1, 'single_cpu_campaign': True,
            'gpu_sharing': {'mode': 'exclusive'}, 'contention': {'policy': 'stop', 'resume_attempts': 0}}
    result = control.spec(base, phase='score', clip=clip, identity={}, assets=assets,
                          deadline=(future + timedelta(minutes=20)).isoformat(), readiness_seconds=20,
                          full_seconds=100, campaign_seconds=150)
    checkpoint = next(item for item in result['extra'] if item['name'] == 'checkpoint')
    assert checkpoint == {'name': 'checkpoint', 'path': remote, 'sha256': 'a' * 64, 'extract': True}
    assets['checkpoint']['checkpoint_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='certified producer'):
        control.spec(base, phase='score', clip=clip, identity={}, assets=assets,
                     deadline=(future + timedelta(minutes=20)).isoformat(), readiness_seconds=20,
                     full_seconds=100, campaign_seconds=150)


def test_forecast_owned_reservation_is_counted_once_across_advance():
    ledger = {'reservation_amount_basis': 'remaining_unspent', 'gpu_total_seconds': 1000, 'cpu_total_seconds': 1000,
              'workstreams': {'ball': {'gpu_limit_seconds': 800, 'cpu_limit_seconds': 800}},
              'charges': [{'phase': 'prior', 'workstream': 'ball', 'gpu_seconds': 100, 'cpu_seconds': 100}],
              'reservations': [{'phase': 'full', 'workstream': 'ball', 'status': 'running', 'gpu_seconds': 600, 'cpu_seconds': 600},
                               {'phase': 'other', 'workstream': 'w8', 'status': 'running', 'gpu_seconds': 100, 'cpu_seconds': 100}]}
    active = {'full': {'workstream': 'ball', 'gpu': 600, 'cpu': 600},
              'other': {'workstream': 'w8', 'gpu': 100, 'cpu': 100}}
    att = {'reviewed': True, 'ledger_sha256': 'a' * 64, 'active_reservations_reconciled': True,
           'active_reservations': active, 'forecast_covered_reservation_phases': ['full']}
    for r in ('gpu', 'cpu'):
        att[r] = {'ball_charged_seconds': 100, 'global_charged_seconds': 100, 'ball_limit_seconds': 800,
                  'spent_and_reserved_seconds': 700, 'global_remaining_seconds': 800}
    data = receipt()
    data.update(budget_ledger_sha256='a' * 64, budget_attestation_sha256='b' * 64,
                forecast_covered_reservation_phases=['full'])
    for r in ('gpu', 'cpu'):
        data['programme'][r].update(ball_remaining_seconds=700, global_remaining_seconds=800)
    control.check_budget(ledger, att, 'a' * 64, 'b' * 64, data)
    control.check_forecast_allocation(att, {r: {'incremental_seconds': 600} for r in ('gpu', 'cpu')})
    ledger['charges'].append({'phase': 'part', 'parent_phase': 'full', 'workstream': 'ball', 'gpu_seconds': 100, 'cpu_seconds': 100})
    for r in ('gpu', 'cpu'):
        ledger['reservations'][0][r + '_seconds'] = 500
        active['full'][r] = 500
        att[r].update(ball_charged_seconds=200, global_charged_seconds=200,
                      spent_and_reserved_seconds=700, global_remaining_seconds=700)
        data['programme'][r].update(ball_remaining_seconds=600, global_remaining_seconds=700)
    control.check_budget(ledger, att, 'a' * 64, 'b' * 64, data)
    control.check_forecast_allocation(att, {r: {'incremental_seconds': 500} for r in ('gpu', 'cpu')})
    with pytest.raises(ValueError, match='reserved allocation'):
        control.check_forecast_allocation(att, {r: {'incremental_seconds': 501} for r in ('gpu', 'cpu')})
    att['forecast_covered_reservation_phases'] = ['full', 'other']
    data['forecast_covered_reservation_phases'] = ['full', 'other']
    with pytest.raises(ValueError, match='active ball allocation'):
        control.check_budget(ledger, att, 'a' * 64, 'b' * 64, data)


@pytest.mark.parametrize('release', ['cost-confirmation', 'full-study', control.PAIR_RELEASE])
def test_control_builders_pass_real_fleet_validator(release: str, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from datetime import datetime, timedelta, timezone
    from experiments.jobs import inbox
    from experiments.tennis import k5d_full_admission as admission

    clip = 'tracknet/game9/Clip5'
    now = datetime.now(timezone.utc)
    gate: dict = {'release': release, 'confirmation_qualification': {'unshare_sha256': 'a' * 64},
                  'confirmation_caps': {r: {'jobs': {f'{clip}:prepare': 500}} for r in ('gpu', 'cpu')},
                  'certificate_validation_reserve_seconds': 2, 'namespace_bootstrap_reserve_seconds': 5,
                  'certificate_sha256': 'b' * 64}
    names = {'full_gate', 'cost_receipt', 'fidelity_gate', 'budget_ledger', 'budget_attestation',
             'native_cpu_receipt', 'gpu_parity_receipt'}
    names |= ({'certificate', 'certificate_microbench', 'certificate_microbench_audit'} if release == 'full-study'
              else {'bridge', 'calibration_source', 'consumer_source'})
    remote = '/home/itec/emanuele/Datasets/pointstream-data/test/'
    assets = {name: {'local': str(tmp_path / name), 'remote': remote + name + '.json', 'sha256': 'a' * 64}
              for name in names}
    monkeypatch.setattr(control.study, 'checked_json', lambda path, *_: (
        {'pair_cost_bound_sha256': 'a' * 64} if path.name == 'fidelity_gate' else gate))
    monkeypatch.setattr(control.benchmark.encoder, 'fidelity_gate', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(control, 'approved', lambda *_: ({'cost': {'prepare': {'seconds16': 1}},
        'programme': {r: {'failed_start_reserve_seconds': 500, 'noncommand_groups': {'whole': 5}}
                      for r in ('gpu', 'cpu')}}, 292))
    monkeypatch.setattr(control.study, 'forecast', lambda *_: 1)
    monkeypatch.setattr(control.cf, 'asset', lambda name, path, digest, _local, *, extract:
                        {'name': name, 'path': path, 'sha256': digest, 'extract': extract})
    monkeypatch.setattr(admission, 'read_certificate', lambda *_: {'valid_until': (now + timedelta(minutes=30)).isoformat()})
    base = json.loads(control.cf.BASE.read_text())
    result = control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                          deadline=(now + timedelta(minutes=20)).isoformat(),
                          readiness_seconds=20, full_seconds=100, campaign_seconds=150)
    normalized = inbox.validate_spec(result)
    assert normalized['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace']
    assert result['arguments'][0] == 'outer'
    staged = [{'name': item['name'], 'path': '/staged/' + item['name']} for item in result['staged_inputs']]
    command = inbox.phase_command(result, 'smoke', staged, '/runtime/python')
    assert command[:4] == ['/runtime/python', '-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']


def test_first_pair_budget_is_exact_bounded_and_not_full_study() -> None:
    data = receipt()
    clip = control.PAIR_CLIP
    gate: dict[str, Any] = {'allowed_clips': [clip], 'future_study_fit_proven': False,
            'confirmation_caps': {resource: {'jobs': {f'{clip}:prepare': 1450, f'{clip}:score': 1700},
                                              'external': {'prepare': 100, 'score': 100}, 'total': 3350}
                                  for resource in ('gpu', 'cpu')}}
    control.validate_calibration(data)
    control.first_pair_budget(gate, data, {})
    gate['confirmation_caps']['gpu']['jobs'][f'{clip}:score'] = 100
    with pytest.raises(ValueError, match='forecast'):
        control.first_pair_budget(gate, data, {})
    gate['confirmation_caps']['gpu']['jobs'][f'{clip}:score'] = 1700
    gate['allowed_clips'].append('tracknet/game10/Clip12')
    with pytest.raises(ValueError, match='registered clip'):
        control.first_pair_budget(gate, data, {})
    gate['allowed_clips'].pop()
    gate['confirmation_caps']['cpu']['total'] = 3351
    with pytest.raises(ValueError, match='allocation'):
        control.first_pair_budget(gate, data, {})
    data['cost']['prepare']['seconds48'] -= 1
    with pytest.raises(ValueError, match='understates'):
        control.validate_calibration(data)


def test_pair_producer_requires_root_audited_real_checkpoint() -> None:
    gate = {'pair_producer': {'audited': True, 'prepare_external_included': True, 'clip': control.PAIR_CLIP, 'frames': 292,
                              'job_id': '20261109T000000Z-12345678', 'archive_sha256': 'a' * 64,
                              'checkpoint_sha256': 'b' * 64, 'audit_sha256': 'c' * 64,
                              'remote': str(control.cf.DATA_ROOT / 'jobs/example.tar')}}
    assert control.pair_producer(gate)['frames'] == 292
    gate['pair_producer']['frames'] = 16
    with pytest.raises(ValueError, match='producer'):
        control.pair_producer(gate)


def test_pair_producer_audit_checks_actual_sealed_record(tmp_path) -> None:
    import hashlib
    import json

    identity = {'source_sha256': '1' * 64, 'snapshot_sha256': '2' * 64, 'environment_sha256': '3' * 64}
    producer = {'audited': True, 'prepare_external_included': True, 'clip': control.PAIR_CLIP, 'frames': 292,
                'job_id': '20261109T000000Z-12345678', 'archive_sha256': 'a' * 64,
                'checkpoint_sha256': 'b' * 64, 'gpu_charge_seconds': 1200, 'cpu_charge_seconds': 1200,
                'remote': str(control.cf.DATA_ROOT / 'jobs/example.tar')}
    audit = dict(producer, schema=1, terminal_status='COMPLETE', exclusive=True, timing_contaminated=False,
                 consumer_execution_identity=identity)
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(audit))
    gate = {'pair_producer': dict(producer, audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest())}
    control.verify_pair_producer_audit(gate, path, identity)
    audit['timing_contaminated'] = True
    path.write_text(json.dumps(audit))
    gate['pair_producer']['audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='producer audit differs'):
        control.verify_pair_producer_audit(gate, path, identity)
    with pytest.raises(ValueError):
        control.pair_producer({'pair_producer': {}})


def test_pair_score_accounts_only_unspent_pair_after_audited_prepare() -> None:
    data = receipt()
    clip = control.PAIR_CLIP
    jobs = {f'{clip}:prepare': 1500, f'{clip}:score': 1650}
    gate: dict[str, Any] = {'allowed_clips': [clip], 'future_study_fit_proven': False,
            'confirmation_caps': {kind: {'jobs': dict(jobs), 'external': {'prepare': 100, 'score': 100}, 'total': 3350}
                                  for kind in ('gpu', 'cpu')},
            'pair_producer': {'audited': True, 'prepare_external_included': True, 'clip': clip, 'frames': 292, 'job_id': '20261109T000000Z-12345678',
                              'archive_sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64, 'audit_sha256': 'c' * 64,
                              'remote': str(control.cf.DATA_ROOT / 'jobs/producer.tar'),
                              'gpu_charge_seconds': 1550, 'cpu_charge_seconds': 1550}}
    for kind in ('gpu', 'cpu'):
        data['programme'][kind]['ball_remaining_seconds'] = 1750
        data['programme'][kind]['global_remaining_seconds'] = 1750
    control.first_pair_budget(gate, data, {}, 'score')
    gate['pair_producer']['gpu_charge_seconds'] = 1600
    control.first_pair_budget(gate, data, {}, 'score')
    gate['pair_producer']['gpu_charge_seconds'] = 1601
    with pytest.raises(ValueError, match='producer charge'):
        control.first_pair_budget(gate, data, {}, 'score')
    gate['pair_producer']['gpu_charge_seconds'] = 1550
    data['programme']['cpu']['ball_remaining_seconds'] = 1749
    with pytest.raises(ValueError, match='reconciled'):
        control.first_pair_budget(gate, data, {}, 'score')


def test_pair_fidelity_gate_is_distinct_and_binds_cost(tmp_path) -> None:
    import json

    from experiments.tennis import k5d_encoder

    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': control.PAIR_RELEASE,
            'archive_sha256': 'archive', 'calibration_job_id': '20261105T000000Z-example',
            'evidence_sha256': 'a' * 64, 'pair_cost_bound_sha256': 'b' * 64,
            'clip': control.PAIR_CLIP, 'max_frames': 292,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'pair_cost']}
    path = tmp_path / 'fidelity.json'
    path.write_text(json.dumps(gate))
    sha = control.oracle.sha(path)
    assert k5d_encoder.fidelity_gate(path, sha, 'archive', purpose=control.PAIR_RELEASE,
                                     clip=control.PAIR_CLIP, frames=292)['passed']
    with pytest.raises(ValueError):
        k5d_encoder.fidelity_gate(path, sha, 'archive', purpose=control.PAIR_RELEASE,
                                  clip=control.PAIR_CLIP, frames=188)
    with pytest.raises(ValueError):
        k5d_encoder.fidelity_gate(path, sha, 'archive')


def test_pair_producer_charges_join_reconciled_ledger() -> None:
    producer = {'audited': True, 'prepare_external_included': True, 'clip': control.PAIR_CLIP, 'frames': 292,
                'job_id': '20261109T000000Z-12345678', 'archive_sha256': 'a' * 64,
                'checkpoint_sha256': 'b' * 64, 'audit_sha256': 'c' * 64,
                'remote': str(control.cf.DATA_ROOT / 'jobs/p.tar'), 'ledger_charge_phases': ['prep', 'close'],
                'gpu_charge_seconds': 1400, 'cpu_charge_seconds': 1450}
    ledger = {'charges': [{'phase': 'prep', 'workstream': 'ball', 'gpu_seconds': 0, 'cpu_seconds': 50},
                          {'phase': 'close', 'workstream': 'ball', 'gpu_seconds': 1400, 'cpu_seconds': 1400}]}
    control.verify_pair_ledger_charge({'pair_producer': producer}, ledger)
    ledger['charges'][1]['gpu_seconds'] = 1399
    with pytest.raises(ValueError, match='charge differs'):
        control.verify_pair_ledger_charge({'pair_producer': producer}, ledger)


@pytest.mark.parametrize('pair', [False, True])
def test_readiness_calls_real_fidelity_gate_for_both_purposes(tmp_path, monkeypatch, pair):
    import argparse
    import json

    from experiments.tennis import k5d_control_readiness as readiness

    purpose = control.PAIR_RELEASE if pair else 'full-retrospective-benchmark'
    checks = ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics',
              'pair_cost' if pair else 'full_cost']
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': purpose,
            'archive_sha256': control.fleet.EXPECTED['archive'], 'calibration_job_id': '2026-example',
            'evidence_sha256': 'a' * 64, 'checks': checks}
    if pair:
        gate.update(clip=control.PAIR_CLIP, max_frames=292, pair_cost_bound_sha256='b' * 64)
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    identity = {'environment_sha256': 'a', 'source_sha256': 'b', 'snapshot_sha256': 'c'}
    (tmp_path / 'dispatch.json').write_text(json.dumps({'staged_environment': {'sha256': 'a'},
        'code': {'source_sha256': 'b', 'snapshot_sha256': 'c'}, 'environment': {'gpu': {'name': 'RTX A6000'}}}))
    monkeypatch.setattr(readiness, 'asset_inputs', lambda *_: {'model_source_tree_sha256': {}})
    monkeypatch.setattr(readiness, 'source', lambda *_: {'frames': 292, 'labels': 292})
    args = argparse.Namespace(fidelity_gate=str(path), fidelity_gate_sha256=control.oracle.sha(path),
                              fidelity_purpose=purpose, clip=control.PAIR_CLIP, frames=292,
                              archive='unused', action='prepare')
    assert readiness.inputs(args, tmp_path, identity)['source']['frames'] == 292


@pytest.mark.parametrize('pair', [False, True])
@pytest.mark.parametrize('qualified', [False, True])
def test_benchmark_prepare_reaches_real_encoder_fidelity_gate(tmp_path, monkeypatch, pair, qualified):
    """Synthetic I/O and device boundary; actual builder, checkpoint and fidelity gate calls."""
    import argparse
    import json
    import sys
    import types

    from experiments.tennis import k5d_benchmark, k5d_checkpoint, k5d_encoder

    purpose = control.PAIR_RELEASE if pair else 'full-retrospective-benchmark'
    checks = ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics',
              'pair_cost' if pair else 'full_cost']
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': purpose,
            'archive_sha256': 'z' * 64, 'calibration_job_id': '2026-example',
            'evidence_sha256': 'a' * 64, 'checks': checks}
    if pair:
        gate.update(clip=control.PAIR_CLIP, max_frames=292, pair_cost_bound_sha256='b' * 64)
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    real_sha = control.oracle.sha
    digest = real_sha(path)
    monkeypatch.setattr(control.oracle, 'sha', lambda p: digest if Path(p) == path else 'z' * 64)
    monkeypatch.setattr(control.fleet, 'SOURCE_SHA', 'z' * 64)
    monkeypatch.setattr(control.fleet, 'EXPOSURE_SHA', 'z' * 64)
    monkeypatch.setattr(control.fleet, 'EXPECTED', {name: 'z' * 64 for name in control.fleet.EXPECTED})
    monkeypatch.setattr(control.oracle, 'select', lambda *_: {'label_rows': 292})
    monkeypatch.setattr(control.oracle, 'tree_sha', lambda *_: 'z' * 64)
    monkeypatch.setattr(control.fleet, 'archive_tree_sha', lambda *_: 'z' * 64)
    monkeypatch.setattr(control.fleet, 'snapshot_head', lambda *_: 'a' * 40)
    monkeypatch.setattr(control.oracle, 'validate_output', lambda *_: None)
    monkeypatch.setattr(control.oracle, 'run', lambda *_: {'qualified': qualified})
    monkeypatch.setattr(os, 'sched_getaffinity', lambda *_: {0}, raising=False)
    fake_torch = types.ModuleType('torch')
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'src.codecs.quality', types.ModuleType('src.codecs.quality'))
    monkeypatch.setitem(sys.modules, 'src.codecs.svtav1', types.ModuleType('src.codecs.svtav1'))
    monkeypatch.setenv('PS_STAGE', 'full')
    monkeypatch.setenv('PS_STAGE_DIR', str(tmp_path))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(tmp_path))
    monkeypatch.setenv('PS_CPU_ALLOWANCE', '1')
    monkeypatch.delenv('PS_LOCAL_ROOT', raising=False)
    (tmp_path / 'dispatch.json').write_text(json.dumps({'code': {}}))
    (tmp_path / 'staging.json').write_text(json.dumps({'inputs': [
        {'name': name, 'source': '/shared/archive.tar', 'sha256': 'z' * 64}
        for name in ('wasb_code', 'balltrack_code')]}))
    names = ('archive', 'balltrack_code', 'balltrack_weight', 'wasb_code', 'wasb_weights', 'wasb_config',
             'image_ckpt', 'video_ckpt', 'lpips_backbone', 'dists_backbone')
    args = argparse.Namespace(**{name: str(tmp_path / name) for name in names}, clip=control.PAIR_CLIP,
                              frames=292, fidelity_gate=str(path), fidelity_gate_sha256=digest)
    if pair:
        args.fidelity_purpose = purpose
    data = k5d_benchmark.build(args)
    class ReachedGate(Exception):
        pass
    monkeypatch.setattr(k5d_encoder.shutil, 'copyfile', lambda *_: (_ for _ in ()).throw(ReachedGate()))
    with pytest.raises(ReachedGate):
        k5d_checkpoint.prepare(data, {})


def test_retained_checkpoint_and_encoder_revalidate_pair_fidelity(tmp_path, monkeypatch):
    import json
    from experiments.tennis import k5d_checkpoint, k5d_encoder

    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': control.PAIR_RELEASE,
            'archive_sha256': 'z' * 64, 'calibration_job_id': '2026-example', 'evidence_sha256': 'a' * 64,
            'pair_cost_bound_sha256': 'b' * 64, 'clip': control.PAIR_CLIP, 'max_frames': 292,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'pair_cost']}
    gate_path = tmp_path / 'fidelity-gate.json'
    gate_path.write_text(json.dumps(gate))
    report = {'frames': 292, 'clip': control.PAIR_CLIP, 'checkpoint_prepare': True, 'full_metrics': False,
              'fidelity_gate': {'sha256': control.oracle.sha(gate_path), 'purpose': control.PAIR_RELEASE},
              'points': []}
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    (tmp_path / 'source-qualification.json').write_text(json.dumps({'qualified': False, 'frames': 292,
        'clip': control.PAIR_CLIP, 'identities': {'archive': {'sha256': 'z' * 64}}, 'rows': []}))
    monkeypatch.setattr(control.oracle, 'validate_report', lambda *_: None)
    # Both real validators get through fidelity and reject the deliberately incomplete synthetic payload.
    with pytest.raises(ValueError, match='complete six-point'):
        k5d_checkpoint.validate_prepared(tmp_path)
    with pytest.raises(KeyError, match='size'):
        k5d_encoder.validate(tmp_path, prepared=True)
    report['fidelity_gate']['purpose'] = 'full-retrospective-benchmark'
    (tmp_path / 'pilot.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        k5d_checkpoint.validate_prepared(tmp_path)
    with pytest.raises(ValueError):
        k5d_encoder.validate(tmp_path, prepared=True)


def test_score_rechecks_retained_pair_checkpoint_before_device(tmp_path, monkeypatch):
    """Synthetic bytes; actual coalesced score calls checkpoint and fidelity verification."""
    import argparse
    import json
    import sys
    import types
    from experiments.tennis import k5d_checkpoint as cp, k5d_coalesced as co

    root = tmp_path / 'prepared'
    root.mkdir()
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': control.PAIR_RELEASE,
            'archive_sha256': 'z' * 64, 'calibration_job_id': '2026-example', 'evidence_sha256': 'a' * 64,
            'pair_cost_bound_sha256': 'b' * 64, 'clip': control.PAIR_CLIP, 'max_frames': 292,
            'checks': ['source_transform', 'native_and_wire', 'receiver_parity', 'perceptual_metrics', 'pair_cost']}
    gate_path = root / 'fidelity-gate.json'
    gate_path.write_text(json.dumps(gate))
    points = []
    for codec, setting in cp.enc.POINTS:
        folder = root / f'{codec}-{setting}'
        folder.mkdir()
        for name in ('native.bin', 'stream.psbl', 'baseline.psbl'):
            (folder / name).write_bytes((name + codec + str(setting)).encode())
        points.append({'codec': codec, 'point': setting, 'native_sha256': cp.oracle.sha(folder / 'native.bin'),
                       'wire_sha256': cp.oracle.sha(folder / 'stream.psbl'), 'encoder_decoded_sha256': 'decode',
                       'encoder_rendered_sha256': 'render', 'decode_parity': True, 'render_parity': True,
                       'baseline': {'metrics': [], 'judges': {}, 'zero_ball_wire_sha256': cp.oracle.sha(folder / 'baseline.psbl')},
                       'streak': {'guard_off': False, 'metrics': [], 'judges': {}, 'reuse': None}})
    report = {'frames': 292, 'clip': control.PAIR_CLIP, 'checkpoint_prepare': True, 'full_metrics': False,
              'fidelity_gate': {'sha256': cp.oracle.sha(gate_path), 'purpose': control.PAIR_RELEASE}, 'points': points}
    (root / 'pilot.json').write_text(json.dumps(report))
    (root / 'source-qualification.json').write_text(json.dumps({'qualified': False,
        'identities': {'archive': {'sha256': 'z' * 64}}}))
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda *_: None)
    identity = {'source_sha256': 'a', 'snapshot_sha256': 'b', 'environment_sha256': 'c'}
    checkpoint = {'schema': 1, 'execution_identity': identity, 'clip': control.PAIR_CLIP, 'frames': 292,
                  'points': [list(p) for p in cp.enc.POINTS], 'files': cp.manifest_files(root)}
    (root / 'checkpoint.json').write_text(json.dumps(checkpoint))
    digest = cp.oracle.sha(root / 'checkpoint.json')
    fake_torch = types.ModuleType('torch')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'src.codecs.quality', types.ModuleType('src.codecs.quality'))
    class ReachedDevice(Exception):
        pass
    monkeypatch.setattr(os, 'sched_getaffinity', lambda *_: (_ for _ in ()).throw(ReachedDevice()), raising=False)
    with pytest.raises(ReachedDevice):
        co.score_all(argparse.Namespace(checkpoint=str(root), output=str(tmp_path / 'score')), digest, identity)
    gate_path.write_text('{}')
    with pytest.raises(ValueError, match='checkpoint content'):
        co.score_all(argparse.Namespace(checkpoint=str(root), output=str(tmp_path / 'score')), digest, identity)


def test_pair_checkpoint_real_archive_staged_sha_is_audited(tmp_path):
    import hashlib
    import io
    import json
    import tarfile
    from experiments.tennis import k5d_encoder

    identity = {'source_sha256': '1' * 64, 'snapshot_sha256': '2' * 64, 'environment_sha256': '3' * 64}
    manifest = json.dumps({'schema': 1, 'clip': control.PAIR_CLIP, 'frames': 292,
                           'execution_identity': identity, 'points': [list(p) for p in k5d_encoder.POINTS]}).encode()
    archive = tmp_path / 'producer.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo('publish/checkpoint.json')
        member.size = len(manifest)
        tar.addfile(member, io.BytesIO(manifest))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(manifest).hexdigest()
    remote = str(control.cf.DATA_ROOT / 'jobs/real-shaped-synthetic-producer.tar')
    producer = {'remote': remote, 'archive_sha256': archive_sha, 'checkpoint_sha256': checkpoint_sha}
    asset = {'local': str(archive), 'remote': remote, 'sha256': archive_sha,
             'archive_sha256': archive_sha, 'checkpoint_sha256': checkpoint_sha}
    control.validate_pair_checkpoint_asset(asset, producer)
    staged = control.study.checkpoint(asset, 'checkpoint', control.PAIR_CLIP, 292, identity)
    assert staged['sha256'] == archive_sha and staged['path'] == remote and staged['extract'] is True
    for key, value in [('sha256', '0' * 64), ('archive_sha256', '0' * 64),
                       ('checkpoint_sha256', '0' * 64), ('remote', remote + '.wrong')]:
        with pytest.raises(ValueError, match='audited producer'):
            control.validate_pair_checkpoint_asset({**asset, key: value}, producer)
    missing = dict(asset)
    del missing['sha256']
    with pytest.raises(ValueError, match='audited producer'):
        control.validate_pair_checkpoint_asset(missing, producer)
