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
    cpu = {'schema': 1, 'audited': True, 'passed': True, 'scope': 'native CPU PID containment and accounting',
           'namespace_wrapper_sha256': 'w' * 64, 'unshare_sha256': 'd' * 64,
           'consumer_execution_identity': {'new': 'new'}, 'job_id': '20261105T000000Z-12345678',
           'cases': ['normal', 'timeout', 'sigterm', 'sigkill'], 'all_owned_pids_absent': True,
           'independent_cpu_match': True}
    parity = {'schema': 1, 'audited': True, 'passed': True, 'scope': 'actual 16-frame GPU native and score parity',
              'consumer_execution_identity': {'new': 'new'}, 'job_id': '20261105T000000Z-abcdef12',
              'gpu_model': 'RTX A6000', 'gpu_capability': [8, 6], 'all_six_exact': True, 'namespace_wrapper_sha256': 'w' * 64}
    objects = {'gate': gate, 'cost': data, 'bridge': bridge, 'ledger': {}, 'budget': {}, 'cpu': cpu, 'parity': parity}
    monkeypatch.setattr(study, 'checked_json', lambda path, _: objects[path.name])
    monkeypatch.setattr(study, 'selection', lambda: {clips[0]: {'label_rows': 292}, clips[1]: {'label_rows': 586}})
    monkeypatch.setattr(control.bridge, 'bridge', lambda *_: bridge)
    monkeypatch.setattr(control.evidence, 'historical', lambda *_: None)
    monkeypatch.setattr(control, 'check_budget', lambda *_: None)
    monkeypatch.setattr(control, 'programme_projection', lambda *_: {'gpu': {'fits': False}, 'cpu': {'fits': False}})
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
    gate['release'] = 'full-study'
    data['budget_fits'] = {'gpu': True, 'cpu': True}
    with pytest.raises(ValueError, match='full programme exceeds'):
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
    base: dict = {'arguments': []}
    result = control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                          deadline='later', readiness_seconds=20, full_seconds=100, campaign_seconds=150)
    assert result['entrypoint'] == ['-m', 'experiments.tennis.k5d_scoped_namespace', 'outer']
    assert result['arguments'][:4] == ['--unshare-sha', 'a' * 64, '--timeout', '{namespace_timeout}']
    assert '--native-cpu-receipt' in result['arguments'] and '--gpu-parity-receipt' in result['arguments']
    with pytest.raises(ValueError, match='job cap'):
        control.spec(base, phase='prepare', clip=clip, identity={}, assets=assets,
                     deadline='later', readiness_seconds=20, full_seconds=490, campaign_seconds=550)
