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
