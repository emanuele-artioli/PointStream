import pytest

from experiments.tennis import k5d_full_cost_builder as builder


def test_full_study_counts_every_smoke_and_original_frame_once_per_phase() -> None:
    costs = {kind: builder.model(100, 132, 1.5, 20) for kind in ('prepare', 'score')}
    allowances = {kind: {'per_stage_staging_seconds': 2, 'per_campaign_validator_seconds': 3,
                         'per_campaign_audit_seconds': 4, 'failed_start_reserve_seconds': 50}
                  for kind in ('gpu', 'cpu')}
    budgets = {kind: {'spent_and_reserved_seconds': 100, 'ball_limit_seconds': 100000,
                      'global_remaining_seconds': 100000} for kind in ('gpu', 'cpu')}
    result = builder.projection(costs, allowances, budgets)
    assert (result['clips'], result['unique_frames'], result['campaigns'], result['smoke_stages']) == (30, 5675, 60, 60)
    assert result['full_native_frame_visits'] == 11350
    assert result['confirmation_order'] == ['tracknet/game9/Clip5', 'tracknet/game10/Clip12']
    assert result['ordered_stages'][0]['clip'] == 'tracknet/game9/Clip5'
    assert result['ordered_stages'][2]['clip'] == 'tracknet/game10/Clip12'
    assert result['cpu_compute_seconds'] is None
    assert result['cpu_incremental_seconds'] is None
    assert result['cpu_noncommand_seconds'] == 120 * 2 + 60 * 3 + 60 * 4 + 50
    assert result['budget_fits'] == {'gpu': True, 'cpu': False}
    budgets['gpu']['global_remaining_seconds'] = 1
    assert builder.projection(costs, allowances, budgets)['budget_fits']['gpu'] is False


def test_fit_rejects_invalid_or_unphysical_evidence() -> None:
    with pytest.raises(ValueError, match='negative startup'):
        builder.model(10, 100, 1.5, 20)
    with pytest.raises(ValueError):
        builder.model(100, 132, True, 20)
    with pytest.raises(ValueError):
        builder.model(100, 132, 1.49, 20)
