"""The proposed resource gate must account setup once and fail on weak evidence."""
import math

import pytest

from experiments.tennis.k6b_real_base_cost import separated_projection


def traces(sample=(4.0, 2.0, 2.0)):
    return {name: list(sample) for name in ('al/control', 'med/control', 'al/base', 'med/base')}


def test_complete_cost_ledger_includes_fixed_cold_warm_and_publication():
    result = separated_projection(42, traces(), 2)
    assert result['fixed_seconds'] == 10
    assert result['projected_worker_seconds'] == pytest.approx(1.35 * (10 + 4 * (80 * 2 + 2)))
    assert result['projected_publication_seconds'] == pytest.approx(72)
    assert result['projected_full_seconds'] == pytest.approx(960.3)


def test_added_setup_is_paid_once_and_never_disappears():
    old = separated_projection(42, traces(), 0)
    new = separated_projection(52, traces(), 0)
    assert new['projected_full_seconds'] - old['projected_full_seconds'] == pytest.approx(13.5)


def test_slower_steady_sample_and_exact_stability_boundary():
    result = separated_projection(100, traces((1, 2, 2.5)), 0)
    assert all(r['warm_seconds'] == 2.5 and r['cold_excess_seconds'] == 0 for r in result['sequences'].values())
    with pytest.raises(ValueError, match='25%'):
        separated_projection(100, traces((1, 2, 2.500001)), 0)


@pytest.mark.parametrize('sample', [(), (1,), (1, 2), (1, 2, 2, 2), (0, 1, 1), (1, 0, 1), (1, -1, 1), (1, math.nan, 1), (1, math.inf, 1), (True, 1, 1)])
def test_incomplete_invalid_or_zero_samples_fail(sample):
    with pytest.raises(ValueError):
        separated_projection(100, traces(sample), 0)


@pytest.mark.parametrize('command,publication', [(-1, 0), (math.nan, 0), (math.inf, 0), (42, -1), (42, math.nan), (42, math.inf), (True, 0), (42, False)])
def test_invalid_native_or_publication_timing_fails(command, publication):
    with pytest.raises(ValueError):
        separated_projection(command, traces(), publication)


def test_timing_overlap_or_missing_sequence_cannot_promote():
    with pytest.raises(ValueError, match='exceeds'):
        separated_projection(31, traces(), 0)
    incomplete = traces()
    incomplete.pop('med/base')
    with pytest.raises(ValueError, match='four complete'):
        separated_projection(100, incomplete, 0)


def test_saved_model_is_explicit_and_rejects_ambiguous_switches():
    from experiments.tennis.k6b_real_base_cost import requested_model
    assert requested_model({'arguments': []}) == 'legacy'
    assert requested_model({'arguments': ['--cost-model', 'separated-v1']}) == 'separated-v1'
    for arguments in (['--cost-model'], ['--cost-model', 'unknown'], ['--cost-model', 'legacy', '--cost-model', 'separated-v1']):
        with pytest.raises(ValueError):
            requested_model({'arguments': arguments})


def test_strict_reservation_and_publication_are_both_binding():
    from experiments.tennis.k6b_real_base_cost import resource_checks
    result = separated_projection(42, traces(), 2)
    assert all(resource_checks(result, 1000).values())
    checks = resource_checks(result, result['projected_full_seconds'])
    assert checks['projection_under_full_reservation']
    assert not checks['projection_including_publication']
    assert not any(resource_checks(result, math.nan).values())


def test_explicit_secondary_timing_keeps_slower_warm_and_budget_checks():
    from experiments.tennis.k6b_real_base_cost import resource_checks
    result = separated_projection(100, traces((1, 1, 2)), 0, require_stability=False)
    assert result['stability_required'] is False
    assert all(r['warm_ratio'] == 2 and r['warm_seconds'] == 2 for r in result['sequences'].values())
    assert result['projected_full_seconds'] == pytest.approx(1.35 * (84 + 640))
    assert not all(resource_checks(result, 550).values())
    with pytest.raises(ValueError, match='25%'):
        separated_projection(100, traces((1, 1, 2)), 0)
