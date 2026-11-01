import copy
import math

import pytest

from experiments.tennis.w8_score_job import projection


def example():
    spec = {'device': 'gpu', 'single_cpu_campaign': True, 'defer_failed_publication': True, 'cpu_threads': 1,
            'scale': {'judged': {'smoke': 3, 'full': 0}}, 'smoke': {'seconds': 40},
            'full': {'seconds': 150}, 'validator_seconds': 20, 'budget_seconds': 230}
    dispatch = {'cpu_affinity': [2], 'campaign_clock': {'started': 100, 'phase_started': 110, 'budget_end': 330}}
    return spec, dispatch, {'passed': True, 'projected_full_seconds': 140}


def test_projection_admits_only_when_the_whole_remaining_work_fits():
    spec, dispatch, validation = example()
    result = projection(spec, dispatch, validation, 130)
    assert result['second_staging_reserve_seconds'] == 15
    assert result['required_seconds'] == 180
    with pytest.raises(ValueError, match='remaining'):
        projection(spec, dispatch, validation, 161)


def test_measured_setup_expands_future_staging_reserve():
    spec, dispatch, validation = example()
    dispatch['campaign_clock']['phase_started'] = 120
    assert projection(spec, dispatch, validation, 130)['second_staging_reserve_seconds'] == 30


@pytest.mark.parametrize('mutation', [
    lambda s, d, v: s.update(single_cpu_campaign=False),
    lambda s, d, v: s.update(defer_failed_publication=False),
    lambda s, d, v: s.update(cpu_threads=2),
    lambda s, d, v: s['scale']['judged'].update(full=3),
    lambda s, d, v: s.update(budget_seconds=231),
    lambda s, d, v: d.pop('campaign_clock'),
    lambda s, d, v: d['campaign_clock'].update(phase_started=99),
    lambda s, d, v: d.update(cpu_affinity=[1, 2]),
    lambda s, d, v: v.update(passed=False),
    lambda s, d, v: v.update(projected_full_seconds=math.nan),
    lambda s, d, v: v.update(projected_full_seconds=151),
])
def test_projection_rejects_missing_unsafe_or_nonfinite_inputs(mutation):
    spec, dispatch, validation = copy.deepcopy(example())
    mutation(spec, dispatch, validation)
    with pytest.raises((ValueError, KeyError)):
        projection(spec, dispatch, validation, 130)
