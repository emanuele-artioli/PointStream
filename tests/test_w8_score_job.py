import copy
import math
import tarfile

import pytest

from experiments.tennis.w8_score_job import MANIFEST_SHA256, check_publication, projection


def example():
    spec = {'device': 'gpu', 'single_cpu_campaign': True, 'defer_failed_publication': True, 'cpu_threads': 1,
            'arguments': ['--repair-manifest', '{staged:repair_manifest}'],
            'staged_inputs': [{'name': 'repair_manifest', 'sha256': MANIFEST_SHA256}],
            'contention': {'policy': 'stop', 'resume_attempts': 0},
            'scale': {'judged': {'smoke': 3, 'full': 0}}, 'smoke': {'seconds': 40},
            'full': {'seconds': 195}, 'validator_seconds': 10, 'budget_seconds': 280}
    dispatch = {'cpu_affinity': [2], 'campaign_clock': {'started': 100, 'phase_started': 110, 'budget_end': 380}}
    return spec, dispatch, {'passed': True, 'projected_full_seconds': 140}


def test_projection_admits_only_when_the_whole_remaining_work_fits():
    spec, dispatch, validation = example()
    result = projection(spec, dispatch, validation, 160)
    assert result['second_staging_reserve_seconds'] == 12.5
    assert result['required_seconds'] == 212.5
    with pytest.raises(ValueError, match='remaining'):
        projection(spec, dispatch, validation, 168)


def test_measured_setup_expands_future_staging_reserve():
    spec, dispatch, validation = example()
    dispatch['campaign_clock']['phase_started'] = 120
    spec['full']['seconds'] = 150
    assert projection(spec, dispatch, validation, 130)['second_staging_reserve_seconds'] == 25


def test_prior_attempts_and_exclusive_allocation_fit_authorized_caps():
    assert 67 + 280 + 30 + 10 + 30 == 417 <= 450
    assert 67 + 280 + 30 == 377 <= 450
    spec, dispatch, validation = example()
    dispatch['campaign_clock']['phase_started'] = 106.61884
    validation['projected_full_seconds'] = 170.7293615
    result = projection(spec, dispatch, validation, 124.82815)
    assert result['required_seconds'] == pytest.approx(208.27355)
    assert result['remaining_seconds'] - result['required_seconds'] == pytest.approx(46.8983)


def test_explicit_exclusive_admission_is_accepted():
    spec, dispatch, validation = example()
    spec['gpu_sharing'] = {'mode': 'exclusive'}
    dispatch['gpu_sharing'] = {'mode': 'exclusive'}
    assert projection(spec, dispatch, validation, 160)['publication_reserve_seconds'] == 5


@pytest.mark.parametrize('mutation', [
    lambda s, d, v: s.update(single_cpu_campaign=False),
    lambda s, d, v: s.update(defer_failed_publication=False),
    lambda s, d, v: s.update(cpu_threads=2),
    lambda s, d, v: s.update(cpu_threads=True),
    lambda s, d, v: s['scale']['judged'].update(full=3),
    lambda s, d, v: s.update(gpu_sharing={'mode': 'shared'}),
    lambda s, d, v: d.update(gpu_sharing={'mode': 'shared'}),
    lambda s, d, v: s.update(contention={'policy': 'continue', 'resume_attempts': 0}),
    lambda s, d, v: s.update(contention={'policy': 'stop', 'resume_attempts': 1}),
    lambda s, d, v: s.update(contention=None),
    lambda s, d, v: s.update(budget_seconds=281),
    lambda s, d, v: s.update(validator_seconds=11),
    lambda s, d, v: s['smoke'].update(seconds=41),
    lambda s, d, v: s['full'].update(seconds=196),
    lambda s, d, v: s.update(arguments=[]),
    lambda s, d, v: s.update(arguments=['--repair-manifest', 'wrong']),
    lambda s, d, v: s.update(staged_inputs=[]),
    lambda s, d, v: s['staged_inputs'][0].update(sha256='a' * 64),
    lambda s, d, v: d.pop('campaign_clock'),
    lambda s, d, v: d['campaign_clock'].update(phase_started=99),
    lambda s, d, v: d.update(cpu_affinity=[1, 2]),
    lambda s, d, v: v.update(passed=False),
    lambda s, d, v: v.update(projected_full_seconds=math.nan),
    lambda s, d, v: v.update(projected_full_seconds=196),
])
def test_projection_rejects_missing_unsafe_or_nonfinite_inputs(mutation):
    spec, dispatch, validation = copy.deepcopy(example())
    mutation(spec, dispatch, validation)
    with pytest.raises((ValueError, KeyError)):
        projection(spec, dispatch, validation, 160)


def test_publication_bound_accepts_two_small_regular_files(tmp_path):
    publish = tmp_path / 'publish'
    publish.mkdir()
    (publish / 'score.json').write_bytes(b'{}')
    (publish / 'scores.npz').write_bytes(b'abc')
    estimate = check_publication(tmp_path)
    archive = tmp_path / 'result.tar'
    with tarfile.open(archive, 'w') as bundle:
        bundle.add(publish, arcname='publish')
    assert archive.stat().st_size <= estimate < 200 * 1024


@pytest.mark.parametrize('size', [512, 7169, 17409, 110593])
def test_publication_tar_estimate_covers_record_padding(tmp_path, size):
    publish = tmp_path / 'publish'
    publish.mkdir()
    (publish / 'score.json').write_bytes(b'x' * size)
    (publish / 'scores.npz').write_bytes(b'y' * 513)
    estimate = check_publication(tmp_path)
    archive = tmp_path / 'result.tar'
    with tarfile.open(archive, 'w') as bundle:
        bundle.add(publish, arcname='publish')
    assert archive.stat().st_size <= estimate


@pytest.mark.parametrize('kind', ['extra', 'symlink', 'oversized'])
def test_publication_bound_rejects_unsafe_outputs(tmp_path, kind):
    publish = tmp_path / 'publish'
    publish.mkdir()
    (publish / 'score.json').write_bytes(b'{}')
    scores = publish / 'scores.npz'
    if kind == 'symlink':
        scores.symlink_to(publish / 'score.json')
    else:
        scores.write_bytes(b'x' * (200 * 1024 if kind == 'oversized' else 3))
    if kind == 'extra':
        (publish / 'extra').write_bytes(b'x')
    with pytest.raises(ValueError):
        check_publication(tmp_path)
