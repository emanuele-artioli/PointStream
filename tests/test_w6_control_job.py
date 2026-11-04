import numpy as np

from experiments.tennis.w6_control_job import compare, comparison


def test_literal_control_requires_all_three_arrays_and_matching_nan_masks():
    reference = np.array([np.nan, 1.0, 2.0])
    assert compare(reference, reference)['passed']
    assert not compare(np.array([np.nan, 1.0 + 1.1e-6, 2.0]), reference)['passed']
    assert not compare(np.array([0.0, 1.0, 2.0]), reference)['passed']
    assert not compare(np.array([np.nan, np.inf]), np.array([np.nan, np.inf]))['passed']
    saved = {'variants': np.array(['hybrid']), 'flicker': reference[None],
             'energy': reference[None], 'energy_src': reference}
    got = {'flicker': reference.copy(), 'energy': reference.copy(), 'source_energy': reference.copy()}
    assert comparison(got, saved, 3)['passed']
    got['energy'][2] += 1.1e-6
    assert not comparison(got, saved, 3)['passed']
