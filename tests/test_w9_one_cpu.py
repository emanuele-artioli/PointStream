from experiments.tennis.w1_encode import cpu_profile_valid


def test_cpu_profile_checks_each_lane_and_rejects_false_confinement():
    assert cpu_profile_valid([3], [[3], [3]], single_cpu=True)
    assert not cpu_profile_valid([3, 4], [[3]], single_cpu=True)
    assert not cpu_profile_valid([3], [[4]], single_cpu=True)
    assert not cpu_profile_valid([3], [], single_cpu=True)
    assert cpu_profile_valid([0, 1, 2, 3, 4, 5], [[0, 1], [2, 3]], single_cpu=False)
    assert not cpu_profile_valid([3], [[3]], single_cpu=False)
