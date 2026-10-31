from experiments.tennis.k6b_real_base_judge import prediction_ticks


def test_frozen_pair_expects_the_full_grid_and_a_manifest_window_its_recorded_ticks():
    grid = list(range(0, 240, 3))
    assert prediction_ticks({"ticks": grid}, manifest=False) == grid
    # A manifest unit without a pose at some frames was predicted on fewer ticks (job 20261031T102650Z-8d510571: 77 of 80).
    recorded = [t for t in grid if t != 84][:77]
    assert prediction_ticks({"ticks": recorded}, manifest=True) == recorded
    assert prediction_ticks({"ticks": grid}, manifest=True) == grid
