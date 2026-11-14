import math

import pytest
from scipy.integrate import quad

from src.pointstream.evaluation import (RegionalFrame, aggregate_regions, anchor_sensitivity,
                                       weighted_region, _linear_rate_area)


def test_every_frame_including_empty_regions_contributes() -> None:
    frames = [RegionalFrame(0.1, 0.3, 10, 90), RegionalFrame(None, 0.8, 0, 100),
              RegionalFrame(0.2, None, 100, 0)]
    result = aggregate_regions(frames)
    assert result['values'] == pytest.approx([0.16, 0.8, 0.2])
    assert result['mean'] == pytest.approx(1.16 / 3)
    assert (result['frames'], result['foreground_only'], result['background_only'],
            result['both_regions']) == (3, 1, 1, 1)


@pytest.mark.parametrize('frame', [RegionalFrame(None, 0.1, 1, 1),
    RegionalFrame(float('nan'), 0.1, 1, 1), RegionalFrame(0.1, float('inf'), 1, 1),
    RegionalFrame(-0.1, 0.1, 1, 1), RegionalFrame(None, None, 0, 0),
    RegionalFrame(0.1, 0.1, 0, 1), RegionalFrame(0.1, 0.1, -1, 1),
    RegionalFrame(True, 0.1, 1, 1), RegionalFrame(0.1, 0.1, True, 1)])
def test_invalid_frames_cannot_be_silently_dropped(frame: RegionalFrame) -> None:
    with pytest.raises(ValueError):
        aggregate_regions([RegionalFrame(0.1, 0.2, 1, 1), frame])


def test_invalid_weight_and_missing_sequence() -> None:
    for weight in [-0.1, 1.1, math.nan, True]:
        with pytest.raises(ValueError):
            weighted_region(RegionalFrame(0.1, 0.2, 1, 1), weight)
    with pytest.raises(ValueError):
        aggregate_regions([])


def test_constant_rate_saving_and_common_support() -> None:
    result = anchor_sensitivity([(80, 1), (160, 2), (320, 3)],
                                [(100, 1), (200, 2), (400, 3)])
    assert result['common_quality_interval'] == [1, 3]
    assert result['bd_rate'] == pytest.approx(dict.fromkeys(
        ['hull_pchip', 'pareto_pchip', 'hull_linear_rate'], -0.2))


def test_geometry_changes_report_without_extrapolation() -> None:
    result = anchor_sensitivity([(120, 1.5), (240, 2.5), (480, 3.5)],
                                [(100, 1), (200, 1.8), (400, 3)])
    assert result['anchor_pareto'] == [(100, 1), (200, 1.8), (400, 3)]
    assert result['anchor_linear_rate_hull'] == [(100, 1), (200, 1.8), (400, 3)]
    assert result['common_quality_interval'] == [1.5, 3]
    rates = result['bd_rate']
    assert isinstance(rates, dict)
    assert rates['hull_pchip'] != pytest.approx(rates['hull_linear_rate'])
    bent = anchor_sensitivity([(100, 1), (400, 3)], [(100, 1), (200, 1.4), (400, 3)])
    assert bent['anchor_linear_rate_hull'] == [(100, 1), (400, 3)]
    bent_rates = bent['bd_rate']
    assert isinstance(bent_rates, dict)
    assert bent_rates['hull_pchip'] != pytest.approx(bent_rates['pareto_pchip'])


def test_literal_linear_integral_matches_quadrature() -> None:
    front = [(10, 1), (300, 2), (300.000001, 3)]
    expected = quad(lambda q: math.log(10 + 290 * (q - 1)), 1.2, 2)[0]
    expected += quad(lambda q: math.log(300 + 0.000001 * (q - 2)), 2, 2.8)[0]
    assert _linear_rate_area(front, 1.2, 2.8) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize('candidate,anchor,options,status', [
    ([(10, 4), (20, 5)], [(10, 1), (20, 2)], {}, 'unsupported_range'),
    ([(10, 1)], [(10, 1), (20, 2)], {}, 'unsupported_range'),
    ([(10, 1), (20, 2)], [(10, 1), (20, 2)], {'minimum_quality_span': 2}, 'unsupported_range'),
    ([(10, 0.1), (20, 0.5)], [(10, 0.2), (20, 0.6)], {'discrete': True}, 'discrete_unsupported'),
])
def test_unsupported_curves_get_no_bd_rate(candidate, anchor, options, status) -> None:
    result = anchor_sensitivity(candidate, anchor, **options)
    assert result['status'] == status
    assert result['common_quality_interval'] is None
    rates = result['bd_rate']
    assert isinstance(rates, dict)
    assert all(value is None for value in rates.values())


@pytest.mark.parametrize('points', [[], [(0, 1)], [(math.inf, 1)], [(10, math.nan)], [(True, 1)]])
def test_invalid_saved_points_fail(points) -> None:
    with pytest.raises(ValueError):
        anchor_sensitivity(points, [(10, 1), (20, 2)])


def test_historical_geometry_reproduces_saved_analysis() -> None:
    from experiments.tennis.k5 import bd_rate

    candidate: list[tuple[float, float]] = [(80, -0.8), (200, -0.4), (450, -0.1)]
    anchor: list[tuple[float, float]] = [(100, -0.9), (200, -0.8), (400, -0.2), (500, -0.1)]
    result = anchor_sensitivity(candidate, anchor)
    rates = result['bd_rate']
    assert isinstance(rates, dict)
    assert rates['hull_pchip'] == pytest.approx(bd_rate(candidate, anchor, 0.0))


def test_duplicate_rates_and_quality_plateaus_are_canonicalized() -> None:
    result = anchor_sensitivity([(20, 1), (20, 2), (40, 2), (80, 3)],
                                [(20, 1), (40, 2), (80, 3)])
    assert result['candidate_pareto'] == [(20, 2), (80, 3)]
    assert result['common_quality_interval'] == [2, 3]


def test_saved_scalar_cli_replay_and_no_overwrite(tmp_path) -> None:
    import json
    from experiments.tennis.anchor_sensitivity import main

    source, output, replay = (tmp_path / name for name in ('input.json', 'result.json', 'replay.json'))
    source.write_text(json.dumps({'curves': [{'metric': 'distance', 'candidate': [[80, 1], [160, 2]],
                                            'anchor': [[100, 1], [200, 2]], 'discrete': False}]}))
    assert main(['--input', str(source), '--output', str(output)]) == 0
    assert main(['--input', str(output), '--output', str(replay)]) == 0
    assert json.loads(output.read_bytes()) == json.loads(replay.read_bytes())
    with pytest.raises(SystemExit):
        main(['--input', str(source), '--output', str(output)])


def test_primary_authentication_failure_and_missing_data(tmp_path) -> None:
    import hashlib
    import json
    from experiments.tennis.anchor_sensitivity import primary_curves

    acceptance, report = tmp_path / 'acceptance.json', tmp_path / 'report.json'
    acceptance.write_text(json.dumps({'posthoc_curves': {'report_sha256': hashlib.sha256(b'expected').hexdigest()},
        'job_id': 'a', 'GPU': {}, 'execution_identity': {}, 'state': 'accepted'}))
    assert primary_curves(acceptance, report)['status'] == 'authenticated_scalars_unavailable'
    report.write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        primary_curves(acceptance, report)
