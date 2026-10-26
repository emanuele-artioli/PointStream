"""The joint pilot pays actual banks/mattes and never promotes partial cohorts."""
import pytest

from experiments.tennis.k6b_foreground_video import PILOTS
from experiments.tennis.k6b_joint_matte import assess


def evidence():
    original, refs, wire, measured = [], [], [], []
    for wid in PILOTS:
        original.append({'window': wid, 'registered_seconds': 2., 'base_checkpoint_bytes': 900000,
            'anchor_points': {'lo': {'kbps': 2., 'lpips': .5, 'dists': .5}, 'hi': {'kbps': 8., 'lpips': .2, 'dists': .2}}})
        refs.append({'window': wid, 'kbps': 3., 'rate_components': {'actual_reference_bank': .1}})
        wire.append({'window': wid, 'mattes': {'exact': {'bytes': 1000}}})
        measured.append({'id': wid, 'judged': [0, 3], 'original_judged': [0, 3],
            'selected_packets': {str(s): {'bytes': 250, 'sha256': 'frozen'} for s in (8, 16, 32)},
            'means': {str(s): {'lpips': .1, 'dists': .1} for s in (0, 8, 16, 32)}})
    return {'rows': original}, {'settings': {'full': {'rows': refs}}}, {'rows': wire}, measured


def test_real_rates_and_paid_vs_granted_exact_matte_are_distinct():
    result = assess(*evidence())['settings']
    assert result['8']['rows'][0]['kbps'] == 4.  # 3 actual method + 1 actual matte
    assert result['0']['rows'][0]['kbps'] == 3.
    assert result['0']['rows'][0]['exact_matte_paid']['lpips']['rate'] == 7.
    assert result['8']['rows'][0]['base_charged']['lpips']['rate'] == 5.
    assert all(result[str(s)]['passes_3_of_4'] for s in (0, 8, 16, 32))


def test_partial_or_changed_pilot_cannot_decide():
    args = evidence()
    args[-1][0]['judged'] = [0]
    with pytest.raises(ValueError, match='complete original'):
        assess(*args)
    args = evidence()
    args[-1].reverse()
    with pytest.raises(ValueError, match='complete original'):
        assess(*args)
