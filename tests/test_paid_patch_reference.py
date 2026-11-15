"""No codecs/models: authenticated saved metrics plus synthetic paid prices."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.tennis import paid_patch_reference as reference, paid_patch_profile as paid
from experiments.tennis import ball_roi_control as common

ROOT = Path(__file__).resolve().parents[1] / 'docs/experiments'


def inputs():
    fixed = [(ROOT / name).read_bytes() for name in (
        'submission-ladder-anchor-sensitivity-20261114.json',
        'submission-ladder-ball-roi-scores-20261115.json',
        'submission-ladder-ball-pixel-oracle-scores-20261115.json')]
    pixel = json.loads(fixed[2])
    curves = {'paid': [], 'native': [], 'streak': []}
    for row in pixel['rows']:
        curves['paid'].append({'crf': row['crf'], 'bytes': row['coded_bytes'] + 1000,
                               'packet_sha256': str(row['crf'] % 10) * 64,
                               'means': common.metric_means(row['frames'])})
    for row in pixel['historical_points']:
        for arm, rate, metrics in (('native', 'native_bytes', 'baseline'), ('streak', 'wire_bytes', 'streak')):
            curves[arm].append({'crf': row['crf'], 'bytes': row[rate], 'means': common.metric_means(row[metrics])})
    synthetic = {'score_sha256': paid.PIXEL_SCORE, 'paid_stage_sha256': 'a' * 64,
                 'paid_archive_sha256': 'b' * 64, 'physical_curves': curves}
    return fixed, synthetic


def run(fixed, data):
    encoded = json.dumps(data).encode()
    return reference.analyse(*fixed, encoded, expected_paid_sha256=reference.sha(encoded))


def test_authentic_quality_joins_three_paid_prices_to_both_references():
    fixed, data = inputs()
    report = run(fixed, data)
    assert len(report['results']) == 5
    assert report['input_sha256']['paid_reprice']
    assert report['paid_stage_sha256'] == 'a' * 64
    assert report['paid_archive_sha256'] == 'b' * 64
    for metric, values in report['results'].items():
        assert len(values['paid_points_kbps_oriented_quality']) == 3
        assert len(values['GT_ROI_points_kbps_oriented_quality']) == 3
        assert len(values['pooled_points_kbps_oriented_quality']) == 9
        sign = 1 if metric == 'rgb_psnr' else -1
        assert values['paid_points_kbps_oriented_quality'][0] == (
            data['physical_curves']['paid'][0]['bytes'] * .005,
            sign * data['physical_curves']['paid'][0]['means'][metric])
    assert 'not standard scalar' in report['results']['full_lpips']['metric_definition']


@pytest.mark.parametrize('change', ['quality', 'matrix', 'underprice', 'packet', 'native_price', 'streak_quality', 'stage'])
def test_hash_authenticated_paid_metadata_still_needs_full_scalar_join(change):
    fixed, original = inputs()
    data = deepcopy(original)
    if change == 'quality':
        data['physical_curves']['paid'][0]['means']['crop_lpips'] += .001
    elif change == 'matrix':
        data['physical_curves']['paid'][1]['crf'] = 54
    elif change == 'underprice':
        data['physical_curves']['paid'][0]['bytes'] -= 1000
    elif change == 'packet':
        del data['physical_curves']['paid'][0]['packet_sha256']
    elif change == 'native_price':
        data['physical_curves']['native'][0]['bytes'] -= 608
    elif change == 'streak_quality':
        data['physical_curves']['streak'][0]['means']['full_dists'] += .001
    else:
        data['paid_stage_sha256'] = 'not-a-hash'
    with pytest.raises(ValueError):
        run(fixed, data)


def test_all_four_inputs_are_hash_authenticated():
    fixed, data = inputs()
    encoded = json.dumps(data).encode()
    for index in range(3):
        modified = list(fixed)
        modified[index] += b'\n'
        with pytest.raises(ValueError, match='hash'):
            reference.analyse(*modified, encoded, expected_paid_sha256=reference.sha(encoded))
    with pytest.raises(ValueError, match='hash'):
        reference.analyse(*fixed, encoded + b'\n', expected_paid_sha256=reference.sha(encoded))


def test_disjoint_quality_support_keeps_null_bd_and_separate_ordinal_evidence():
    result = reference.compare_curve([(100., -.01), (200., -.005)], [(90., -.2), (210., -.1)])
    assert result['bd_rate']['status'] == 'unsupported_range'
    assert result['bd_rate']['bd_rate']['hull_pchip'] is None
    assert result['ordinal']['status'] == 'candidate_strictly_better'
    assert result['ordinal']['common_rate_interval'] == [100., 200.]
