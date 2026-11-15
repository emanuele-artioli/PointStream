import copy
import math

import pytest

from experiments.tennis import ball_confirmation_analysis as analysis


def source_rows(values):
    return [{'source_id': source_id, 'comparisons': {'reference': {
        'crop_dists': {'bd_rate': {'hull_pchip': value}}}}}
        for source_id, value in zip(analysis.physical.SOURCES, values)]


def test_equal_source_log_ratio_and_all_27_match_resamples():
    result = analysis.source_summary(source_rows([-.1, -.2, -.3]), 'crop_dists')
    assert result['mean_bd_rate'] == pytest.approx((.9 * .8 * .7) ** (1 / 3) - 1)
    assert len(result['bootstrap_resamples']) == 27
    assert result['direction_supported'] is True
    assert result['bootstrap_95_interval'][1] < 0
    assert result['replicate_unit'] == 'physical match'


def test_unsupported_or_non_improving_group_cannot_be_removed():
    result = analysis.source_summary(source_rows([-.1, None, -.3]), 'crop_dists')
    assert result['mean_bd_rate'] is None and not result['direction_supported']
    assert result['status'] == 'inconclusive_unsupported_common_quality'
    assert not analysis.source_summary(source_rows([-.4, .01, -.2]), 'crop_dists')['direction_supported']
    with pytest.raises(ValueError):
        analysis.source_summary(source_rows([-.1, -.2, -.3])[:2], 'crop_dists')
    duplicated = source_rows([-.1, -.2, -.3])
    duplicated[2]['source_id'] = duplicated[0]['source_id']
    with pytest.raises(ValueError):
        analysis.source_summary(duplicated, 'crop_dists')


def test_sparse_crop_mean_retains_all_frame_full_mean():
    labels = [(1, 50., 50.), (-1, 0., 0.), (0, 0., 0.)]
    frames = []
    for index in range(3):
        frames.append({'sse': 10, 'pixels': 100,
            **{key: (0.2 if index == 0 else None) if key.startswith('crop_') else float(index)
               for key in analysis.METRICS[:-1]}})
    row = {'frames': frames}
    means = analysis.means(row, labels)
    assert means['crop_dists'] == pytest.approx(.2)
    assert means['full_dists'] == 1.0
    assert means['rgb_psnr'] == pytest.approx(10 * math.log10(255**2 * 300 / 30))
    tampered = copy.deepcopy(row)
    tampered['frames'][1]['crop_lpips'] = 0.0
    with pytest.raises(ValueError):
        analysis.means(tampered, labels)
