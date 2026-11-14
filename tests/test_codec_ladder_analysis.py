import copy

import pytest

from experiments.codec_ladder.analysis import analyse


def score():
    rows = []
    for arm in ('native', 'roi', 'semantic_patch', 'error_patch'):
        for i, rate in enumerate((35, 50, 65)):
            value = (.05, .15, .3)[i]
            n = (1000, 600, 320)[i]
            n = n if arm == 'native' else int(n * .75)
            rows.append(dict(arm=arm, crf=rate, packet_bytes=n, bound={'bytes': n - 20}, means={
                'weighted_dists': value, 'weighted_lpips': value, 'full_dists': value,
                'full_lpips': value, 'full_rgb_psnr': 40 - i * 5}))
    for row in rows:
        row['frames'] = {k: [v] * 3 for k, v in row['means'].items()}
    return dict(rows=rows, domain='tennis', source_group='synthetic', annotation_provenance='synthetic', frames=3)


def test_development_target_and_simpler_roi_explanation():
    report = analyse(score())
    assert report['decisions']['semantic_patch']['native']['development_target_met'] is True
    assert report['decisions']['semantic_patch']['roi']['development_target_met'] is False
    assert report['decisions']['roi']['native']['weighted_dists_bd_rate'] == pytest.approx(-.25)


def test_missing_or_duplicate_frames_and_nonfinite_metrics_fail():
    for change in ('missing', 'duplicate', 'nonfinite', 'price', 'omitted_frame', 'wrong_mean'):
        data = copy.deepcopy(score())
        if change == 'missing':
            data['rows'].pop()
        elif change == 'duplicate':
            data['rows'].append(data['rows'][0])
        elif change == 'nonfinite':
            data['rows'][0]['means']['weighted_dists'] = float('nan')
        elif change == 'omitted_frame':
            data['rows'][0]['frames']['full_dists'].pop()
        elif change == 'wrong_mean':
            data['rows'][0]['means']['full_dists'] += .1
        else:
            data['rows'][-1]['bound']['bytes'] = 999999
        with pytest.raises(ValueError):
            analyse(data)
