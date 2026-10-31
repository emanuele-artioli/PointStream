import math

import pytest

from experiments.tennis import w7_analysis, w7_billing_corrected


def test_correction_changes_the_tier_decision_without_changing_scores():
    refs = [{'file': f'{crop}.{codec}', 'codec': codec, 'bytes': 50_000}
            for crop in ('a', 'b') for codec in ('png', 'jpeg70', 'jpeg85', 'jpeg95')]
    transport = {'window': 'alcaraz_hurkacz/028@120', 'match': 'alcaraz_hurkacz',
                 'reference_files': refs, 'references': {'png_bytes': 100_000},
                 'keypoints': {'bytes': 1, 'contract': {'fps': [25, 1]}}}
    bank, old, delta = w7_billing_corrected.corrected_bank(transport)
    assert (bank, old, delta) == pytest.approx((600_000, 150_000, .5))
    rate = 14.8
    crossing = 15.05
    threshold = .6 + (.2 - .6) * math.log(crossing / 10) / math.log(2)
    means = {'full_lpips': [.5, .5], 'foreground_lpips': [.5, threshold],
             'full_dists': [.5, .5], 'foreground_dists': [.5, .4]}
    quality = {'rows': [{'id': transport['window'], 'rates': [{'scale': 4, 'total_kbps': rate}], 'means': means}]}
    points = [{'crf': 60, 'rate_kbps': 10, 'whole_lpips': .5, 'foreground_lpips': .6,
               'whole_dists': .5, 'foreground_dists': .6},
              {'crf': 50, 'rate_kbps': 20, 'whole_lpips': .5, 'foreground_lpips': .2,
               'whole_dists': .5, 'foreground_dists': .6}]
    score = {'analysis': {'by_policy': {key: points for key in w7_analysis.SVT}}}
    before = w7_analysis.analyse_window('al028', quality, score, [])
    quality['rows'][0]['rates'][0]['total_kbps'] += delta
    after = w7_analysis.analyse_window('al028', quality, score, [])
    assert before['hybrid'] == after['hybrid']
    assert before['gap']['lpips']['delta'] > 0 > after['gap']['lpips']['delta']
    assert w7_analysis.summarize_group([before, before, before])['holds']
    assert not w7_analysis.summarize_group([after, after, after])['holds']


def test_refuses_existing_output_directory(tmp_path):
    with pytest.raises(FileExistsError):
        w7_billing_corrected.analyse(tmp_path, tmp_path)
