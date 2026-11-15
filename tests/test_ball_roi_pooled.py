from pathlib import Path

import pytest

from experiments.tennis.ball_roi_pooled import analyse, fixed_interval

ROOT = Path(__file__).resolve().parents[1] / 'docs/experiments'
PRIMARY = ROOT / 'submission-ladder-anchor-sensitivity-20261114.json'
ROI = ROOT / 'submission-ladder-ball-roi-scores-20261115.json'


def test_original_price_metric_join_and_stronger_reference_keep_intervals_distinct():
    result = analyse(PRIMARY.read_bytes(), ROI.read_bytes())
    row = result['results']['ROI_LPIPS']
    assert row['original_six_point_comparison']['bd_rate']['hull_pchip'] == pytest.approx(-.7696525232176281)
    assert row['with_three_ROI_points']['bd_rate']['hull_pchip'] == pytest.approx(-.6094519396526397)
    assert row['with_ROI_matched_interval_hull_pchip'] == pytest.approx(-.7107145355543578)
    assert row['matched_interval'] != row['with_three_ROI_points']['common_quality_interval']
    with pytest.raises(ValueError, match='within common'):
        fixed_interval(row['with_three_ROI_points'], [-100., 100.])
    assert 'WASB_recall10' not in result['results']


def test_cannot_silently_modify_accepted_primary_or_new_ROI_scores():
    with pytest.raises(ValueError, match='evidence hash'):
        analyse(PRIMARY.read_bytes() + b'\n', ROI.read_bytes())
    with pytest.raises(ValueError, match='evidence hash'):
        analyse(PRIMARY.read_bytes(), ROI.read_bytes() + b'\n')
