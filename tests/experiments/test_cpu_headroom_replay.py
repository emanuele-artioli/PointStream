import io
import numpy as np
import pytest

from experiments.headroom.cpu_replay import (
    compare_curves, group_sensitivity, score_frame, summarize_scores, y4m_frames,
)


def test_y4m_rejects_truncation_instead_of_shortening_denominator():
    data = b"YUV4MPEG2 W4 H4 F24:1 C420mpeg2\nFRAME\n" + bytes(23)
    with pytest.raises(ValueError, match="truncated"):
        list(y4m_frames(io.BytesIO(data)))


def test_y4m_preserves_frame_tags_and_strict_pixel_geometry():
    data = b"YUV4MPEG2 W4 H4 F24:1 C420mpeg2\nFRAME Xtest\n" + bytes(range(16)) + bytes(8)
    decoded = list(y4m_frames(io.BytesIO(data)))
    assert len(decoded) == 1
    assert decoded[0].tolist() == np.arange(16).reshape(4,4).tolist()


def test_psnr_aggregation_distinguishes_frame_mean_and_pixel_pool():
    mask = np.eye(4,dtype=bool)
    source = np.zeros((4,4),np.uint8)
    rows = [score_frame(source,np.full_like(source,x),mask) for x in [4,16]]
    result = summarize_scores(rows)
    assert result['frames'] == 2
    assert result['foreground']['selected_pixels'] == 8
    assert result['whole']['pooled_pixel_mse'] == 136
    assert result['whole']['mean_frame_psnr_db'] != result['whole']['pooled_pixel_psnr_db']
    identical = summarize_scores([score_frame(source,source,mask), rows[0]])
    assert identical['whole']['mean_frame_psnr_db'] is None
    assert identical['whole']['identical_frames'] == 1


def test_bd_rate_uses_common_support_and_rejects_extrapolation():
    anchor = [(20,100),(30,200),(40,400)]
    candidate = [(q,b*1.2) for q,b in anchor]
    result = compare_curves(anchor,candidate)
    assert result['bd_rate_percent'] == pytest.approx(20)
    assert compare_curves(anchor,[(45,100),(50,200),(55,400)])['bd_rate_percent'] is None


def test_group_sensitivity_retains_duplicate_scene_weights():
    rows = {'a/x': {'plate_vs_original':{'saving':.1}},
            'a/y': {'plate_vs_original':{'saving':.3}},
            'b/x': {'plate_vs_original':{'saving':.5}}}
    result = group_sensitivity({'fg':{'av1':rows}},draws=1000)['codecs']['av1']
    assert result['scene_mean_percent'] == pytest.approx(30)
    assert result['equal_source_mean_percent'] == pytest.approx(35)
    assert result['source_groups'] == 2
    assert result['leave_one_source_out_scene_mean_range'] == [20,50]
