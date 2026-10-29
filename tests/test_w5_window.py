"""W5: the W1 grid parametrised by window, and the real-base judge's early-stop switch."""
import numpy as np
import pytest

from experiments.tennis import k6b_real_base_judge as judge
from experiments.tennis import w1_roi as w
from experiments.tennis import w1_score as sc


def test_med_window_matches_the_recorded_med_constants():
    med = w.MED
    assert med.id == 'medvedev_struff/014@194' and med.fps == 50 and med.source_start == 74
    assert med.duration == pytest.approx(4.8)
    # plain base files of the real-base base manifest, as rates (W3 table: 19.557, 34.140, 80.253 kbps)
    rates = [w.rate_kbps(med.control_ivf_bytes[s], w.FRAMES, med.fps) for s in (8, 4, 2)]
    assert rates == pytest.approx([19.557, 34.140, 80.253], abs=1e-3)
    assert w.WINDOWS == {'al': w.AL, 'med': w.MED}


def test_al_window_keeps_the_w1_constants():
    assert w.AL.control_ivf_bytes is w.CONTROL_IVF_BYTES and w.AL.source_sha == w.SOURCE_SHA
    assert w.start_crf(2, 0) == 67 and w.start_crf(4, -20, True) == 40 and w.start_crf(2, 0, window=w.AL) == 67
    assert w.rebilled_kbps(w.TARGET_KBPS, 4236, 2701) == pytest.approx(w.secondary_kbps(2701))


def test_med_rebill_subtracts_the_keypoint_saving_over_4_8_seconds():
    saving = (3542 - 2193) * 8 / 4.8 / 1000
    assert saving == pytest.approx(2.2483, abs=1e-4)
    assert w.rebilled_kbps(42.5, 3542, 2193, w.MED) == pytest.approx(42.5 - saving)
    assert w.start_crf(2, 0, window=w.MED) == 70 and w.start_crf(4, -40, True, w.MED) == 20


def test_rate_follows_the_window_frame_rate():
    assert w.rate_kbps(12000, 240, 50) == pytest.approx(2 * w.rate_kbps(12000, 240, 25))


def _quality(kps_bytes=3542):
    means = {'full_lpips': [0, 0, 0, 0.3, 0, 0], 'foreground_lpips': [0, 0, 0, 0.4, 0, 0],
             'full_dists': [0, 0, 0, 0.2, 0, 0], 'foreground_dists': [0, 0, 0, 0.25, 0, 0]}
    rates = [{'scale': 8, 'total_kbps': 30.0, 'kps_bytes': kps_bytes}, {'scale': 4, 'total_kbps': 42.5, 'kps_bytes': kps_bytes}]
    return {'rows': [{'id': 'alcaraz_hurkacz/031@129', 'means': {}, 'rates': []}, {'id': w.MED.id, 'means': means, 'rates': rates}]}


def test_hybrid_point_reads_scale_4_method_and_rebills_kps2():
    point = sc.hybrid_point(_quality(), w.MED)
    assert point['hybrid']['lpips'] == {'whole': 0.3, 'foreground': 0.4}
    assert point['hybrid']['dists'] == {'whole': 0.2, 'foreground': 0.25}
    assert point['kps1_kbps'] == 42.5 and point['kps2_kbps'] == pytest.approx(42.5 - 1349 * 8 / 4.8 / 1000)
    with pytest.raises(ValueError):
        sc.hybrid_point(_quality(4236), w.MED)


def test_assemble_applies_the_given_target_and_hybrid():
    points = [{'kind': 'roi', 'policy': 's4_o0', 'name': f'p{i}', 'crf': 58 + i, 'rate_kbps': r}
              for i, r in enumerate((44.0, 41.0))]
    scores = {'p0': np.full((4, 3), 0.3), 'p1': np.full((4, 3), 0.35)}
    hybrid = {'lpips': {'whole': 0.4, 'foreground': 0.4}, 'dists': {'whole': 0.2, 'foreground': 0.2}}
    out = sc.assemble(points, scores, (4,), None, 42.5, hybrid)
    assert out['target_kbps'] == 42.5 and out['at_target']['s4_o0'] is not None
    assert out['decision']['dominating']['lpips'] == ['s4_o0'] and out['decision']['dominating']['dists'] == []
    far = sc.assemble(points, scores, (4,), None, 60.0, hybrid)
    assert far['at_target']['s4_o0'] is None and far['decision']['outcome'] == 'A'


def _item(lost):
    comparisons = {f'{f}_{m}': {'scale4': {'won': f'{f}_{m}' not in lost}} for f in ('full', 'foreground')
                   for m in ('lpips', 'dists')}
    return {'comparisons': comparisons}


def test_early_stop_fires_on_two_losses_in_one_family_unless_disabled():
    both = _item({'full_lpips', 'full_dists'})
    assert judge.early_stop_applies(both, False) is True
    assert judge.early_stop_applies(both, True) is False
    assert judge.early_stop_applies(_item({'full_lpips', 'foreground_dists'}), False) is False
