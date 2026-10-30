"""W1 job logic that runs without the real encoder, GPU or data."""
import json
import math

import numpy as np
import pytest

from experiments.tennis import w1_encode as enc
from experiments.tennis import w1_masks as masks
from experiments.tennis import w1_roi as w
from experiments.tennis import w1_score as sc


def test_search_offset_stops_at_the_encode_budget_and_publishes_each_encode(tmp_path, monkeypatch):
    monkeypatch.setenv('PS_STAGE', 'smoke')
    monkeypatch.delenv('PS_JOB_DIR', raising=False)
    seen = []

    def fake(raw, work, *, scale, frames, crf, regions, offset, cpus, outside=False, window=w.AL):
        stream = tmp_path / f'tmp{len(seen)}.ivf'
        stream.write_bytes(b'x')
        seen.append(crf)
        return {'scale': scale, 'offset': offset, 'crf': crf, 'rate_kbps': 40.0 * math.exp(-0.08 * (crf - 60)),
                '_stream': str(stream)}

    monkeypatch.setattr(enc, 'encode_point', fake)
    ledger = enc.Ledger(tmp_path / 'publish')
    result = enc.search_offset(tmp_path, tmp_path, ledger, scale=2, offset=-20, frames=24,
                               regions=np.zeros((24, 4, 4), bool), cpus=[0], max_encodes=2, target=w.TARGET_KBPS)
    assert result['status'] == 'budget' and len(seen) == 2 and seen[0] == w.start_crf(2, -20)
    assert len(ledger.rows) == 2
    assert (tmp_path / 'publish' / 'streams' / f's2_o20_crf{seen[0]}.ivf').exists()
    assert (tmp_path / 'publish' / 'encodes.json').exists()
    full = enc.search_offset(tmp_path, tmp_path, ledger, scale=2, offset=0, frames=240,
                             regions=np.zeros((240, 4, 4), bool), cpus=[0], max_encodes=8, target=w.TARGET_KBPS)
    assert full['status'] == 'complete' and len(full['selected']) == 3


def test_projection_inflates_the_longest_chain_and_the_controls():
    summary = {'rows': [{'wall_seconds': 12.0, 'frames': 24}], 'setup_seconds': 30.0, 'control_seconds': 6.0,
               'has_controls': True}
    result = enc.projection(summary, 8, 3000.0)
    assert result['projected_full_seconds'] == pytest.approx(1.35 * (30 + 0.5 * 240 * 8 + 60))
    assert result['under_allowance'] is True
    assert enc.projection(summary, 8, 900.0)['under_allowance'] is False


def test_masks_pack_round_trip():
    mask = np.zeros((1080, 1920), bool)
    mask[100:300, 500:900] = True
    assert (masks.unpack(np.frombuffer(masks.pack_bits(mask), np.uint8)) == mask).all()


def test_score_helpers_select_three_per_policy_and_flag_a_bad_control(tmp_path):
    (tmp_path / 'streams').mkdir()
    rows = [{'scale': 2, 'offset': -20, 'crf': c, 'stream': f'streams/s2_o20_crf{c}.ivf', 'rate_kbps': r,
             'size': [960, 544], 'decoded_sha256': 'a' * 64} for c, r in ((68, 24.0), (69, 22.0), (70, 20.0), (67, 27.0))]
    rows.append({'scale': 2, 'offset': 0, 'crf': 63, 'stream': 'streams/control_s2_crf63.ivf', 'rate_kbps': 33.7,
                 'size': [960, 544], 'decoded_sha256': 'b' * 64})
    summary = {'scale': 2, 'searches': [{'scale': 2, 'offset': -20, 'selected': [68, 69, 70]}, {'scale': 2, 'offset': 0, 'selected': None}],
               'controls': [{**rows[-1], 'scale': 2}]}
    (tmp_path / 'encodes.json').write_text(json.dumps(rows))
    (tmp_path / 'encode.json').write_text(json.dumps(summary))
    points = sc.selected_points(tmp_path)
    assert [p['name'] for p in points] == ['s2_o20_crf68', 's2_o20_crf69', 's2_o20_crf70', 'control_s2_crf63']
    assert points[-1]['kind'] == 'control'
    hybrid = {name: np.zeros((6, 3)) for name in sc.SERIES}
    values = np.zeros((4, 3))
    assert max(sc.control_errors(values, hybrid, 2, 3).values()) == 0
    values[1, 2] = 2e-4
    assert sc.control_errors(values, hybrid, 2, 3)['foreground_lpips'] == pytest.approx(2e-4)


def test_assemble_applies_the_rule_to_scored_points():
    points = [{'kind': 'roi', 'policy': 's2_o0', 'name': f's2_o0_crf{c}', 'crf': c, 'rate_kbps': r}
              for c, r in ((69, 21.0), (68, 24.0))]
    scores = {p['name']: np.array([[0.23], [0.61], [0.16], [0.45]]) for p in points}
    result = sc.assemble(points, scores, (1, 2))
    assert result['at_target']['s2_o0']['whole_lpips'] == pytest.approx(0.23)
    assert result['at_target']['s1_o0'] is None
    assert result['decision']['outcome'] == 'A'


def test_roicheck_text_helpers_round_trip_and_psnr():
    from experiments.tennis import w1_roicheck as rc

    region = np.zeros((2, 128, 128), bool)
    region[0, 70, 70] = True
    inside = w.roi_text(region, 128, 128, -40)
    outside = rc.outside_text(region, 128, 128, 20)
    g_in, g_out = rc.parse_text(inside, 2, 2), rc.parse_text(outside, 2, 2)
    assert (g_in[0] != 0).tolist() == [[False, False], [False, True]]
    assert (g_out[0] != 0).tolist() == [[True, True], [True, False]]
    assert (g_in[1] == 0).all() and (g_out[1] == 20).all()
    with pytest.raises(ValueError):
        rc.parse_text('1 0 0 0 0', 2, 2)
    assert rc.psnr(0.0, 10) == 99.0 and rc.psnr(1.0, 0) is None
    assert rc.psnr(255.0 ** 2, 1) == pytest.approx(0.0)


def test_outside_convention_and_search_start():
    region = np.zeros((1, 128, 128), bool)
    region[0, 70, 70] = True
    assert w.roi_text(region, 128, 128, -20, outside=True).split()[1:] == ['20', '20', '20', '0']
    assert w.roi_text(region, 128, 128, -20).split()[1:] == ['0', '0', '0', '-20']
    assert w.start_crf(2, -40, outside=True) == 27 and w.start_crf(2, -40) == 67 and w.start_crf(2, 0, outside=True) == 67


def test_assemble_carries_the_plain_policy_from_an_earlier_run():
    carried = {'s2_o0': [{'crf': c, 'rate_kbps': r, 'whole_lpips': 0.2, 'foreground_lpips': 0.5, 'whole_dists': 0.1,
                          'foreground_dists': 0.4} for c, r in ((66, 22.0), (65, 24.0))]}
    points = [{'kind': 'roi', 'policy': 's2_o20', 'name': f's2_o20_crf{c}', 'crf': c, 'rate_kbps': r}
              for c, r in ((50, 21.0), (49, 24.0))]
    scores = {p['name']: np.array([[0.3], [0.4], [0.2], [0.3]]) for p in points}
    result = sc.assemble(points, scores, (2,), carried)
    assert result['at_target']['s2_o0'] is not None and result['at_target']['s2_o20'] is not None


def test_maxbg_command_modes_and_search_variable(tmp_path):
    from experiments.tennis import w1_maxbg as mb

    aq0 = mb.command('aq0', tmp_path / 'a.yuv', tmp_path / 'a.ivf', 960, 544, 240, None)
    cqp = mb.command('cqp', tmp_path / 'a.yuv', tmp_path / 'a.ivf', 960, 544, 240, tmp_path / 'roi.txt')
    assert aq0[aq0.index('--crf') + 1] == '70' and aq0[aq0.index('--aq-mode') + 1] == '0' and '--roi-map-file' not in aq0
    assert '--crf' not in cqp and cqp[cqp.index('--cqp') + 1] == '70' and '--roi-map-file' in cqp
    assert w.maxbg_roi_value(70) == 0 and w.maxbg_roi_value(30) == -40 and w.maxbg_k(-63) == 7
    assert w.policy_name(2, w.MAXBG) == 's2_mb'


def test_maxbg_search_stays_inside_the_offset_range():
    known: dict[int, float] = {}
    for _ in range(30):
        status, k = w.next_crf(known, w.TARGET_KBPS, 44, lo=mb_low(), hi=70)
        if k is None:
            break
        assert 7 <= k <= 70
        known[k] = 60.0 * math.exp(-0.05 * (k - 20))
    assert status == 'complete'


def mb_low():
    return w.MAXBG_BASE_Q - 63
