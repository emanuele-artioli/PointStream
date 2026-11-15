import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.jobs import inbox
from experiments.tennis import ball_pixel_support_control as control


def test_paid_spatial_masks_and_absence():
    base = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    tile = np.full((2, 64, 64, 3), 17, dtype=np.uint8)
    positions = [[4, 6], None]
    full = control.compose(base, tile, positions, 'whole32')
    centre = control.compose(base, tile, positions, 'centre16')
    annulus = control.compose(base, tile, positions, 'annulus')
    assert np.count_nonzero(full[0]) == 32 * 32 * 3
    assert np.count_nonzero(centre[0]) == 16 * 16 * 3
    assert np.count_nonzero(annulus[0]) == (32 * 32 - 16 * 16) * 3
    assert np.array_equal(full[0], centre[0] + annulus[0])
    assert not full[1].any() and not centre[1].any() and not annulus[1].any()
    with pytest.raises(ValueError):
        control.compose(base, tile, [[5, 6], None], 'whole32')


def test_source_point_coverage_uses_half_open_regions_and_none():
    positions = [[10, 20]] * 5 + [None]
    labels = [(1, 18., 28.), (1, 33.999, 43.999), (1, 34., 44.),
              (1, 41.999, 51.999), (1, 42., 52.), (1, 20., 30.)]
    coverage = control.point_coverage(labels, positions)
    assert coverage['frame_status'] == ['centre16', 'centre16', 'annulus', 'annulus', 'outside32', 'sender_none']
    assert coverage['counts'] == {'centre16': 2, 'annulus': 2, 'outside32': 1, 'sender_none': 1}


def test_full_metric_control_rejects_perturbations_and_prefix():
    data = json.loads(Path('docs/experiments/submission-ladder-ball-pixel-oracle-scores-20261115.json').read_bytes())
    row = data['rows'][0]
    assert control.compare_full(row['frames'], row['frames'], row['full_lpips_standard'], row['full_lpips_standard'])['passed']
    bad = copy.deepcopy(row['frames'])
    bad[3]['crop_dists'] += 0.01
    with pytest.raises(ValueError):
        control.compare_full(bad, row['frames'], row['full_lpips_standard'], row['full_lpips_standard'])
    with pytest.raises(ValueError):
        control.compare_full(row['frames'][:16], row['frames'], row['full_lpips_standard'], row['full_lpips_standard'])
    wrong_standard = row['full_lpips_standard'].copy()
    wrong_standard[0] += 0.01
    with pytest.raises(ValueError):
        control.compare_full(row['frames'], row['frames'], wrong_standard, row['full_lpips_standard'])


def test_ratios_undefined_when_no_quality_improvement():
    assert control.ratio(.5, .3, .2) == pytest.approx(2 / 3)
    assert control.ratio(.2, .1, .2) is None
    assert control.ratio(.1, .1, .2) is None


def test_analysis_survives_json_roundtrip_without_weakening_numeric_comparison():
    saved = json.loads(Path('docs/experiments/submission-ladder-ball-pixel-oracle-scores-20261115.json').read_bytes())
    rows = [{'crf': row['crf'], 'packet_bytes': row['coded_bytes'],
             'point_coverage': {'counts': {}},
             **{arm: {'frames': row['frames']} for arm in ('whole32', 'centre16', 'annulus')}}
            for row in saved['rows']]
    analysis = control.analyse(rows, saved['historical_points'])
    assert json.loads(json.dumps(analysis)) == analysis
    changed = copy.deepcopy(analysis)
    changed['points'][0]['retained_reduction']['crop_dists']['annulus'] += .01
    assert changed != analysis


def test_spec_is_frozen_and_full_smoke_control_is_explicit():
    s = inbox.validate_spec(control.spec('2026-11-16T12:00:00+00:00'))
    assert s['scale']['frames'] == {'smoke': 16, 'full': 48}
    assert s['smoke']['seconds'] == 450 and s['full']['seconds'] == 1800 and s['budget_seconds'] == 2430
    assert s['cpu_threads'] == 1 and s['gpu_memory_mib'] == 8000 and s['gpu_sharing']['margin_mib'] == 8192
    staged = {x['name']: x['sha256'] for x in s['staged_inputs']}
    assert staged['paid_archive'] == control.PAID_ARCHIVE and staged['saved_scores'] == control.SCORE_ARCHIVE
    assert control.gpu_cost(394, 1, 16)['passed'] is False
