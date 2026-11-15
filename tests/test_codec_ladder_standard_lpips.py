"""Metric contract checks without CUDA, weights, or model downloads."""
from copy import deepcopy

import pytest

from experiments.codec_ladder.__main__ import (
    LEGACY_PROTOCOL, METRIC_DEFINITIONS, METRIC_PROTOCOL, STANDARD_LPIPS,
    metric_profile, standard_lpips_scalar,
)
from experiments.codec_ladder.analysis import analyse
from experiments.codec_ladder.fleet import validate_scores


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Network:
    spatial = True
    fail = False

    def __call__(self, x, y):
        if self.fail:
            raise RuntimeError('forward failed')
        return Scalar(.25 if self.spatial else .2)


def test_distinct_pooling_and_restoration_even_on_failure():
    network = Network()
    assert network(None, None).item() == .25
    assert standard_lpips_scalar(network, None, None) == .2
    assert network.spatial is True
    network.fail = True
    with pytest.raises(RuntimeError, match='forward failed'):
        standard_lpips_scalar(network, None, None)
    assert network.spatial is True
    network.fail = False
    network.spatial = False
    assert standard_lpips_scalar(network, None, None) == .2
    assert network.spatial is False


def scores(standard=False):
    rows = []
    for arm in ('native', 'roi', 'semantic_patch', 'error_patch'):
        for i, crf in enumerate((35, 50, 65)):
            value = (.05, .15, .3)[i]
            means = {name: value for name in ('full_lpips', 'foreground_lpips', 'background_lpips',
                     'weighted_lpips', 'full_dists', 'foreground_dists', 'background_dists', 'weighted_dists')}
            means['full_rgb_psnr'] = 40. - i * 5
            if standard:
                means[STANDARD_LPIPS] = value * .9
            rows.append(dict(arm=arm, crf=crf, packet_bytes=(1000, 600, 320)[i], native_base_bytes=(1000, 600, 320)[i],
                             bound={'bytes': (980, 580, 300)[i]}, means=means,
                             frames={k: [v, v] for k, v in means.items()}, temporal={'pairs': 1, 'values': {'full_residual_mae': [0.]}}))
    result = dict(protocol=LEGACY_PROTOCOL, frames=2, available_receiver_frames=2, rows=rows,
                  domain='tennis', source_group='synthetic', annotation_provenance='synthetic',
                  metric_control={'source_self_lpips': 0., 'source_self_dists': 0.},
                  gpu_execution={'device_count': 1, 'network_parameters_cuda': True, 'maximum_reserved_bytes': 1000},
                  config={'rates': [35, 50, 65]}, region_pixels={'foreground': [10, 10], 'background': [10, 10]})
    if standard:
        result.update(metric_protocol=METRIC_PROTOCOL, metric_definitions=deepcopy(METRIC_DEFINITIONS))
        result['metric_control']['source_self_lpips_standard'] = 0.
    return result


def test_explicit_legacy_and_new_analysis_contracts():
    legacy, current = scores(), scores(True)
    assert metric_profile(legacy) is False
    assert all(validate_scores(legacy).values())
    assert all(validate_scores(current).values())
    before, after = analyse(legacy), analyse(current)
    assert STANDARD_LPIPS not in before['comparisons']['roi']['native']
    assert STANDARD_LPIPS in after['comparisons']['roi']['native']
    assert before['decisions'] == after['decisions']
    for metric, value in before['comparisons']['roi']['native'].items():
        assert after['comparisons']['roi']['native'][metric] == value
    current['rows'][0]['frames'][STANDARD_LPIPS][0] = float('nan')
    assert not all(validate_scores(current).values())
    with pytest.raises(ValueError, match='complete-frame'):
        analyse(current)


def test_profiles_cannot_silently_reinterpret_or_omit_standard_values():
    data = scores()
    del data['protocol']
    with pytest.raises(ValueError, match='unidentified'):
        validate_scores(data)
    data = scores(True)
    del data['metric_protocol']
    with pytest.raises(ValueError, match='explicit v2'):
        analyse(data)
    data = scores(True)
    del data['rows'][0]['frames'][STANDARD_LPIPS]
    assert not all(validate_scores(data).values())
    with pytest.raises(ValueError, match='complete-frame'):
        analyse(data)
    data = scores(True)
    del data['metric_control']['source_self_lpips_standard']
    assert not all(validate_scores(data).values())
