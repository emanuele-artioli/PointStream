from fractions import Fraction

import numpy as np
import pytest

from experiments.jobs import inbox
from experiments.tennis import fresh_ball_sender as sender
from src.pointstream import ball, ball_stream


class AbsentTracker:
    def run(self, frames):
        return [(False, 0., 0.) for _ in frames]


def test_generic_sender_six_bases_keeps_absence_and_native_fps():
    source = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    bases = [(key, source.copy()) for key in sender.POINTS]
    for fps in (Fraction(25, 1), Fraction(60, 1)):
        result = sender.derive(source, bases, fps, AbsentTracker())
        assert result['fps'] == [fps.numerator, fps.denominator]
        assert [tuple((p['codec'], p['point'])) for p in result['points']] == list(sender.POINTS)
        for row in result['points']:
            assert row['centres'] == [None] * 3 and row['origins'] == [None] * 3
            track, _ = ball.decode_frames(bytes.fromhex(row['ball_payload_hex']), 3, 64, 64)
            assert track == [None] * 3
    with pytest.raises(ValueError):
        sender.derive(source, bases[:-1], Fraction(25), AbsentTracker())
    with pytest.raises(ValueError):
        sender.derive(source, list(reversed(bases)), Fraction(25), AbsentTracker())


def test_guard_disables_harmful_fitted_track(monkeypatch):
    source = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    bases = [(key, source.copy()) for key in sender.POINTS]
    appearance = ball.Appearance((255, 255, 255), 4, 0, 3)
    monkeypatch.setattr(sender.old, 'fit_streak', lambda *args: [
        {'track': [(20, 20), (20, 20)], 'appearances': [appearance, appearance], 'delta_se': -1.0}
        for _ in sender.POINTS])
    result = sender.derive(source, bases, Fraction(25), AbsentTracker())
    assert all(row['guard_disabled'] and row['centres'] == [None, None] for row in result['points'])


def test_full_parity_requires_original_positions_and_quantized_origins():
    n = 48
    actual = {'frames': n, 'points': []}
    packets = {}
    oracle = {'points': []}
    for codec, point in sender.POINTS:
        track = [None] * n
        payload = ball.encode_frames(track, [None] * n, 1280, 720)
        row = {'codec': codec, 'point': point, 'centres': [None] * n, 'origins': [None] * n,
               'ball_payload_hex': payload.hex()}
        actual['points'].append(row)
        if codec == 'svt':
            stream = ball_stream.Stream(1280, 720, Fraction(30), n, 'svt-av1-ivf', 'svt', b'native', payload)
            packets[(codec, point)] = ball_stream.pack(stream)
            oracle['points'].append({'crf': point, 'positions': [{'position': None, 'present': False} for _ in range(n)]})
    result = sender.parity(actual, packets, oracle)
    assert result['passed'] and len(result['checks']) == 3
    actual['points'][3]['origins'][0] = [0, 0]
    assert not sender.parity(actual, packets, oracle)['passed']
    actual['frames'] = 16
    assert sender.parity(actual, packets, oracle)['passed'] is None


def test_frozen_parity_spec_has_correct_resources_and_no_promotion_sweep():
    s = inbox.validate_spec(sender.spec('2026-11-16T12:00:00+00:00'))
    assert s['scale']['frames'] == {'smoke': 16, 'full': 48}
    assert s['cpu_threads'] == 1 and s['gpu_memory_mib'] == 8000 and s['gpu_sharing']['margin_mib'] == 8192
    assert s['smoke']['seconds'] == 450 and s['full']['seconds'] == 900 and s['budget_seconds'] == 1530
    staged = {x['name']: x for x in s['staged_inputs']}
    assert staged['producer']['sha256'] == sender.common.PRODUCER
    assert staged['model_code']['sha256'] == sender.MODEL_CODE and staged['model_code']['extract']
    assert staged['model_weight']['sha256'] == sender.MODEL_WEIGHT


def test_validation_binds_stage_frames_and_dispatch_identity():
    dispatch = {'code': {'source_sha256': 'source', 'snapshot_sha256': 'snapshot'},
                'staged_environment': {'sha256': sender.isolated_runtime()}}
    identity = {'source_sha256': 'source', 'snapshot_sha256': 'snapshot',
                'environment_sha256': sender.isolated_runtime()}
    row = {'frames': 16, 'promotable': False, 'execution_identity': identity}
    assert all(sender.binding_checks({'frames': 16}, row, dispatch, 'smoke').values())
    assert not all(sender.binding_checks({'frames': 16}, row, dispatch, 'full').values())
    assert not all(sender.binding_checks({'frames': 48}, row, dispatch, 'full').values())
    full = {**row, 'frames': 48, 'promotable': True}
    assert all(sender.binding_checks({'frames': 48}, full, dispatch, 'full').values())
    full['execution_identity'] = {**identity, 'source_sha256': 'wrong'}
    assert not sender.binding_checks({'frames': 48}, full, dispatch, 'full')['dispatch_identity']
