import hashlib
import random
import tarfile

import numpy as np
import pytest

from experiments.tennis import w2_pose_rate as w2
from experiments.tennis.k6b_rate_streams import decode_keypoints, encode_keypoints
from experiments.tennis.w2_pose_rate import (
    Decoder,
    Encoder,
    Setting,
    best_at_levels,
    code_signed,
    decode_window,
    encode_window,
    extent_height,
    interpolate,
    key_times,
    load_packets,
    player_error,
    run_setting,
)


def walk(n=40, seed=0, step=3):
    """A smooth-ish integer body with 17 joints, visible about 85% of the time."""
    rng = np.random.default_rng(seed)
    base = np.rint(np.stack([np.linspace(500, 560, 17), np.linspace(300, 420, 17)], 1))
    drift = np.cumsum(rng.integers(-step, step + 1, size=(n, 1, 2)), axis=0)
    xy = base[None] + drift + rng.integers(-1, 2, size=(n, 17, 2))
    vis = (rng.random((n, 17)) < 0.85).astype(float)
    return {'t': np.arange(n), 'kp': np.concatenate([xy, vis[:, :, None]], 2).astype(np.float64)}


def test_range_coder_roundtrip_random_bits_and_contexts():
    rng = random.Random(1)
    for length in (0, 1, 2, 7, 500, 5000):
        bits = []
        for _ in range(length):
            ctx = rng.randrange(3)
            bits.append((ctx, int(rng.random() < [0.02, 0.5, 0.97][ctx])))
        enc = Encoder()
        for ctx, b in bits:
            enc.bit(ctx, b)
        data = enc.finish()
        dec = Decoder(data)
        assert [dec.bit(ctx) for ctx, _ in bits] == [b for _, b in bits]


def test_skewed_bits_cost_less_than_one_bit_each():
    enc = Encoder()
    for _ in range(4000):
        enc.bit('c', 0)
    assert len(enc.finish()) < 40  # the adaptive model learns the skew


def test_signed_integers_roundtrip_including_escape_values():
    values = [0, 1, -1, 2, -2, 14, 15, -16, 17, 100, -1000, 70000, 0, 0, 3]
    enc = Encoder()
    for v in values:
        code_signed(enc, v, 'k', 1, 2)
    dec = Decoder(enc.finish())
    assert [code_signed(dec, None, 'k', 1, 2) for _ in values] == values


@pytest.mark.parametrize('predictor', ['previous', 'velocity'])
def test_lossless_setting_reproduces_input_exactly(predictor):
    unit = walk()
    enc = encode_window([unit], Setting(1, 1, predictor))
    (out,) = decode_window(enc.blob)
    assert np.array_equal(out['t'], unit['t'])
    assert np.array_equal(out['kp'][:, :, :2], unit['kp'][:, :, :2])
    assert np.array_equal(out['kp'][:, :, 2] >= 0.5, unit['kp'][:, :, 2] >= 0.5)


@pytest.mark.parametrize('setting', [
    Setting(4, 1, 'velocity'), Setting(8, 2, 'previous', 'cubic'), Setting(16, 4, 'velocity', 'cubic', 'keys'),
    Setting(32, 8, 'previous', 'linear', 'keys'), Setting(2, 3, 'velocity', 'linear', 'all'),
])
def test_decoder_matches_encoder_expectation_for_lossy_settings(setting):
    units = [walk(41, 1), walk(41, 2)]
    enc = encode_window(units, setting)
    out = decode_window(enc.blob)
    for e, d in zip(enc.expected, out):
        assert np.array_equal(e['t'], d['t']) and np.array_equal(e['kp'], d['kp'])


def test_gapped_runs_and_header_accounting():
    unit = walk(30, 3)
    unit['t'] = np.concatenate([np.arange(0, 12), np.arange(20, 38)])
    enc = encode_window([unit], Setting(1, 1))
    (out,) = decode_window(enc.blob)
    assert np.array_equal(out['t'], unit['t'])
    assert np.array_equal(out['kp'][:, :, :2], unit['kp'][:, :, :2])
    # 5 fixed bytes, run count 2, then (start, length) twice: nothing else precedes the payload.
    assert enc.header_bytes == 5 + 1 + 4 * 1 + 0
    assert enc.blob[:enc.header_bytes][:1] == bytes([w2.MAGIC])


def test_rate_is_produced_bytes_including_header():
    units = [walk(60, 4)]
    r = run_setting(units, Setting(4, 4), seconds=2.4)
    enc = encode_window(units, Setting(4, 4))
    assert r['bytes'] == len(enc.blob) and r['kbps'] == len(enc.blob) * 8 / 2.4 / 1000
    assert r['header_bytes'] == enc.header_bytes and r['round_trip_exact']
    # the model's ideal cost agrees with the produced payload to within a few bytes
    estimate = sum(enc.cost_bits.values()) / 8
    assert abs(estimate - (len(enc.blob) - enc.header_bytes)) < 6


def test_exact_recoding_of_the_original_packet_format():
    """Keys every 4 frames on the 4 px grid + linear interpolation = the original decoder, bit for bit."""
    rng = np.random.default_rng(5)
    n = 24
    samples = np.cumsum(rng.integers(-3, 4, size=((n + 3) // 4, 17, 2)), axis=0) * 4 + 800
    kp = np.zeros((n, 17, 3))
    for i in range((n + 3) // 4):
        kp[4 * i:4 * i + 4, :, :2] = samples[i]
    vis = rng.random((n, 17)) < 0.8
    kp[:, :, 2] = vis
    units = [{'name': 'u', 't': np.arange(n), 'kp': kp}]
    meta = {'native_frames': n, 'fps': [25, 1]}
    _, ref = decode_keypoints(encode_keypoints(meta, units))
    r = run_setting(ref, Setting(4, 4, 'velocity', 'linear'), seconds=n / 25)
    assert r['players'][0]['mean'] < 1e-12 and r['players'][0]['p95'] < 1e-12  # float roundoff only
    assert r['players'][0]['visibility_mismatch'] == 0 and r['players'][0]['jitter_ratio'] == pytest.approx(1)


def test_key_times_always_include_last_frame():
    assert key_times(10, 4) == [0, 4, 8, 9]
    assert key_times(9, 4) == [0, 4, 8]
    assert key_times(1, 4) == [0]
    with pytest.raises(ValueError):
        key_times(0, 4)


def test_interpolation_linear_cubic_and_endpoints():
    n = 13
    keys_t = key_times(n, 4)
    ramp = np.zeros((len(keys_t), 17, 2))
    ramp[:] = np.array(keys_t, float)[:, None, None] * 2 + 5  # a straight line in time
    for interp in ('linear', 'cubic'):
        out = interpolate(ramp, n, Setting(1, 4, interp=interp))
        assert np.allclose(out[:, 0, 0], np.arange(n) * 2 + 5)  # both reproduce a line exactly
    bend = ramp.copy()
    assert len(keys_t) == 4
    bend[:, :, 0] = np.array([0., 8., 0., 8.])[:, None]
    lin = interpolate(bend, n, Setting(1, 4, interp='linear'))
    cub = interpolate(bend, n, Setting(1, 4, interp='cubic'))
    assert np.array_equal(lin[keys_t], bend) and np.array_equal(cub[keys_t], bend)  # keys are hit exactly
    assert lin[2, 0, 0] == 4.0 and cub[2, 0, 0] != lin[2, 0, 0]


def test_player_error_normalization_and_jitter():
    ref = walk(30, 6)
    r = ref['kp']
    height = extent_height(r)
    assert height > 0
    shifted = {'t': ref['t'], 'kp': r.copy()}
    shifted['kp'][:, :, 0] += 3.0  # every joint moves 3 px sideways
    e = player_error(ref, shifted)
    assert e['mean'] == pytest.approx(3.0 / height) and e['p95'] == pytest.approx(3.0 / height)
    assert e['jitter_ratio'] == pytest.approx(1.0)  # a rigid shift leaves the second difference alone
    smooth = {'t': ref['t'], 'kp': r.copy()}
    smooth['kp'][:, :, :2] = np.mean(r[:, :, :2], axis=0)  # a frozen pose has no jitter
    assert player_error(ref, smooth)['jitter_dec'] == 0
    with pytest.raises(ValueError):
        player_error(ref, {'t': ref['t'][1:], 'kp': r[1:]})


def test_best_at_levels_picks_lowest_rate_among_passing():
    def row(label, nbytes, mean, p95):
        return {'label': label, 'bytes': nbytes, 'kbps': nbytes / 10,
                'players': [{'mean': mean, 'p95': p95, 'visibility_mismatch': 0.0}]}
    rows = {'Al': [row('a', 100, 0.001, 0.002), row('b', 50, 0.006, 0.012), row('c', 20, 0.02, 0.06)]}
    out = best_at_levels(rows)
    assert out['0.33%']['Al']['label'] == 'a'
    assert out['0.66%']['Al']['label'] == 'b'
    assert best_at_levels({'Al': [row('c', 20, 0.02, 0.06)]})['0.33%']['Al'] is None
    assert out['1.32%']['Al']['label'] == 'b'
    assert out['2.64%']['Al']['label'] == 'b'  # c's p95 exceeds twice the level


def test_decoder_rejects_foreign_or_truncated_streams():
    with pytest.raises(ValueError):
        decode_window(b'')
    with pytest.raises(ValueError):
        decode_window(bytes([0x00, 0, 4, 4, 1, 1, 0, 10]))
    good = encode_window([walk(10)], Setting(4, 2)).blob
    with pytest.raises(ValueError):
        decode_window(good[:7])


def test_setting_validation():
    for bad in ({'q': 0}, {'key': 0}, {'predictor': 'x'}, {'interp': 'spline'}, {'vis': 'none'}):
        with pytest.raises(ValueError):
            Setting(**bad)


def test_archive_sha_checked_before_members_are_read(tmp_path):
    path = tmp_path / 'published.tar'
    with tarfile.open(path, 'w') as tar:
        info = tarfile.TarInfo('packets/x')
        info.size = 1
        import io
        tar.addfile(info, io.BytesIO(b'x'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() != w2.TRANSPORT_SHA256
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_packets(path)
