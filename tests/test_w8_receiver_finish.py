import numpy as np
import pytest

from experiments.tennis import w8_receiver_finish as finish


def test_pixel_fallback_is_per_channel_and_checks_every_tick(monkeypatch):
    monkeypatch.setattr(finish, 'TICKS', np.arange(2))
    golden = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    candidate = golden.copy()
    candidate[1, 1, 1, 2] = 1
    result = finish._compare_rgb(candidate, golden, 'frame')
    assert result == {'exact': False, 'max_per_frame': [0, 1]}
    candidate[1, 1, 1, 2] = 2
    with pytest.raises(ValueError, match='one channel'):
        finish._compare_rgb(candidate, golden, 'frame')


def test_metrics_reject_single_bad_tick_even_when_mean_passes():
    original = {name: np.zeros(80) for name in finish.METRICS}
    candidate = {name: value.copy() for name, value in original.items()}
    candidate['foreground_dists'][17] = 0.00101
    with pytest.raises(ValueError, match='tolerance'):
        finish._compare_scores(candidate, original)
    candidate['foreground_dists'][17] = 0.001
    assert finish._compare_scores(candidate, original)['foreground_dists']['max_tick_difference'] == 0.001
    candidate['foreground_dists'][17] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        finish._compare_scores(candidate, original)


def test_isolation_requires_real_denials_and_loopback():
    good = {'landlock_abi': 2, 'pre_sandbox_threads': 1, 'extra_fds': False,
            'network_interfaces': ['lo'], 'namespaces': {'user': 'user:[1]', 'net': 'net:[2]'},
            'denials': {'/protected': 13}}
    finish._isolation(good)
    for change in ({'denials': {}}, {'network_interfaces': ['lo', 'eth0']}, {'extra_fds': True}):
        with pytest.raises(ValueError, match='isolation'):
            finish._isolation(good | change)


def test_native_command_is_bound_to_binary_ivf_and_fresh_output(tmp_path, monkeypatch):
    import hashlib

    binary, ivf, output = (tmp_path / name for name in ('ffmpeg', 'video.ivf', 'base.yuv'))
    binary.write_bytes(b'ffmpeg')
    ivf.write_bytes(b'ivf')
    monkeypatch.setattr(finish, 'FFMPEG_SHA', hashlib.sha256(b'ffmpeg').hexdigest())
    monkeypatch.setattr(finish, 'IVF_SHA', hashlib.sha256(b'ivf').hexdigest())
    argv = finish.native_argv(binary, ivf, output)
    assert argv[argv.index('-c:v') + 1] == 'libdav1d'
    assert '-n' in argv
    output.touch()
    with pytest.raises(ValueError, match='already exists'):
        finish.native_argv(binary, ivf, output)


def test_masks_require_exact_boolean_full_coverage():
    empty = np.broadcast_to(np.bool_(False), (80, 784, 512))
    assert finish._compare_masks(empty, empty)['exact'] is True
    wrong = np.zeros((80, 784, 512), dtype=bool)
    wrong[79, 783, 511] = True
    with pytest.raises(ValueError, match='mask mismatch'):
        finish._compare_masks(wrong, empty)
    with pytest.raises(ValueError, match='mask mismatch'):
        finish._compare_masks(empty.astype(np.uint8), empty)
