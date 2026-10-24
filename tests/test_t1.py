"""T1 must bill transmitted frames even when registration excludes most of them."""
import json

import numpy as np
import pytest

from experiments.tennis import t1
from src.segmentation.masks import ClipMasks


def test_measure_bills_whole_window_and_marks_missing_ball(tmp_path, monkeypatch):
    root = tmp_path / 'windows' / 'match__clip@120'
    root.mkdir(parents=True)
    (root / 'features.json').write_text(json.dumps({'aliases': {}}))
    entry = {'window': {'start': 0, 'frames': 4, 'fps': '2'},
             'files': {'features.json': t1.file_sha256(root / 'features.json')}, 'fits': {'0': np.eye(3).tolist()}}
    (tmp_path / 'inputs.json').write_text(json.dumps({'windows': {'match/clip@120': entry}}))
    monkeypatch.setattr(t1.k6, 'FRAME', (32, 16))
    monkeypatch.setattr(t1.ClipMasks, 'load', lambda _: ClipMasks(('person', 'racket'), 16, 32, 2, [[], [], [], []]))
    monkeypatch.setattr(t1.k6, 'window_players', lambda *a: {'players': ['person:1']})
    region = np.zeros((16, 32), bool)
    region[:, :16] = True
    monkeypatch.setattr(t1.k6, 'frame_regions', lambda *a: {'regions': {'person:1': region}})
    out = t1.measure(tmp_path, 'match/clip@120', 240)
    assert out['seconds'] == 2
    assert out['scored_frames'] == 1
    assert out['share_all']['foreground'] == .5
    assert out['mask']['8']['kbps'] == out['mask']['8']['bytes'] * 8 / 2 / 1000
    assert not out['ball_masks_available']
    assert out['share_all']['ball'] is None
    assert all(f['ball'] is None for f in out['frame_shares'])
    (root / 'features.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed input'):
        t1.measure(tmp_path, 'match/clip@120', 240)


def test_rate_relaxation_keeps_all_mask_and_container_framing():
    from fractions import Fraction
    from experiments.tennis.t1_oracle import container_bytes, relaxed_bytes
    from src.pointstream import bitstream as bs

    background, foreground, mask = b'b' * 141, b'f' * 1000, b'm' * 51
    packed = container_bytes(background, foreground, mask, 'dcvc', (960, 544), (480, 272), 4, Fraction(50), 240)
    decoded = bs.Container.unpack(packed)
    assert decoded.coded(bs.Tag.BACKGROUND)[0].stream == background
    assert decoded.coded(bs.Tag.FOREGROUND)[0].stream == foreground
    assert next(p for t, p in decoded.segments if t == bs.Tag.MASK) == mask
    relaxed = relaxed_bytes(packed, foreground)
    assert relaxed < len(packed)
    assert relaxed > len(background) + len(mask)
    assert relaxed == sum(decoded.breakdown().values()) - len(foreground)


def test_cpu_encoder_defers_cuda_and_honors_its_cpu_subset(tmp_path, monkeypatch):
    import argparse
    import struct
    import sys
    from fractions import Fraction
    from types import SimpleNamespace
    from experiments.tennis import t1_oracle
    from src.codecs import svtav1

    def forbidden_cuda():
        raise AssertionError('CPU encoder worker touched CUDA')

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(empty_cache=forbidden_cuda)))
    monkeypatch.setattr(t1_oracle.k6, 'scaled_size', lambda _: (8, 8))
    called = {}

    def encode(source, work, **kwargs):
        called.update(kwargs)
        stream, decoded = work / 'stream.ivf', work / 'decoded.yuv'
        stream.write_bytes(b'DKIF' + struct.pack('<HH', 0, 32) + b'\0' * 24 + struct.pack('<IQ', 1, 0) + b'x')
        decoded.write_bytes(b'\0' * (8 * 8 * 3 // 2))
        return {'stream': str(stream), 'decoded': str(decoded), 'cpus': kwargs['cpus']}

    monkeypatch.setattr(svtav1, 'code', encode)
    streams = tmp_path / 'streams'
    streams.mkdir()
    result = t1_oracle.encode(argparse.Namespace(encoder='svt', decoder='dav1d', threads=32), ('svt', 62, 1),
                              np.zeros((1, 8, 8, 3), np.uint8), {'frames': 1, 'fps': Fraction(50)},
                              tmp_path / 'work', streams, 'anchor', cpus=[2, 3], defer_rgb=True)
    assert called['cpus'] == [2, 3] and called['threads'] == 2
    assert result['_native_yuv'].endswith('decoded.yuv')
    from pathlib import Path
    assert not Path(result['decoded']).exists()
