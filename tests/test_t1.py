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
