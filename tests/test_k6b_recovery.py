"""Clip-local duplicate checks must preserve roles, sizes and absolute frame IDs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.datasets.k6b_recovery import exact_pairs, parse_hashes, screen


def test_native_hash_parser_rejects_invalid_digest():
    h = 'a' * 64
    assert parse_hashes('# header\n0, 0, 0, 1, 24, ' + h) == [h]
    with pytest.raises(ValueError):
        parse_hashes('0, 0, 0, 1, 24, short')


def test_exact_pairs_use_frame_offsets_and_never_self_match():
    old = {'role': 'old', 'clip': 'old', 'size': '1920x1080', 'hashes': ['a', 'b']}
    candidate = {'role': 'candidate', 'clip': 'new', 'size': '1920x1080', 'start': 93, 'hashes': ['b', 'c']}
    another = {'role': 'candidate', 'clip': 'other', 'size': '1920x1080', 'start': 141, 'hashes': ['c']}
    different_size = {'role': 'candidate', 'clip': '4k', 'size': '3840x2160', 'start': 0, 'hashes': ['b']}
    downscaled = {**old, 'clip': 'screen-only', 'hash_mode': 'screen_only_other_dimensions'}
    result = exact_pairs([old, candidate, another, different_size, downscaled])
    assert [(r['candidate'], r['frame'], r['other'], r['other_frame']) for r in result] == [
        ('new', 93, 'old', 1), ('other', 141, 'new', 94)]


def test_descriptor_screen_detects_candidate_copy_and_checks_bytes(tmp_path: Path):
    data = np.asarray([np.full(64*36, v, np.uint8) for v in (0, 10, 10)])
    p = tmp_path / 'descriptors.bin'
    p.write_bytes(data.tobytes())
    rows = [{'clip': clip, 'role': role, 'offset': i, 'frames': 1, 'start': start,
             'size': '64x36', 'hashes': [str(i)*64]}
            for i, (clip, role, start) in enumerate([('old', 'old', 0), ('a', 'candidate', 205), ('b', 'candidate', 18)])]
    (tmp_path / 'audit.json').write_text(json.dumps({'complete': True, 'records': rows,
        'descriptors_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))
    result = screen(tmp_path)
    assert result['candidates'][0]['minimum_rmse'] == 10
    assert result['candidates'][1]['nearest'][0] == {'frame': 18, 'rmse': 0, 'other': 'a', 'other_frame': 205}
    manifest = json.loads((tmp_path / 'audit.json').read_text())
    manifest['expected_clips'] = ['a', 'b', 'missing', 'old']
    (tmp_path / 'audit.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='missing source clips'):
        screen(tmp_path)
    manifest['complete'] = False
    manifest.pop('expected_clips')
    (tmp_path / 'audit.json').unlink()
    (tmp_path / 'partial.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete expected inventory'):
        screen(tmp_path)
    manifest['complete'] = True
    (tmp_path / 'partial.json').unlink()
    (tmp_path / 'audit.json').write_text(json.dumps(manifest))
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        screen(tmp_path)


def test_merger_requires_complete_inventory_and_preserves_offsets(tmp_path: Path):
    folders = [tmp_path / 'part0', tmp_path / 'part1']
    for i, p in enumerate(folders):
        p.mkdir()
        binary = p / 'descriptors.bin'
        binary.write_bytes(bytes(64*36))
        row = {'clip': 'old' if i == 0 else 'a', 'role': 'old' if i == 0 else 'candidate',
               'frames': 1, 'start': 0 if i == 0 else 37, 'offset': 0,
               'size': '1920x1080', 'hashes': ['0'*64]}
        payload = {'complete': bool(i), 'records': [row]}
        if i:
            payload.update(expected_clips=['a', 'old'], descriptors_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        (p / ('audit.json' if i else 'partial.json')).write_text(json.dumps(payload))
    result = screen(folders[0], [folders[1]])
    assert result['clips'] == result['decoded_frames'] == 2
    assert result['exact_pairs'][0]['frame'] == 37
    assert result['candidates'][0]['nearest'][0]['other_frame'] == 0
    with pytest.raises(ValueError, match='duplicate collected clip'):
        screen(folders[0], [folders[1], folders[1]])
