import json
from pathlib import Path

import pytest

from experiments.tennis import k5d_checkpoint as cp


def make_preparation(root, monkeypatch):
    root.mkdir()
    monkeypatch.setattr(cp.oracle, 'validate_report', lambda _: {})
    (root / 'source-qualification.json').write_text('{"qualified": true}')
    points = []
    for codec, setting in cp.enc.POINTS:
        folder = root / f'{codec}-{setting}'
        folder.mkdir()
        for name in ('native.bin', 'stream.psbl', 'baseline.psbl'):
            (folder / name).write_bytes((name + codec + str(setting)).encode())
        points.append({'codec': codec, 'point': setting, 'native_sha256': cp.oracle.sha(folder / 'native.bin'),
                       'wire_sha256': cp.oracle.sha(folder / 'stream.psbl'),
                       'encoder_decoded_sha256': 'decode', 'encoder_rendered_sha256': 'render',
                       'decode_parity': True, 'render_parity': True,
                       'baseline': {'metrics': [], 'judges': {}, 'zero_ball_wire_sha256': cp.oracle.sha(folder / 'baseline.psbl')},
                       'streak': {'guard_off': False, 'metrics': [], 'judges': {}, 'reuse': None}})
    (root / 'pilot.json').write_text(json.dumps({'frames': 2, 'clip': 'sample', 'checkpoint_prepare': True, 'full_metrics': True, 'points': points}))
    identity = {'source_sha256': 'source', 'snapshot_sha256': 'snapshot', 'environment_sha256': 'runtime'}
    cp.seal(root, identity)
    return identity, points, cp.oracle.sha(root / 'checkpoint.json')


def test_checkpoint_rejects_tampering_runtime_and_links(tmp_path, monkeypatch):
    root = tmp_path / 'prepare'
    identity, _, digest = make_preparation(root, monkeypatch)
    assert cp.verify(root, digest, identity)['frames'] == 2
    with pytest.raises(ValueError, match='runtime'):
        cp.verify(root, digest, {**identity, 'environment_sha256': 'different'})
    file = root / 'dcvc-0' / 'native.bin'
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint content'):
        cp.verify(root, digest, identity)
    file.unlink()
    file.symlink_to(root / 'pilot.json')
    with pytest.raises(ValueError, match='link'):
        cp.verify(root, digest, identity)


def test_assembly_requires_six_distinct_bound_score_receipts(tmp_path, monkeypatch):
    root = tmp_path / 'prepare'
    identity, points, digest = make_preparation(root, monkeypatch)
    receipts = []
    for i, p in enumerate(points):
        p = json.loads(json.dumps(p))
        p['baseline'].update(metrics=[{'frame': 0}, {'frame': 1}], judges={'wasb': 'ok'})
        p['streak'].update(metrics=[{'frame': 0}, {'frame': 1}], judges={'wasb': 'ok'})
        file = tmp_path / f'{i}.json'
        file.write_text(json.dumps({'checkpoint_sha256': digest, 'execution_identity': identity, 'point': p,
                                    'receiver_replay': {'decoded_yuv_sha256': 'decode', 'rendered_rgb_sha256': 'render'}}))
        receipts.append({'path': str(file), 'sha256': cp.oracle.sha(file)})
    monkeypatch.setattr(cp.enc, 'validate', lambda out: {'frames': 2, 'points': len(json.loads((out / 'pilot.json').read_text())['points'])})
    with pytest.raises(ValueError, match='six'):
        cp.assemble(root, digest, receipts[:-1], tmp_path / 'short', identity)
    with pytest.raises(ValueError, match='duplicate'):
        cp.assemble(root, digest, receipts[:-1] + receipts[:1], tmp_path / 'duplicate', identity)
    assert cp.assemble(root, digest, receipts, tmp_path / 'complete', identity) == {'frames': 2, 'points': 6}
    Path(receipts[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        cp.assemble(root, digest, receipts, tmp_path / 'changed', identity)
