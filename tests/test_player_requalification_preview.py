import json

import numpy as np
import pytest

from experiments.tennis import player_requalification_preview as preview


def fixture(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{}')
    data = {"t": np.arange(24), "box": np.tile([1, 2, 3, 4], (24, 1)),
            "reference": np.full((16, 16, 3), 127, dtype=np.uint8)}
    entry = {"unit": "unit", "t": data["t"].tolist(), "reference_sha256": "r", "pose_sha256": "p"}
    plan = {"source": {"native_fps": [50, 1]}, "units": [entry]}
    monkeypatch.setattr(preview, "resolve_units", lambda config, bindings: {"units": [entry]})
    monkeypatch.setattr(preview, "make_plan", lambda config, digest, count: (plan, [data]))
    folders = []
    for profile in preview.PROFILES:
        folder = tmp_path / profile
        folder.mkdir()
        path = folder / f"unit@{profile}.npz"
        np.savez(path, t=data["t"], box=data["box"], crops=np.full((24, 16, 16, 3), 127, dtype=np.uint8))
        record = {"unit": "unit", "profile": profile, "plan_sha256": preview.digest_json(plan),
                  "reference_sha256": "r", "pose_sha256": "p", "output_sha256": preview.sha256(path)}
        path.with_suffix('.json').write_text(json.dumps(record))
        receipt = {"schema": 1, "scope": "24-native-frame-context-only", "profile": profile,
                   "plan": plan, "plan_sha256": preview.digest_json(plan), "outputs": [record]}
        (folder / 'receipt.json').write_text(json.dumps(receipt))
        folders.append(folder)
    return config, preview.sha256(config), folders


def test_incomplete_preview_and_existing_output(tmp_path, monkeypatch):
    config, digest, folders = fixture(tmp_path, monkeypatch)
    # This tests the review contract; native media are qualified separately.
    def encode_stub(crops, path, fps):
        assert crops.shape == (24, 16, 16, 3)
        assert fps == 50
        path.write_bytes(b'test display proxy')

    monkeypatch.setattr(preview, 'encode', encode_stub)
    monkeypatch.setattr(preview.shutil, 'which', lambda name: None)
    monkeypatch.setattr(preview.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected native executable'))
    output = tmp_path / 'preview'
    preview.build(config, digest, [], folders[:1], output, 1)
    page = (output / 'index.html').read_text()
    assert 'INCOMPLETE' in page and 'not source-ground-truth video' in page
    assert '<option>A</option>' not in page
    assert (output / 'unit-A-50.mp4').is_file()
    with pytest.raises(FileExistsError):
        preview.build(config, digest, [], folders[:1], output, 1)


def test_rejects_plan_and_hash_and_ticks(tmp_path, monkeypatch):
    config, digest, folders = fixture(tmp_path, monkeypatch)
    receipt_path = folders[1] / 'receipt.json'
    original = receipt_path.read_text()
    receipt = json.loads(original)
    receipt['plan']['source']['native_fps'] = [25, 1]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='unequal'):
        preview.authenticate(config, digest, [], folders)
    receipt_path.write_text(original)
    path = folders[0] / 'unit@original.npz'
    with np.load(path) as f:
        crops, boxes, ticks = f['crops'], f['box'], f['t'].copy()
    ticks[0] = 3
    np.savez(path, t=ticks, box=boxes, crops=crops)
    with pytest.raises(ValueError, match='hash'):
        preview.authenticate(config, digest, [], folders)
    sidepath = path.with_suffix('.json')
    side = json.loads(sidepath.read_text())
    side['output_sha256'] = preview.sha256(path)
    sidepath.write_text(json.dumps(side))
    rp = folders[0] / 'receipt.json'
    receipt = json.loads(rp.read_text())
    receipt['outputs'][0] = side
    rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='ticks'):
        preview.authenticate(config, digest, [], folders)
    with pytest.raises(ValueError, match='config hash'):
        preview.authenticate(config, 'bad', [], folders)


def test_source_identity_and_offset_guard_before_decoding(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'not a video')
    plan = {'source': {'source_video_sha256': 'wrong'}, 'units': []}
    monkeypatch.setattr(preview.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected subprocess'))
    with pytest.raises(ValueError, match='source hash'):
        preview.source_crops(source, plan, [], tmp_path / 'config', [])
    plan['source'] = {'source_video_sha256': preview.sha256(source), 'window': 'nadal_kyrgios/000@120',
                      'width': 1920, 'height': 1080, 'native_fps': [50, 1]}
    plan['units'] = [{'t': list(range(1, 25))}]
    with pytest.raises(ValueError, match='offset'):
        preview.source_crops(source, plan, [], tmp_path / 'config', [])
    plan['units'] = [{'t': list(range(24))}]
    plan['source']['width'] = 1280
    with pytest.raises(ValueError, match='geometry'):
        preview.source_crops(source, plan, [], tmp_path / 'config', [])


def test_authenticate_resolves_only_consumed_units(tmp_path, monkeypatch):
    real_resolve = preview.resolve_units
    config, _, folders = fixture(tmp_path, monkeypatch)
    unit_root = tmp_path / 'conditioning'
    (unit_root / 'publish').mkdir(parents=True)
    (unit_root / 'publish' / 'unit.npz').write_bytes(b'unit')
    config.write_text(json.dumps({'units': [{'path': 'staged:conditioning/publish/unit.npz'}],
                                  'weights': {'unused': {'path': 'staged:unavailable/model'}},
                                  'tennis_checkpoint': {'path': 'staged:unavailable/base'},
                                  'match_checkpoint': {'path': 'staged:unavailable/match'}}))
    monkeypatch.setattr(preview, 'resolve_units', real_resolve)
    preview.authenticate(config, preview.sha256(config), [['conditioning', str(unit_root)]], folders)
    with pytest.raises(ValueError, match='extra'):
        real_resolve(json.loads(config.read_text()), [['conditioning', str(unit_root)], ['extra', str(tmp_path)]])
    bad = json.loads(config.read_text())
    bad['units'][0]['path'] = 'staged:conditioning/../unit.npz'
    with pytest.raises(ValueError, match='invalid staged'):
        real_resolve(bad, [['conditioning', str(unit_root)]])


def test_motion_unpaired_preview_and_frame_authentication(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'schema': 2, 'scope': 'player-motion-development',
                                  'native_frames': {'smoke': 44, 'full': 240}, 'units': [{}],
                                  'tennis_checkpoint': {'sha256': preview.BASE_SHA}}))
    data = {'t': np.arange(44), 'box': np.tile([1, 2, 3, 4], (44, 1)),
            'reference': np.full((16, 16, 3), 127, np.uint8)}
    entry = {'unit': 'unit', 't': data['t'].tolist(), 'reference_sha256': 'r', 'pose_sha256': 'p',
             'review_label': 'Player 2'}
    plan = {'source': {'native_fps': [50, 1]}, 'units': [entry], 'native_frames': 44}
    monkeypatch.setattr(preview, 'resolve_units', lambda config, bindings: config)
    monkeypatch.setattr(preview, 'make_plan', lambda config, digest, count, frames: (plan, [data]))
    folder = tmp_path / 'publish'
    folder.mkdir()
    path = folder / 'unit@tennis_only.npz'
    crops = np.full((44, 16, 16, 3), 127, np.uint8)
    np.savez(path, t=data['t'], box=data['box'], crops=crops)
    record = {'unit': 'unit', 'profile': 'tennis_only', 'plan_sha256': preview.digest_json(plan),
              'reference_sha256': 'r', 'pose_sha256': 'p', 'output_sha256': preview.sha256(path)}
    path.with_suffix('.json').write_text(json.dumps(record))
    receipt = {'schema': 2, 'scope': 'player-motion-development', 'native_frames': 44,
               'profile': 'tennis_only', 'plan': plan, 'plan_sha256': preview.digest_json(plan), 'outputs': [record],
               'fit_sha256': preview.BASE_SHA, 'base_sha256': None,
               'model_record': {'fit': {'checkpoint_sha256': preview.BASE_SHA, 'base': None}}}
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    digest = preview.sha256(config)
    preview.authenticate(config, digest, [], [folder])
    with pytest.raises(ValueError, match='source'):
        preview.build(config, digest, [], [folder], tmp_path / 'missing', 1)
    monkeypatch.setattr(preview, 'source_crops', lambda *args: ([crops], [0]))
    monkeypatch.setattr(preview, 'encode', lambda frames, path, fps: path.write_bytes(b'video'))
    monkeypatch.setattr(preview.shutil, 'which', lambda name: None)
    source = tmp_path / 'source'
    source.write_bytes(b'source')
    preview.build(config, digest, [], [folder], tmp_path / 'view', 1, source=source)
    page = (tmp_path / 'view' / 'index.html').read_text()
    assert 'Unpaired common-tennis motion review' in page and '44 native ticks' in page
    assert '<h2>Player 2 — ticks' in page and '<h2>Player 1 — ticks' not in page
    assert '<option>Acceptable</option>' in page and '<option>Not acceptable</option>' in page
    assert '<option>A</option>' not in page and '<option>Tie</option>' not in page
    np.savez(path, t=data['t'], box=data['box'], crops=crops[:-1])
    record['output_sha256'] = preview.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(record))
    receipt['outputs'] = [record]
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='crops'):
        preview.authenticate(config, digest, [], [folder])


def test_motion_source_probe_rejects_mismatched_format(tmp_path, monkeypatch):
    import types

    source = tmp_path / 'source'
    source.write_bytes(b'source')
    plan = {'source': {'source_video_sha256': preview.sha256(source), 'window': 'nadal_kyrgios/000@120',
                       'width': 1920, 'height': 1080, 'native_fps': [50, 1]},
            'units': [{'t': list(range(44))}]}
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        stream = {'width': 1920, 'height': 1080, 'r_frame_rate': '50/1',
                  'avg_frame_rate': '50/1', 'nb_frames': '239'}
        return types.SimpleNamespace(stdout=json.dumps({'streams': [stream]}).encode())

    monkeypatch.setattr(preview.subprocess, 'run', probe)
    with pytest.raises(ValueError, match='frame count'):
        preview.source_crops(source, plan, [], tmp_path / 'config', [])
    assert len(calls) == 1 and calls[0][0] == 'ffprobe'
    plan['units'][0]['t'][15] = 16
    with pytest.raises(ValueError, match='offset'):
        preview.source_crops(source, plan, [], tmp_path / 'config', [])
    assert len(calls) == 1


def test_motion_review_label_is_html_escaped(tmp_path, monkeypatch):
    data = {'reference': np.zeros((16, 16, 3), np.uint8)}
    entry = {'unit': 'unit', 't': list(range(44)), 'review_label': '<Player & 2>'}
    plan = {'units': [entry]}
    folder = tmp_path / 'publish'
    folder.mkdir()
    np.savez(folder / 'unit@tennis_only.npz', crops=np.zeros((44, 16, 16, 3), np.uint8))
    (folder / 'receipt.json').write_text('{}')
    monkeypatch.setattr(preview, 'authenticate', lambda *args: (plan, [data], [{'schema': 2, 'profile': 'tennis_only'}]))
    monkeypatch.setattr(preview, 'source_crops', lambda *args: ([np.zeros((44, 16, 16, 3), np.uint8)], [0]))
    monkeypatch.setattr(preview, 'encode', lambda frames, path, fps: path.write_bytes(b'video'))
    monkeypatch.setattr(preview.shutil, 'which', lambda name: None)
    source = tmp_path / 'source'
    source.write_bytes(b'source')
    preview.build(tmp_path / 'config', 'a' * 64, [], [folder], tmp_path / 'out', 1, source=source)
    page = (tmp_path / 'out' / 'index.html').read_text()
    assert '<h2>&lt;Player &amp; 2&gt; — ticks' in page
    assert '<h2><Player' not in page
