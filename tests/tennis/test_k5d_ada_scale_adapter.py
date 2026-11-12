import pytest

from experiments.tennis import k5d_ada_scale_adapter as adapter


def test_producer_rejects_legacy_receipt_without_actual_denial_verifier(tmp_path, monkeypatch):
    (tmp_path / 'checkpoint.json').write_text('{}')
    (tmp_path / 'pilot.json').write_text('{}')
    marker = {'producer_job': '20261112T000000Z-test', 'producer_stage': 'smoke', 'frames': 16,
              'gpu': {'name': adapter.GPU_NAME, 'uuid': 'GPU-test'}}
    monkeypatch.setattr(adapter.cp, 'verify', lambda *a, **kw: None)
    monkeypatch.setattr(adapter.receiver, 'verify', lambda *a, **kw: marker)
    with pytest.raises(ValueError, match='denial verifier unavailable'):
        adapter.producer(tmp_path, {}, job=marker['producer_job'], stage='smoke', frames=16)
    with pytest.raises(ValueError, match='producer job'):
        adapter.producer(tmp_path, {}, job='20261112T000000Z-other', stage='smoke', frames=16)
    with pytest.raises(ValueError, match='paired 16/48'):
        adapter.producer(tmp_path, {}, job=marker['producer_job'], stage='full', frames=188)


def test_score16_uses_pinned_ada_archive_and_exact_comparison(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def unpack(path, digest):
        assert path == tmp_path / 'archive' and digest == adapter.REFERENCE_SHA256
        yield tmp_path / 'ada'

    monkeypatch.setattr(adapter.cost, 'unpack', unpack)
    monkeypatch.setattr(adapter, 'reference', lambda root: ({}, {}))
    monkeypatch.setattr(adapter.fresh, 'compare', lambda *args: {'paths': args})
    result = adapter.score16(tmp_path / 'archive', tmp_path / 'scored', tmp_path / 'accepted')
    assert result['paths'] == (tmp_path / 'ada/scored/assembled', tmp_path / 'scored', tmp_path / 'accepted')


def test_prepare16_accepts_exact_six_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import copy
    import json
    from contextlib import contextmanager

    points = [{'codec': codec, 'point': point, 'native_sha256': 'n', 'native_bytes': 2,
               'wire_sha256': 'w', 'wire_bytes': 3, 'encoder_decoded_sha256': 'd',
               'encoder_rendered_sha256': 'r', 'baseline': {'native_bytes': 2, 'native_sha256': 'n',
               'zero_ball_wire_bytes': 4, 'zero_ball_wire_sha256': 'z'}, 'streak': {'guard_off': False}}
              for codec, point in adapter.enc.POINTS]
    old = {'frames': 16, 'conversion': {'yuv_sha256': 'y'}, 'points': points}
    source = {'rows': [1], 'scores': {'x': 1}}
    (tmp_path / 'checkpoint.json').write_text('{}')
    (tmp_path / 'source-qualification.json').write_text(json.dumps(source))
    (tmp_path / 'pilot.json').write_text(json.dumps(old))

    @contextmanager
    def unpack(*_):
        yield tmp_path

    monkeypatch.setattr(adapter.cost, 'unpack', unpack)
    monkeypatch.setattr(adapter.cost, 'compare_contract', lambda *a, **k: None)
    monkeypatch.setattr(adapter.cp, 'verify', lambda *a, **k: None)
    monkeypatch.setattr(adapter, 'reference', lambda _: (old, source))
    adapter.prepare16(tmp_path / 'ref', tmp_path, {})
    changed = copy.deepcopy(old)
    changed['points'][2]['wire_bytes'] += 1
    (tmp_path / 'pilot.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='wire_bytes'):
        adapter.prepare16(tmp_path / 'ref', tmp_path, {})
