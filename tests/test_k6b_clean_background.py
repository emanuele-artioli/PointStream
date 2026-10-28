import struct

import numpy as np
import pytest

from experiments.tennis.k6b_clean_background_prep import erase_yuv, stream_info
from experiments.tennis.k6b_clean_background_score import anchor_at, assess, new_images


def test_erase_preserves_source_outside_each_plane_mask():
    frame = np.full((1620, 1920), 100, np.uint8)
    frame[100:110, 100:110] = 210
    region = np.zeros((1080, 1920), bool)
    region[100:110, 100:110] = True
    result = erase_yuv(frame, region)
    assert np.array_equal(result[:1080][~region], frame[:1080][~region])
    chroma_region = region[::2, ::2]
    for original, filled in zip(frame[1080:].reshape(2, 540, 960), result[1080:].reshape(2, 540, 960)):
        assert np.array_equal(filled[~chroma_region], original[~chroma_region])
    assert result[105, 105] < 210
    with pytest.raises(ValueError):
        erase_yuv(frame, np.ones_like(region))


def test_ivf_requires_every_native_tick_and_rejects_truncation(tmp_path):
    path = tmp_path / 'stream.ivf'
    header = b'DKIF' + struct.pack('<HH4sHHIIII', 0, 32, b'AV01', 480, 272, 25, 1, 2, 0)
    payload = header + struct.pack('<IQ', 1, 0) + b'a' + struct.pack('<IQ', 1, 1) + b'b'
    path.write_bytes(payload)
    assert stream_info(path, 2)['framing_bytes'] == 56
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        stream_info(path, 2)
    path.write_bytes(header + struct.pack('<IQ', 1, 0) + b'a' + struct.pack('<IQ', 1, 0) + b'b')
    with pytest.raises(ValueError, match='reordered'):
        stream_info(path, 2)


def test_decision_uses_one_new_policy_and_existing_controls_cannot_promote():
    scores = {'lpips': np.array([[.01], [.01], [.19], [.21]]),
              'dists': np.array([[.01], [.01], [.21], [.19]])}
    assert assess(scores, {'lpips': .2, 'dists': .2})['verdict'] == 'MIXED/UNRESOLVED'
    scores['dists'][2] = .19
    assert assess(scores, {'lpips': .2, 'dists': .2})['verdict'] == 'HEADROOM'
    scores['lpips'][2:] = .21
    scores['dists'][2:] = .21
    assert assess(scores, {'lpips': .2, 'dists': .2})['verdict'] == 'NO-GO'
    scores['lpips'][2] = .1995
    assert assess(scores, {'lpips': .2, 'dists': .2})['verdict'] == 'MIXED/UNRESOLVED'
    assert assess(scores, None)['verdict'] == 'UNRESOLVED/ANCHOR-COVERAGE'
    scores['lpips'][2] = np.nan
    with pytest.raises(ValueError):
        assess(scores, None)


def test_anchor_does_not_extrapolate():
    curves = {m: [(10., -.5), (30., -.1)] for m in ('lpips', 'dists')}
    assert anchor_at(curves, 20.) == {'lpips': .3, 'dists': .3}
    assert anchor_at(curves, 9.9) is None
    assert anchor_at(curves, 30.1) is None


def test_new_base_never_inherits_old_base_outside_written_support():
    images = np.zeros((3, 1080, 1920, 3), np.uint8)
    images[0] = 40
    images[1:] = 80
    written = np.zeros((1080, 1920), bool)
    region = written.copy()
    written[10:20, 10:20] = True
    region[5:25, 5:25] = True
    overlay = np.full_like(images[0], 120)
    images[2][written] = 120
    base = np.full_like(images[0], 20)
    result = new_images(images, overlay, written, region, base)
    assert np.all(result[3][~written] == 20) and np.all(result[3][written] == 120)
    assert np.all(result[4][~region] == 20) and np.all(result[4][region] == 40)
    images[2, 0, 0] = 2
    with pytest.raises(ValueError, match='replay'):
        new_images(images, overlay, written, region, base)


@pytest.mark.parametrize('mutation', ['none', 'memory', 'input', 'budget', 'timing'])
def test_native_quality_validator_keeps_hard_contract(tmp_path, monkeypatch, mutation):
    import hashlib
    import io
    import json
    import tarfile
    from experiments.tennis.k6b_clean_background_prep import TOOLS
    from experiments.tennis.k6b_clean_background_score import POLICIES, WEIGHTS, validate

    stage = tmp_path / 'smoke'
    stage.mkdir()
    pixels = io.BytesIO()
    np.save(pixels, np.zeros((3, 1080, 1920, 3), np.uint8))
    summary = {'ticks': [0, 3, 6], 'policies': list(POLICIES),
               'timings': {p: [1., .15, .3] for p in POLICIES},
               'timing_secondary': True, 'source_rgb_checks': 3, 'cuda': True, 'gpu': 'RTX A6000',
               'peak_allocated_mib': 3500, 'peak_reserved_mib': 4000,
               'control_errors': {m: 0. for m in WEIGHTS}, 'weight_sha256': WEIGHTS,
               'prepared_sha256': 'input', 'hashes': {str(t): ['h'] * 5 for t in (0, 3, 6)},
               'native_decoder': {'sha256': TOOLS['dav1d']}, 'side_kbps': 5.607128888888889,
               'clean_base_rgb_sha256': hashlib.sha256(pixels.getvalue()).hexdigest()}
    spec = {'arguments': ['--timing-secondary'], 'full': {'seconds': 350}, 'budget_seconds': 800,
            'scale': {'judged': {'smoke': 3, 'full': 0}},
            'staged_inputs': [{'name': 'prepared', 'sha256': 'input'}]}
    if mutation == 'memory':
        summary['peak_reserved_mib'] = 8100
    elif mutation == 'input':
        summary['prepared_sha256'] = 'changed'
    elif mutation == 'budget':
        spec['budget_seconds'] = 900
    elif mutation == 'timing':
        spec['arguments'] = []
        summary['timings'] = {p: [1., .3, .3] for p in POLICIES}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    (stage / 'quality-summary.json').write_text(json.dumps(summary))
    (stage / 'execution.json').write_text(json.dumps({'seconds': 15}))
    (stage / 'staging.json').write_text(json.dumps({'publication_seconds': .1}))
    data = io.BytesIO()
    np.savez(data, ticks=[0, 3, 6], **{m: np.zeros((4, 3)) for m in WEIGHTS})
    with tarfile.open(stage / 'published.tar', 'w') as tar:
        for name, payload in [('scores.npz', data.getvalue()), ('quality.json', json.dumps(summary).encode()),
                              ('clean-base-rgb.npy', pixels.getvalue())]:
            member = tarfile.TarInfo('publish/' + name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    result = tmp_path / 'validation.json'
    monkeypatch.setenv('PS_VALIDATION_PATH', str(result))
    if mutation == 'none':
        validate()
        assert json.loads(result.read_text())['passed']
    else:
        with pytest.raises(SystemExit):
            validate()
        assert not json.loads(result.read_text())['passed']
