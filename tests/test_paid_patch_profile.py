import hashlib
import json
import shutil

import pytest

from experiments.jobs import inbox, staging
from experiments.tennis import paid_patch_profile as paid
from src.pointstream import tracked_patch


def sample_point():
    return {'crf': 48, 'wire_sha256': 'a' * 64,
            'positions': [{'position': [10, 20] if i != 1 else None} for i in range(48)]}


def test_config_uses_serialized_centres_and_none():
    point = sample_point()
    config = paid.config_for('tennis', 16, point)
    assert config['positions'][:2] == [[10, 20], None]
    assert config['position_provenance']['wire_sha256'] == 'a' * 64
    assert len(config['positions']) == 16 and config['crf'] == 48
    ego = paid.config_for('ego', 64, None)
    assert all(x is None for x in ego['positions']) and ego['fps'] == [50, 1]
    assert ego['full_range'] and ego['crf'] == 50 and ego['preset'] == 8


def test_full_parity_requires_all_pixel_and_both_stream_hashes():
    base, patch = b'base', b'patch'
    frames = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(48)]
    point = {'base_sha256': paid.common.digest(base), 'patch_sha256': paid.common.digest(patch),
             'replay': {'composite_frame_sha256': frames, 'composite_rgb_sha256': 'c' * 64}}
    decoded = {'rgb_frame_sha256': list(frames), 'decoded_sha256': 'c' * 64}
    assert paid.parity(point, base, patch, decoded, 48)['passed']
    assert paid.parity(point, base, patch, decoded, 16) == {'applicable': False, 'passed': None,
                                                             'reason': 'prefix encoding has different temporal context'}
    decoded['rgb_frame_sha256'][17] = '0' * 64
    assert not paid.parity(point, base, patch, decoded, 48)['passed']
    assert not paid.parity(point, b'changed', patch, decoded, 48)['passed']


def test_real_egocentric_decode_helper_and_tamper(tmp_path):
    if not all(shutil.which(x) for x in ('SvtAv1EncApp', 'dav1d')):
        pytest.skip('native tools unavailable')
    source, packet, rendered = (tmp_path / name for name in ('in.yuv', 'out.ptp', 'rgb'))
    source.write_bytes((bytes([80]) * 4096 + bytes([128]) * 2048) * 2)
    config = {'profile': 'tracked-pixel-v1', 'width': 64, 'height': 64, 'fps': [50, 1], 'frames': 2,
              'full_range': True, 'crf': 50, 'preset': 8, 'input_format': 'yuv420p', 'positions': [None, None]}
    result = tracked_patch.encode(source, packet, config, scratch=tmp_path)
    source.unlink()
    fresh = tracked_patch.decode(packet, rendered, scratch=tmp_path)
    assert len(fresh['rgb_frame_sha256']) == 2 and fresh['decoded_sha256'] == result['decoded_sha256']
    assert tracked_patch.inspect(packet.read_bytes())['patch_bytes'] == 0
    corrupted = bytearray(packet.read_bytes())
    corrupted[-33] ^= 1
    with pytest.raises(ValueError):
        tracked_patch.unpack(bytes(corrupted))


def test_thin_publication_lifecycle_and_hash_tamper(tmp_path):
    scratch, stage = tmp_path / 'scratch', tmp_path / 'stage'
    pub = scratch / 'paid-private' / 'publication'
    pub.mkdir(parents=True)
    stage.mkdir()
    for name in paid.publication_members('tennis'):
        (pub / name).write_text(json.dumps({'name': name}))
    archive = stage / 'published.tar'
    identity = paid.publish(pub, archive, 'tennis')
    assert staging.publish_scratch(scratch, stage) is None
    staging.release_scratch(scratch, stage)
    root = paid.extract(archive, identity, 'tennis', tmp_path / 'replay')
    assert {x.name for x in root.iterdir()} == set(paid.publication_members('tennis'))
    with pytest.raises(ValueError):
        paid.extract(archive, dict(identity, sha256='0' * 64), 'tennis', tmp_path / 'tampered')
    assert not any(p.suffix in ('.rgb', '.yuv') for p in root.iterdir())


def test_specs_cost_and_prefix_copy(tmp_path):
    for domain, full in (('tennis', 48), ('ego', 64)):
        config = inbox.validate_spec(paid.spec(domain, '2026-11-16T12:00:00+00:00'))
        assert config['hosts'] == ['gpu6'] and config['cpu_affinity_index'] == 0
        assert config['scale']['frames'] == {'smoke': 16, 'full': full}
        assert config['required_commands'] == ['ffmpeg'] and config['budget_seconds'] == 1530
        assert paid.cost_record(100, 80, 16, domain)['passed']
        assert not paid.cost_record(200, 180, 16, domain)['passed']
    source, out = tmp_path / 'source', tmp_path / 'prefix'
    source.write_bytes(bytes(range(64)))
    assert paid.prefix(source, out, 2, 16, 32).read_bytes() == bytes(range(32))


def test_ordinal_separation_reports_no_fabricated_bd_rate():
    test = [(100, -.0151), (200, -.0100), (300, -.008)]
    old = [(80, -.035), (180, -.025), (250, -.0157)]
    result = paid.ordinal_separation(test, old)
    assert result['status'] == 'candidate_strictly_better'
    assert result['common_rate_interval'] == [100, 250]
    assert 'bd_rate' not in result
    absent = paid.ordinal_separation([(1, 10), (2, 20)], [(3, 1), (4, 2)])
    assert absent['status'] == 'no_common_rate_interval'


def test_reprice_rejects_unauthenticated_score_before_any_receiver(tmp_path):
    score, stage, archive = (tmp_path / x for x in ('score.json', 'stage.json', 'packet.tar'))
    score.write_text('{}')
    stage.write_text('{}')
    archive.write_bytes(b'')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        paid.reprice(score, stage, '0' * 64, archive, 'dav1d', tmp_path)
