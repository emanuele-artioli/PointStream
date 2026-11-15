from fractions import Fraction

import numpy as np
import pytest

from experiments.tennis import ball_roi_control as control, k5d_encoder, w1_roi
from src.codecs import svtav1


def rows(x=0.0, y=0.0):
    return [{'label_kind': 1, 'label_x': x, 'label_y': y} for _ in range(48)]


def test_gt_crop_matches_historical_crop_with_boundary_and_ties():
    for x, y in [(0., 0.), (1279., 719.), (100.5, 112.5)]:
        mask = control.regions(rows(x, y), 16)[0]
        image = np.arange(control.W * control.H, dtype=np.int32).reshape(control.H, control.W)
        assert np.array_equal(image[mask], k5d_encoder.crop64(image, x, y).ravel())
        assert mask.sum() == 4096
        cells = w1_roi.roi_text(mask[None], control.W, control.H, -40, outside=True).split()
        assert cells[0] == '0'
        assert set(cells[1:]) == {'0', '40'}


def test_reject_invalid_visibility_and_frame_count():
    bad = rows()
    bad[0]['label_kind'] = 0
    with pytest.raises(ValueError):
        control.regions(bad, 16)
    with pytest.raises(ValueError):
        control.regions(rows(), 17)


def test_native_command_preserves_historical_flags_and_only_adds_roi(tmp_path):
    opts = dict(width=1280, height=720, fps=Fraction(30, 1), frames=48,
                crf=62, preset=4, full_range=False)
    base = svtav1.encode_command('svt', tmp_path / 'in', tmp_path / 'out', **opts)
    roi = svtav1.encode_command('svt', tmp_path / 'in', tmp_path / 'out',
                                 extra=['--roi-map-file', 'roi.txt'], **opts)
    assert roi == base + ['--roi-map-file', 'roi.txt']
    assert base[base.index('--lp') + 1] == '6'
    assert base[base.index('--keyint') + 1] == '-1'
    assert '--color-primaries' not in base


def test_hash_gate_and_control_decode_identity(tmp_path):
    path = tmp_path / 'source'
    path.write_bytes(b'source')
    control.require_hash(path, control.digest(b'source'))
    with pytest.raises(ValueError):
        control.require_hash(path, control.digest(b'other'))
    point = {'codec': 'svt', 'point': 62, 'native_sha256': control.NATIVE62,
             'encoder_decoded_sha256': control.NATIVE62_DECODE}
    assert control.expected_native_decode({'points': [point]}) == control.NATIVE62_DECODE
    point['encoder_decoded_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        control.expected_native_decode({'points': [point]})


def test_complete_ivf_counts_header_and_frame_overhead():
    import struct

    data = b'DKIF' + b'\0\0' + struct.pack('<H', 32) + bytes(24)
    data += struct.pack('<I', 3) + bytes(8) + b'abc'
    assert svtav1.ivf_frames(data) == [3]
    assert len(data) == 47


def test_private_publication_survives_fleet_release_and_detects_tampering(tmp_path):
    from experiments.jobs import staging

    stage = tmp_path / 'stage'
    scratch = tmp_path / 'scratch'
    publication = scratch / 'roi-private' / 'publication'
    stage.mkdir()
    publication.mkdir(parents=True)
    for name in control.MEMBERS:
        (publication / name).write_bytes(('independent content ' + name).encode())
    archive = stage / 'published.tar'
    identity = control.publish(publication, archive)
    before = control.k5d_source_oracle.sha(archive)
    assert staging.publish_scratch(scratch, stage) is None
    staging.release_scratch(scratch, stage)
    assert not scratch.exists()
    assert control.k5d_source_oracle.sha(archive) == before
    replay = control.extract_thin(archive, identity, tmp_path / 'replay')
    assert {p.name for p in replay.iterdir()} == set(control.MEMBERS)
    assert not any(p.suffix in ('.yuv', '.rgb', '.npy') for p in replay.iterdir())
    damaged = dict(identity, sha256='0' * 64)
    with pytest.raises(ValueError):
        control.extract_thin(archive, damaged, tmp_path / 'damaged')


def test_frozen_spec_and_cost_gate():
    spec = control.cpu_spec('2026-11-16T12:00:00+00:00')
    assert spec['hosts'] == ['gpu6']
    assert spec['required_commands'] == ['ffmpeg']
    assert spec['cpu_threads'] == 1 and spec['cpu_affinity_index'] == 0
    assert spec['scale']['frames'] == {'smoke': 16, 'full': 48}
    assert spec['budget_seconds'] == 1530
    assert control.cost_record(190, 180, 16)['passed']
    assert not control.cost_record(194, 180, 16)['passed']
    assert not control.cost_record(901, 800, 48)['passed']


def test_historical_metric_control_requires_all_rows_and_pixel_parity():
    original = [{'frame': i, 'sse': i + 1, 'pixels': 100, 'frame_psnr': 31.,
                 'full_lpips': .1, 'full_dists': .2, 'crop_lpips': .3, 'crop_dists': .4}
                for i in range(48)]
    copy = [dict(row) for row in original]
    assert control.compare_metric_control(copy, original)['passed']
    copy[2]['full_lpips'] = .100002
    with pytest.raises(ValueError):
        control.compare_metric_control(copy, original)
    copy = [dict(row) for row in original]
    copy[0]['sse'] = 0
    with pytest.raises(ValueError):
        control.compare_metric_control(copy, original)
    with pytest.raises(ValueError):
        control.compare_metric_control(original[:16], original[:16])


def test_score_member_is_content_bound(tmp_path):
    import io
    import tarfile

    path = tmp_path / 'score.tar'
    content = b'score information'
    with tarfile.open(path, 'w') as tar:
        entry = tarfile.TarInfo('publish/point-svt-62.json')
        entry.size = len(content)
        tar.addfile(entry, io.BytesIO(content))
    assert control.tar_member(path, 'publish/point-svt-62.json', control.digest(content)) == content
    with pytest.raises(ValueError):
        control.tar_member(path, 'publish/point-svt-62.json', '0' * 64)


def test_bd_analysis_rejects_smoke_and_uses_complete_packet_sizes():
    def metrics(seed):
        return [{'frame': i, 'sse': 200 + seed * 20, 'pixels': 100,
                 'full_lpips': .1 + seed * .01, 'full_dists': .2 + seed * .01,
                 'crop_lpips': .3 + seed * .01, 'crop_dists': .4 + seed * .01}
                for i in range(48)]

    record = {'frames': 16, 'available_receiver_frames': 48, 'metric_control': {'passed': True}, 'producer_sha256': control.PRODUCER,
              'source_archive_sha256': control.ARCHIVE, 'historical_score_archive_sha256': control.SCORE_ARCHIVE,
              'metric_definition': {}, 'rows': [
                  {'crf': crf, 'file_bytes': size, 'frames': metrics(seed)}
                  for crf, size, seed in [(35, 1000, 0), (48, 700, 1), (62, 400, 2)]],
              'historical_points': [
                  {'crf': crf, 'native_bytes': native, 'wire_bytes': wire,
                   'baseline': metrics(seed), 'streak': metrics(seed)}
                  for crf, native, wire, seed in [(48, 900, 920, 0), (55, 600, 620, 1), (62, 300, 320, 2)]]}
    with pytest.raises(ValueError):
        control.analyse_score(record)
    record['frames'] = 48
    analysis = control.analyse_score(record)
    assert [p['bytes'] for p in analysis['physical_curves']['streak']] == [920, 620, 320]
    assert [p['bytes'] for p in analysis['physical_curves']['native']] == [900, 600, 300]
    assert analysis['comparisons']['streak_vs_roi']['full_lpips']['status'] == 'supported'


def test_score_fleet_spec_and_forecast():
    from experiments.jobs.inbox import validate_spec

    spec = control.gpu_spec('2026-11-16T12:00:00+00:00', '/data/a.tar', 'a' * 64,
                            '/data/a.json', 'b' * 64)
    assert validate_spec(spec)['budget_seconds'] == 2430
    assert spec['required_commands'] == ['ffmpeg']
    assert spec['cpu_affinity_index'] == 1 and spec['gpu_models'] == ['RTX 6000 Ada']
    assert control.gpu_cost(390, 300, 16)['passed']
    assert not control.gpu_cost(394, 300, 16)['passed']
