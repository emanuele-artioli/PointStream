from fractions import Fraction

import numpy as np
import pytest

from experiments.jobs import inbox, staging
from experiments.tennis import ball_pixel_oracle as oracle, ball_roi_control as common
from src.pointstream import ball, ball_stream


def test_even_clamped_transmitted_positions_and_absence():
    assert oracle.placement(None)['present'] is False
    assert oracle.placement((0, 0))['left'] == 0
    assert oracle.placement((1279, 719))['left'] == 1248
    assert oracle.placement((1279, 719))['top'] == 688
    row = oracle.placement((101, 103))
    assert row['left'] % 2 == 0 and row['top'] % 2 == 0
    with pytest.raises(ValueError):
        oracle.placement((1280, 0))


def test_padded_yuv_crop_and_absent_frame(tmp_path):
    source = tmp_path / 'source.yuv'
    with source.open('wb') as out:
        out.truncate(48 * common.FRAME_BYTES)
    raw = np.memmap(source, mode='r+', dtype=np.uint8, shape=(48, common.FRAME_BYTES))
    raw[:] = 0
    ysize = common.W * common.H
    raw[0, :ysize] = 73
    raw[0, ysize:ysize + ysize // 4] = 99
    raw[0, ysize + ysize // 4:] = 151
    raw.flush()
    del raw
    patches, rows = oracle.patch_frames(source, [(100, 100), None] + [None] * 46, 16)
    y = patches[0, :4096].reshape(64, 64)
    u = patches[0, 4096:5120].reshape(32, 32)
    v = patches[0, 5120:].reshape(32, 32)
    assert np.all(y[16:48, 16:48] == 73) and y[0, 0] == 16
    assert np.all(u[8:24, 8:24] == 99) and u[0, 0] == 128
    assert np.all(v[8:24, 8:24] == 151) and v[0, 0] == 128
    assert np.all(patches[1, :4096] == 16) and np.all(patches[1, 4096:] == 128)
    assert not rows[1]['present'] and len(rows) == 16


def test_compose_pastes_only_visible_inner_patch():
    base = np.zeros((2, common.H, common.W, 3), dtype=np.uint8)
    patches = np.full((2, 64, 64, 3), 255, dtype=np.uint8)
    rows = [dict(oracle.placement((100, 100)), frame=0), dict(oracle.placement(None), frame=1)]
    result = oracle.compose(base, patches, rows)
    assert np.count_nonzero(result[0]) == 32 * 32 * 3
    assert np.count_nonzero(result[1]) == 0
    assert np.count_nonzero(base) == 0


def test_positions_are_bound_to_entire_psbl():
    track = [None] * 48
    payload = ball.encode_frames(track, [None] * 48, common.W, common.H)
    stream = ball_stream.Stream(common.W, common.H, Fraction(30, 1), 48, 'svt-av1-ivf', 'unit', b'native', payload)
    point = {'resource_id': 'unit', 'ball_payload_hex': payload.hex(),
             'wire_sha256': common.digest(ball_stream.pack(stream))}
    assert oracle.reconstruct_packet(point, b'native').ball_payload == payload
    with pytest.raises(ValueError):
        oracle.reconstruct_packet(point, b'other')


def test_thin_archive_survives_fleet_scratch_release(tmp_path):
    scratch, stage = tmp_path / 'scratch', tmp_path / 'stage'
    root = scratch / 'pixel-private' / 'publication'
    root.mkdir(parents=True)
    stage.mkdir()
    for name in oracle.members():
        (root / name).write_text(name)
    archive = stage / 'published.tar'
    binding = oracle.publish(root, archive)
    assert staging.publish_scratch(scratch, stage) is None
    staging.release_scratch(scratch, stage)
    assert not scratch.exists()
    extracted = oracle.unpack(archive, binding, tmp_path / 'replayed')
    assert {item.name for item in extracted.iterdir()} == set(oracle.members())
    with pytest.raises(ValueError):
        oracle.unpack(archive, dict(binding, sha256='0' * 64), tmp_path / 'bad')


def test_fleet_specs_and_no_prefix_analysis():
    cpu = inbox.validate_spec(oracle.make_spec('2026-11-16T12:00:00+00:00'))
    gpu = inbox.validate_spec(oracle.score_spec('2026-11-16T12:00:00+00:00', '/data/a', 'a' * 64, '/data/b', 'b' * 64))
    assert cpu['cpu_threads'] == 1 and cpu['cpu_affinity_index'] == 0
    assert gpu['gpu_sharing']['mode'] == 'shared' and gpu['gpu_memory_mib'] == 8000
    assert gpu['gpu_sharing']['margin_mib'] == 8192
    assert gpu['arguments'][:3] == ['--fleet', '--mode', 'score']
    with pytest.raises(ValueError):
        oracle.analyse({'frames': 16, 'available_receiver_frames': 48, 'metric_control': {'passed': True}})
