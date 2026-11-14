from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.codec_ladder import transport
from experiments.codec_ladder.__main__ import prepare, select_box, validate_config, run, validate


def config():
    return dict(domain='tennis', source_group='synthetic-only', annotation_provenance='synthetic-not-evidence',
                width=128, height=128, fps=[25, 1], frames=2, full_range=False,
                rates=[35, 50, 65], preset=12, patch_size=[64, 64], patch_crf=28, roi_outside_offset=20)


def native(width=128, height=128):
    payload = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', width, height, 25, 1, 2, 0)
    for t in range(2):
        payload += struct.pack('<IQ', 3, t) + b'abc'
    return payload


def metadata():
    return dict(profile='static-pixel-patch-v1', box=[0, 0, 64, 64], native=dict(
        version=1, profile='native-av1', width=128, height=128, fps=[25, 1],
        frames=2, full_range=False, decoder_sha256='0' * 64))


def test_patch_packet_roundtrip_and_all_bytes_count():
    meta = metadata()
    packet = transport.pack(meta, native(), native(64, 64))
    assert transport.unpack(packet) == (meta, native(), native(64, 64))
    assert len(packet) > len(native()) + len(native(64, 64))
    for bad in (packet + b'x', packet[:-1], packet[:30] + b'x' + packet[31:]):
        with pytest.raises(ValueError):
            transport.unpack(bad)


@pytest.mark.parametrize('box', [[1, 0, 64, 64], [0, 0, 32, 64], [-2, 0, 64, 64], [66, 0, 64, 64]])
def test_packet_rejects_impossible_placement(box):
    with pytest.raises(ValueError):
        transport.pack({**metadata(), 'box': box}, native(), native(64, 64))


def test_exact_planar_crop_and_composite():
    base = np.arange(128 * 128 * 3 // 2, dtype=np.uint8)
    box = [32, 48, 64, 64]
    cropped = np.frombuffer(transport.crop(base, 128, 128, box), np.uint8)
    reconstructed = np.frombuffer(transport.composite(np.zeros_like(base), cropped, 128, 128, box), np.uint8)
    for original, output, scale in zip(transport.planes(base, 128, 128), transport.planes(reconstructed, 128, 128), (1, 2, 2)):
        x, y, w, h = [v // scale for v in box]
        np.testing.assert_array_equal(output[y:y + h, x:x + w], original[y:y + h, x:x + w])
        assert not output[:y].any()


def test_content_based_even_rectangle_and_first_tie():
    importance = np.zeros((128, 128))
    importance[64:, 64:] = 1
    assert select_box(importance, [64, 64]) == [64, 64, 64, 64]
    assert select_box(np.zeros((128, 128)), [64, 64]) == [0, 0, 64, 64]
    with pytest.raises(ValueError):
        select_box(np.full((128, 128), np.nan), [64, 64])


def test_prepare_identity_and_validation(tmp_path):
    source, masks, manifest = tmp_path / 'input.yuv', tmp_path / 'masks.npz', tmp_path / 'input.json'
    source.write_bytes(bytes(128 * 128 * 3))
    np.savez_compressed(masks, foreground=np.zeros((2, 128, 128), bool))
    record = prepare(config(), source, masks, manifest)
    assert record['mask_coverage']['empty_frames'] == 2
    assert json.loads(manifest.read_text())['source']['sha256']
    with pytest.raises(FileExistsError):
        prepare(config(), source, masks, manifest)
    with pytest.raises(ValueError):
        validate_config({**config(), 'rates': [35, 35, 50]})
    with pytest.raises(ValueError):
        validate_config({**config(), 'unknown': True})


@pytest.mark.skipif(not all(shutil.which(t) for t in ['SvtAv1EncApp', 'dav1d']), reason='native tools absent')
def test_native_billed_screen_and_packet_only_receiver(tmp_path: Path):
    # Real tiny native codec correctness, not scientific ROI/patch evidence.
    source = tmp_path / 'input.yuv'
    rng = np.random.default_rng(19)
    data = rng.integers(16, 235, (2, 128 * 128 * 3 // 2), dtype=np.uint8)
    source.write_bytes(data.tobytes())
    masks = tmp_path / 'masks.npz'
    foreground = np.zeros((2, 128, 128), bool)
    foreground[:, 64:, 64:] = True
    np.savez_compressed(masks, foreground=foreground)
    manifest = tmp_path / 'input.json'
    prepare(config(), source, masks, manifest)
    root = tmp_path / 'run'
    result = run(manifest, root)
    assert len(result['rates']) == 12
    assert validate(root)['passed']
    source.unlink()
    # Decoder has no source argument and runs after the original is gone.
    chosen = root / 'semantic_patch-50.plp'
    decoded = tmp_path / 'fresh.yuv'
    receipt = transport.decode(chosen.read_bytes(), decoded, scratch=tmp_path)
    assert receipt['decoded_sha256'] == next(r['receipt']['decoded_sha256'] for r in result['rates'] if r['arm'] == 'semantic_patch' and r['crf'] == 50)
    assert decoded.stat().st_size == 128 * 128 * 3
    from experiments.codec_ladder.fleet import archive_run, extract_archive, restore_decoded
    bundle = tmp_path / 'published.tar'
    archive = archive_run(root, bundle)
    assert not any(name.endswith('.yuv') or name.endswith('.ivf') for name in archive['members'])
    recovered = extract_archive(bundle, archive['sha256'], tmp_path / 'recovered')
    assert all(restore_decoded(recovered, 'dav1d').values())
    assert validate(recovered)['passed']
    changed = root / 'roi-35.plp'
    changed.write_bytes(changed.read_bytes() + b'tampered')
    with pytest.raises(ValueError):
        validate(root)


def test_cost_gate_is_complete_scaled_wall_without_intercept():
    from experiments.codec_ladder.fleet import cost_record
    assert cost_record(wall=10, cpu=8, frames=16, target=64, cap=100, overhead=30)['forecast_seconds'] == 90
    assert cost_record(wall=10, cpu=8, frames=16, target=64, cap=89, overhead=30)['passed'] is False
    assert cost_record(wall=10, cpu=8, frames=64, target=64, cap=11, overhead=30)['passed'] is True
    with pytest.raises(ValueError):
        cost_record(wall=10, cpu=8, frames=64, target=16, cap=100, overhead=30)


def test_production_tool_version_gate(monkeypatch):
    from experiments.codec_ladder.fleet import qualify_tools
    from src.codecs import svtav1
    monkeypatch.setattr(svtav1, 'tool', lambda name: dict(version='SVT-AV1 v4.2.0' if name == 'encoder' else '1.5.0', sha256='0' * 64))
    assert qualify_tools('encoder', 'decoder')['decoder']['version'] == '1.5.0'
    monkeypatch.setattr(svtav1, 'tool', lambda name: dict(version='SVT-AV1 v4.0.1', sha256='0' * 64))
    with pytest.raises(ValueError, match='4.2.0'):
        qualify_tools('encoder', 'decoder')


def test_scalar_validator_rejects_corruption_and_preserves_empty_regions():
    import copy
    from experiments.codec_ladder.fleet import validate_scores
    frames = {'full_lpips': [.2, .2], 'foreground_lpips': [.1, None], 'background_lpips': [.3, .3],
              'weighted_lpips': [.16, .3], 'full_dists': [.2, .2], 'foreground_dists': [.1, None],
              'background_dists': [.3, .3], 'weighted_dists': [.16, .3], 'full_rgb_psnr': [30., 30.]}
    row = dict(arm='native', crf=35, frames=frames,
               means={name: sum(v for v in values if v is not None) / sum(v is not None for v in values) for name, values in frames.items()},
               temporal=dict(pairs=1, values={'full_residual_mae': [0.], 'foreground_residual_mae': [0.]}))
    scores = dict(frames=2, config={'rates': [35, 50, 65]}, region_pixels={'foreground': [10, 0], 'background': [10, 20]},
                  rows=[{**copy.deepcopy(row), 'arm': arm, 'crf': crf} for arm in ('native', 'roi', 'semantic_patch', 'error_patch') for crf in (35, 50, 65)])
    assert all(validate_scores(scores).values())
    bad = copy.deepcopy(scores)
    bad['rows'][0]['frames']['full_lpips'][0] = float('nan')
    assert not all(validate_scores(bad).values())
    bad = copy.deepcopy(scores)
    bad['rows'][0]['frames']['weighted_dists'][0] = .1
    assert not all(validate_scores(bad).values())
