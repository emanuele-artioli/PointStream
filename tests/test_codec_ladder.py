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
    scores = dict(frames=2, metric_control={'source_self_lpips': 0., 'source_self_dists': 0.},
                  gpu_execution={'device_count': 1, 'network_parameters_cuda': True, 'maximum_reserved_bytes': 1000}, config={'rates': [35, 50, 65]}, region_pixels={'foreground': [10, 0], 'background': [10, 20]},
                  rows=[{**copy.deepcopy(row), 'arm': arm, 'crf': crf} for arm in ('native', 'roi', 'semantic_patch', 'error_patch') for crf in (35, 50, 65)])
    assert all(validate_scores(scores).values())
    bad = copy.deepcopy(scores)
    bad['rows'][0]['frames']['full_lpips'][0] = float('nan')
    assert not all(validate_scores(bad).values())
    bad = copy.deepcopy(scores)
    bad['rows'][0]['frames']['weighted_dists'][0] = .1
    assert not all(validate_scores(bad).values())
    bad = copy.deepcopy(scores)
    bad['metric_control']['source_self_dists'] = .1
    assert not all(validate_scores(bad).values())


def test_prepared_archive_binding_and_tamper_guard(tmp_path):
    import hashlib
    import tarfile
    from experiments.codec_ladder.fleet import prepared_inputs
    source = tmp_path / 'source.yuv'
    source.write_bytes(bytes(128 * 128 * 3 // 2 * 2))
    masks = tmp_path / 'masks.npz'
    np.savez_compressed(masks, foreground=np.zeros((2, 128, 128), bool))
    cfg = config()
    info = dict(width=128, height=128, frames=2, fps='25/1', source_group=cfg['source_group'],
                source_record={'full_range': False}, mask_record={'provenance': cfg['annotation_provenance']},
                outputs={p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (source, masks)})
    meta = tmp_path / 'preparation.json'
    meta.write_text(json.dumps(info))
    archive = tmp_path / 'prepared.tar'
    with tarfile.open(archive, 'w') as bundle:
        for path in (source, masks, meta):
            bundle.add(path, arcname=path.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    a, b, manifest, binding = prepared_inputs(archive, digest, tmp_path / 'out', cfg)
    assert a.read_bytes() == source.read_bytes() and b.exists() and manifest.exists()
    assert binding['archive_sha256'] == digest
    with pytest.raises(ValueError, match='identity'):
        prepared_inputs(archive, '0' * 64, tmp_path / 'bad', cfg)
    with pytest.raises(ValueError, match='configuration'):
        prepared_inputs(archive, digest, tmp_path / 'bad-config', {**cfg, 'source_group': 'wrong'})


@pytest.mark.skipif(not all(shutil.which(t) for t in ['SvtAv1EncApp', 'dav1d']), reason='native tools absent')
def test_fleet_publication_lifecycle_keeps_thin_archive(tmp_path, monkeypatch):
    from experiments.codec_ladder import fleet
    from experiments.jobs import staging
    from src.codecs import svtav1
    source = tmp_path / 'input.yuv'
    source.write_bytes(np.random.default_rng(31).integers(16, 235, (2, 128 * 128 * 3 // 2), np.uint8).tobytes())
    masks = tmp_path / 'masks.npz'
    foreground = np.zeros((2, 128, 128), bool)
    foreground[:, 32:96, 32:96] = True
    np.savez_compressed(masks, foreground=foreground)
    manifest = tmp_path / 'manifest.json'
    prepare(config(), source, masks, manifest)
    stage, scratch = tmp_path / 'stage', tmp_path / 'scratch'
    stage.mkdir()
    scratch.mkdir()
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    # Local native tools are infrastructure only; production version gating
    # remains separately tested and is not bypassed in any fleet specification.
    monkeypatch.setattr(fleet, 'qualify_tools', lambda e, d: {'encoder': svtav1.tool(e), 'decoder': svtav1.tool(d)})
    monkeypatch.setattr('sys.argv', ['fleet', '--mode', 'run', '--manifest', str(manifest), '--source', str(source),
                                  '--masks', str(masks), '--frames', '2', '--target-frames', '2', '--full-cap-seconds', '60'])
    assert fleet.main() == 0
    saved_sha = fleet.worker.file_sha(stage / 'published.tar')
    assert staging.publish_scratch(scratch, stage) is None
    staging.release_scratch(scratch, stage)
    assert not scratch.exists()
    assert fleet.worker.file_sha(stage / 'published.tar') == saved_sha
    assert fleet.validate(stage, scratch)['passed']
